"""Derive static JAX rollout group specs from training env budget and format weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.config.schema import TrainConfig

_SUPPORTED_PLAYER_COUNTS = frozenset({2, 4})
_DEFAULT_GROUP_NAMES = {2: "two_player", 4: "four_player"}


@dataclass(frozen=True, slots=True)
class RolloutGroupSpec:
    """One statically compiled JAX rollout collector."""

    name: str
    player_count: int
    num_envs: int


def infer_static_format_weights(cfg: TrainConfig) -> dict[int, float]:
    """Return normalized static format weights from training config."""

    raw = dict(cfg.training.format_weights or {})
    if not raw:
        return {int(cfg.task.player_count): 1.0}
    return normalize_format_weights(raw)


def normalize_format_weights(weights: dict[int, float]) -> dict[int, float]:
    """Normalize weights for supported player counts."""

    filtered: dict[int, float] = {}
    for key, value in weights.items():
        player_count = int(key)
        if player_count not in _SUPPORTED_PLAYER_COUNTS:
            raise ValueError(
                "training.format_weights keys must be player counts 2 or 4, "
                f"got {player_count}."
            )
        weight = float(value)
        if weight < 0.0 or not isfinite(weight):
            raise ValueError(
                f"training.format_weights[{player_count}] must be finite and non-negative."
            )
        filtered[player_count] = weight
    total = sum(filtered.values())
    if total <= 0.0:
        raise ValueError("training.format_weights must sum to > 0.")
    return {player_count: weight / total for player_count, weight in filtered.items()}


def curriculum_compile_player_counts(cfg: TrainConfig) -> set[int]:
    """Player counts that need compiled collectors across curriculum + static weights."""

    counts: set[int] = set()
    static = infer_static_format_weights(cfg)
    counts.update(pc for pc, weight in static.items() if weight > 0.0)

    curriculum = cfg.curriculum
    if bool(curriculum.enabled):
        for stage in curriculum.stages or []:
            if not isinstance(stage, dict):
                continue
            raw_weights = dict(stage.get("format_weights", {}) or {})
            if not raw_weights:
                counts.update(pc for pc, weight in static.items() if weight > 0.0)
                continue
            normalized = normalize_format_weights(
                {int(key): float(value) for key, value in raw_weights.items()}
            )
            counts.update(pc for pc, weight in normalized.items() if weight > 0.0)
    return counts


def allocate_split(total_envs: int, weights: dict[int, float]) -> dict[int, int]:
    """Split ``total_envs`` across formats using Hamilton largest-remainder allocation."""

    if total_envs <= 0:
        raise ValueError("training.num_envs must be >= 1.")
    if not weights:
        raise ValueError("format weights must be non-empty for split allocation.")

    exact = {player_count: total_envs * weight for player_count, weight in weights.items()}
    floors = {player_count: int(value) for player_count, value in exact.items()}
    remainders = {
        player_count: exact[player_count] - float(floors[player_count])
        for player_count in weights
    }
    alloc = dict(floors)
    leftover = total_envs - sum(floors.values())
    order = sorted(weights.keys(), key=lambda player_count: (-remainders[player_count], player_count))
    for index in range(leftover):
        player_count = order[index % len(order)]
        alloc[player_count] += 1
    return alloc


def resolve_rollout_group_specs(cfg: TrainConfig) -> list[RolloutGroupSpec]:
    """Resolve rollout groups for static JAX collector initialization."""

    compile_counts = curriculum_compile_player_counts(cfg)
    if not compile_counts:
        raise ValueError("At least one rollout format must be active.")

    total_envs = int(cfg.training.num_envs)
    if total_envs <= 0:
        raise ValueError("training.num_envs must be >= 1.")

    static_weights = infer_static_format_weights(cfg)
    active_weights = {
        player_count: static_weights[player_count]
        for player_count in sorted(compile_counts)
        if player_count in static_weights and static_weights[player_count] > 0.0
    }
    if not active_weights:
        active_weights = {
            player_count: 1.0 / float(len(compile_counts))
            for player_count in sorted(compile_counts)
        }

    if cfg.training.rotate_format_rollouts:
        alloc = {player_count: total_envs for player_count in compile_counts}
    else:
        split_weights = normalize_format_weights(active_weights)
        if total_envs < len(compile_counts):
            raise ValueError(
                f"training.num_envs={total_envs} too small for split mode with "
                f"{len(compile_counts)} active formats (need >= {len(compile_counts)})."
            )
        alloc = allocate_split(total_envs, split_weights)

    specs: list[RolloutGroupSpec] = []
    for player_count in sorted(compile_counts):
        num_envs = int(alloc.get(player_count, 0))
        if num_envs < 1:
            raise ValueError(
                f"Resolved {player_count}p rollout group has num_envs={num_envs}; "
                "increase training.num_envs or adjust format weights."
            )
        specs.append(
            RolloutGroupSpec(
                name=_DEFAULT_GROUP_NAMES.get(player_count, f"{player_count}p"),
                player_count=player_count,
                num_envs=num_envs,
            )
        )
    return specs


def rollout_player_counts(cfg: TrainConfig) -> list[int]:
    """Sorted player counts with compiled rollout groups."""

    return sorted({spec.player_count for spec in resolve_rollout_group_specs(cfg)})


def run_name_env_count(cfg: TrainConfig) -> int:
    """Env count label for run naming and throughput helpers."""

    if cfg.training.rotate_format_rollouts:
        return int(cfg.training.num_envs)
    return sum(spec.num_envs for spec in resolve_rollout_group_specs(cfg))


def validate_rollout_allocation(cfg: TrainConfig) -> None:
    """Validate rollout allocation and microbatch divisibility at compose time."""

    specs = resolve_rollout_group_specs(cfg)
    microbatch = cfg.training.rollout_microbatch_envs
    if microbatch is None:
        return
    microbatch_envs = int(microbatch)
    if microbatch_envs <= 0:
        return
    for spec in specs:
        if microbatch_envs > spec.num_envs:
            raise ValueError(
                "training.rollout_microbatch_envs must be <= each rollout group's num_envs "
                f"({spec.player_count}p has {spec.num_envs})."
            )
        if spec.num_envs % microbatch_envs != 0:
            raise ValueError(
                "training.rollout_microbatch_envs must evenly divide each rollout group's "
                f"num_envs ({spec.player_count}p has {spec.num_envs})."
            )


def validate_curriculum_format_weights(cfg: TrainConfig) -> None:
    """Validate optional curriculum stage format weight maps."""

    if not cfg.curriculum.enabled:
        return
    for index, stage in enumerate(cfg.curriculum.stages or []):
        if not isinstance(stage, dict):
            continue
        raw_weights = dict(stage.get("format_weights", {}) or {})
        if not raw_weights:
            continue
        try:
            normalize_format_weights(
                {int(key): float(value) for key, value in raw_weights.items()}
            )
        except ValueError as exc:
            raise ValueError(
                f"curriculum.stages[{index}].format_weights invalid: {exc}"
            ) from exc
