from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Observed accelerator profile used for first-pass throughput sizing."""

    accelerator_id: str
    gpu_name: str
    memory_gb: float


def estimate_training_overrides(
    hardware: HardwareProfile,
    model_config: Mapping[str, object],
    task_config: Mapping[str, object],
    *,
    hydra_overrides: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Estimate stable high-throughput Hydra overrides for a Kaggle worker.

    The estimator is intentionally conservative. Calibration sweeps should probe
    around these values before a long run.
    """

    memory_gb = max(float(hardware.memory_gb), 1.0)
    hidden_size = int(model_config.get("hidden_size", 128) or 128)
    layers = int(model_config.get("planet_transformer_layers", 2) or 2)
    history = int(task_config.get("feature_history_steps", 1) or 1)
    shield_horizon = int(task_config.get("trajectory_shield_horizon", 10) or 10)

    model_scale = max(hidden_size / 128.0, 1.0) * max(layers / 2.0, 1.0)
    horizon_scale = max(history / 2.0, 1.0) * max(shield_horizon / 10.0, 1.0)
    pressure = model_scale * horizon_scale

    env_budget = int(memory_gb * 2.0 / pressure)
    num_envs = _round_to_multiple(max(min(env_budget, 96), 8), 8)
    group_counts = rollout_group_env_counts(hydra_overrides, default_num_envs=num_envs)
    fixes_groups = format_fixes_rollout_group_envs(hydra_overrides)
    per_group_envs = min(group_counts)
    sizing_envs = sum(group_counts) if fixes_groups else num_envs

    microbatch_envs = _round_to_multiple(max(min(per_group_envs // 2, 32), 4), 4)
    microbatch_envs = largest_compatible_microbatch(microbatch_envs, group_counts)
    rollout_steps = 256 if memory_gb >= 24 and pressure <= 3.0 else 128
    minibatch_size = _round_to_multiple(
        max(min(sizing_envs * rollout_steps // 4, 4096), 256),
        128,
    )
    chunk_max = max(minibatch_size * 2, 1024)

    overrides: list[str] = [
        f"training.rollout_microbatch_envs={microbatch_envs}",
        f"training.rollout_steps={rollout_steps}",
        f"training.minibatch_size={minibatch_size}",
        f"training.update_chunk_rows_min={min(chunk_max, 2048)}",
        f"training.update_chunk_rows_max={chunk_max}",
    ]
    if not fixes_groups:
        overrides.insert(0, f"training.num_envs={num_envs}")
    return finalize_rollout_shape_overrides(tuple(overrides), hydra_overrides)


def calibration_grid(
    base_overrides: tuple[str, ...],
    *,
    hydra_overrides: Sequence[str] | None = None,
) -> list[tuple[str, ...]]:
    """Generate a bounded calibration grid around estimator output."""

    parsed = dict(item.split("=", 1) for item in base_overrides)
    rollout_steps = int(parsed["training.rollout_steps"])
    group_counts = rollout_group_env_counts(hydra_overrides)
    fixes_groups = format_fixes_rollout_group_envs(hydra_overrides)

    if fixes_groups:
        total_envs = sum(group_counts)
        per_group = min(group_counts)
        micro_candidates = _compatible_microbatch_candidates(group_counts, per_group)
        variants: list[tuple[str, ...]] = []
        for micro in micro_candidates:
            minibatch = _round_to_multiple(
                max(total_envs * rollout_steps // 4, 128),
                128,
            )
            variant = _replace_override(
                _replace_override(
                    base_overrides,
                    "training.rollout_microbatch_envs",
                    micro,
                ),
                "training.minibatch_size",
                minibatch,
            )
            variants.append(finalize_rollout_shape_overrides(variant, hydra_overrides))
        return variants

    num_envs = int(parsed["training.num_envs"])
    variants = []
    for env_multiplier in (0.5, 1.0, 1.5):
        envs = _round_to_multiple(max(int(num_envs * env_multiplier), 4), 4)
        micro = _round_to_multiple(max(min(envs // 2, 32), 4), 4)
        micro = largest_compatible_microbatch(micro, [envs])
        minibatch = _round_to_multiple(max(envs * rollout_steps // 4, 128), 128)
        variant = tuple(
            _replace_override(
                _replace_override(
                    _replace_override(base_overrides, "training.num_envs", envs),
                    "training.rollout_microbatch_envs",
                    micro,
                ),
                "training.minibatch_size",
                minibatch,
            )
        )
        variants.append(finalize_rollout_shape_overrides(variant, hydra_overrides))
    return variants


def rollout_group_env_counts(
    hydra_overrides: Sequence[str] | None,
    *,
    default_num_envs: int = 8,
) -> list[int]:
    """Return active per-group env counts for the resolved Hydra config."""

    if not hydra_overrides:
        return [default_num_envs]

    from src.config import compose_hydra_train_config

    cfg = compose_hydra_train_config(list(hydra_overrides))
    counts: list[int] = []
    for group in cfg.format.rollout_groups:
        num_envs = int(group.get("num_envs", cfg.training.num_envs))
        if num_envs > 0:
            counts.append(num_envs)
    if counts:
        return counts
    return [int(cfg.training.num_envs)]


def format_fixes_rollout_group_envs(hydra_overrides: Sequence[str] | None) -> bool:
    """Return True when format YAML owns per-group env counts."""

    if not hydra_overrides:
        return False

    from src.config import compose_hydra_train_config

    cfg = compose_hydra_train_config(list(hydra_overrides))
    return any(int(group.get("num_envs", 0)) > 0 for group in cfg.format.rollout_groups)


def largest_compatible_microbatch(
    requested: int, group_env_counts: Sequence[int]
) -> int:
    """Pick the largest microbatch <= requested that divides every group env count."""

    if not group_env_counts:
        return max(int(requested), 1)
    min_group = min(int(count) for count in group_env_counts)
    candidate = min(max(int(requested), 1), min_group)
    while candidate > 0:
        if all(int(count) % candidate == 0 for count in group_env_counts):
            return candidate
        candidate -= 1
    return 1


def finalize_rollout_shape_overrides(
    overrides: Sequence[str],
    hydra_overrides: Sequence[str] | None,
) -> tuple[str, ...]:
    """Normalize rollout shape overrides against resolved format groups."""

    group_counts = rollout_group_env_counts(hydra_overrides)
    parsed = dict(item.split("=", 1) for item in overrides)
    normalized = list(overrides)
    if "training.rollout_microbatch_envs" in parsed:
        micro = largest_compatible_microbatch(
            int(parsed["training.rollout_microbatch_envs"]),
            group_counts,
        )
        normalized = list(
            _replace_override(
                tuple(normalized), "training.rollout_microbatch_envs", micro
            )
        )
    if format_fixes_rollout_group_envs(hydra_overrides):
        normalized = [
            item for item in normalized if not item.startswith("training.num_envs=")
        ]
    return tuple(normalized)


def _compatible_microbatch_candidates(
    group_env_counts: Sequence[int],
    per_group_envs: int,
) -> list[int]:
    raw_candidates = {
        4,
        8,
        16,
        max(per_group_envs // 2, 4),
        per_group_envs,
    }
    compatible = sorted(
        {
            largest_compatible_microbatch(candidate, group_env_counts)
            for candidate in raw_candidates
        },
        reverse=True,
    )
    return compatible[:3] or [largest_compatible_microbatch(4, group_env_counts)]


def _replace_override(
    overrides: tuple[str, ...], key: str, value: int
) -> tuple[str, ...]:
    return tuple(
        f"{key}={value}" if item.startswith(f"{key}=") else item for item in overrides
    )


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))
