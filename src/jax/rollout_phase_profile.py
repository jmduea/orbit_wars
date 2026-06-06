"""Offline rollout phase profiler (admission-shaped geometry, short run)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
import jax.random

import jax
from src.config import (
    TrainConfig,
    compose_hydra_train_config,
    train_config_from_omegaconf,
)
from src.jax.device import ensure_jax_accelerator_backend
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import concatenate_transition_batches, ppo_update_jax
from src.jax.rollout.phase_timing import ROLLOUT_PHASE_TIMING_KEYS
from src.jax.rollout.phase_timing_report import (
    PhaseTimingWindow,
    extract_rollout_phase_breakdown_from_records,
    format_rollout_phase_breakdown,
)
from src.jax.train import init_train_state
from src.jax.train.metrics import (
    finalize_rollout_phase_timing_metrics,
    sum_metric_dicts,
)
from src.jax.train.rollout_groups import (
    active_group_indices,
    init_profile_rollout_groups,
    replace_rollout_group_state,
)
from src.jax.train.snapshots import init_historical_snapshot_pool
from src.training.curriculum import CurriculumController

ADMISSION_PROFILE_OVERRIDES: tuple[str, ...] = (
    "model=transformer_factorized_small",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "telemetry.metric_groups.action_decision=true",
    "task=shield_cheap",
    "seed=42",
    "model.max_moves_k=2",
    "training=2p4p_32_split",
    "training.rollout_steps=256",
    "task.candidate_count=3",
    "opponents=noop_only",
    "curriculum=off",
    "artifacts.replay.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
)

# Host-timed collect syncs per rollout step; full admission geometry (32×256) can
# stall 30+ min on first compile. Quick mode keeps model/task/opponents but shrinks
# envs/steps so phase fractions are interactive (not throughput-comparable).
QUICK_GEOMETRY_OVERRIDES: tuple[str, ...] = ("training=smoke",)


@dataclass(frozen=True, slots=True)
class RolloutPhaseProfileResult:
    overrides: tuple[str, ...]
    warmup: int
    updates: int
    measured_updates: int
    per_update_records: tuple[dict[str, object], ...]
    breakdown: dict[str, object]
    seconds_total: float


def resolve_profile_overrides(
    *,
    preset: str | None,
    extra_overrides: Sequence[str] = (),
    updates: int,
    model: str | None = None,
    quick: bool = True,
) -> tuple[str, ...]:
    if preset == "admission":
        base = list(ADMISSION_PROFILE_OVERRIDES)
        if quick:
            base = [
                item
                for item in base
                if not item.startswith("training=2p4p_32_split")
                and item != "training.rollout_steps=256"
            ]
            base.extend(QUICK_GEOMETRY_OVERRIDES)
        if model is not None:
            base = [f"model={model}", *base[1:]]
        base.append(f"training.total_updates={int(updates)}")
        base.extend(str(item) for item in extra_overrides)
        return tuple(dict.fromkeys(base))
    if preset is not None:
        raise ValueError(f"unknown profile preset {preset!r}; supported: admission")
    overrides = [f"training.total_updates={int(updates)}"]
    if model is not None:
        overrides.insert(0, f"model={model}")
    overrides.extend(str(item) for item in extra_overrides)
    return tuple(dict.fromkeys(overrides))


def compose_profile_config(
    *,
    preset: str | None = "admission",
    extra_overrides: Sequence[str] = (),
    updates: int = 5,
    model: str | None = None,
    quick: bool = True,
) -> TrainConfig:
    overrides = resolve_profile_overrides(
        preset=preset,
        extra_overrides=extra_overrides,
        updates=updates,
        model=model,
        quick=quick,
    )
    return train_config_from_omegaconf(compose_hydra_train_config(list(overrides)))


def _emit_profile_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _phase_record_from_metrics(
    *,
    update: int,
    rollout_seconds: float,
    metrics: Mapping[str, jax.Array],
) -> dict[str, object]:
    host = jax.device_get(
        {key: metrics[key] for key in ROLLOUT_PHASE_TIMING_KEYS if key in metrics}
    )
    record: dict[str, object] = {
        "update": update,
        "rollout_seconds": rollout_seconds,
    }
    record.update({key: float(host[key]) for key in host})
    return record


def run_rollout_phase_profile(
    cfg: TrainConfig,
    *,
    warmup: int = 2,
    updates: int | None = None,
    window: PhaseTimingWindow | None = None,
) -> RolloutPhaseProfileResult:
    """Short in-process train loop using host-timed rollout collect only."""

    ensure_jax_accelerator_backend()
    resolved_updates = int(
        updates if updates is not None else cfg.training.total_updates
    )
    resolved_window = window or PhaseTimingWindow(
        warmup=int(warmup),
        max_measured_update=min(resolved_updates, 20),
    )
    started = time.perf_counter()

    key = jax.random.PRNGKey(cfg.seed)
    _, rollout_init_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    key, rollout_groups = init_profile_rollout_groups(rollout_init_key, cfg, policy)
    historical_pool = init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    update_fn = jax.jit(lambda ts, tr: ppo_update_jax(ts, policy, tr, cfg))

    per_update_records: list[dict[str, object]] = []
    total_iterations = int(warmup) + resolved_updates
    _emit_profile_progress(
        "rollout-phase-profile: init done; starting updates "
        f"(warmup={warmup}, total={resolved_updates}, "
        f"envs={cfg.training.num_envs}, rollout_steps={cfg.training.rollout_steps})"
    )

    for iteration in range(total_iterations):
        update = iteration + 1
        update_start = time.perf_counter()
        _emit_profile_progress(
            f"rollout-phase-profile: update={update} collect starting "
            "(first update may compile several minutes on GPU)"
        )
        stage_view = curriculum.stage_view(
            update,
            snapshot_ids=historical_pool.snapshot_ids,
            snapshot_valid_mask=historical_pool.valid_mask,
            snapshot_updates=historical_pool.snapshot_updates,
        )
        active_indices = active_group_indices(
            rollout_groups,
            curriculum.current_format_weights(),
            update=update,
            rotate_format_rollouts=cfg.training.rotate_format_rollouts,
        )
        key, *rollout_keys = jax.random.split(key, len(active_indices) + 1)
        rollout_start = time.perf_counter()
        transitions_by_group = []
        rollout_metrics_by_group = []
        next_groups = []
        for group_idx, rollout_key in zip(active_indices, rollout_keys, strict=True):
            group = rollout_groups[group_idx]
            (
                _,
                env_state,
                turn_batch,
                transitions,
                rollout_metrics,
            ) = group.collect_fn(
                rollout_key,
                group.env_state,
                group.turn_batch,
                train_state,
                stage_view,
                historical_pool.params,
                jnp.asarray(update, dtype=jnp.int32),
            )
            next_groups.append(
                replace_rollout_group_state(group, env_state, turn_batch)
            )
            transitions_by_group.append(transitions)
            rollout_metrics_by_group.append(rollout_metrics)
        merged_groups = list(rollout_groups)
        for group_idx, updated_group in zip(active_indices, next_groups, strict=True):
            merged_groups[group_idx] = updated_group
        rollout_groups = merged_groups
        rollout_seconds = time.perf_counter() - rollout_start

        transitions = concatenate_transition_batches(transitions_by_group)
        merged_metrics = finalize_rollout_phase_timing_metrics(
            sum_metric_dicts(rollout_metrics_by_group)
        )
        train_state, _ = update_fn(train_state, transitions)
        update_seconds = time.perf_counter() - update_start
        _emit_profile_progress(
            f"rollout-phase-profile: update={update} done "
            f"rollout_s={rollout_seconds:.1f} update_s={update_seconds:.1f}"
        )

        if iteration >= warmup:
            per_update_records.append(
                _phase_record_from_metrics(
                    update=update,
                    rollout_seconds=rollout_seconds,
                    metrics=merged_metrics,
                )
            )

    breakdown = extract_rollout_phase_breakdown_from_records(
        per_update_records,
        window=resolved_window,
    )
    breakdown["profile_seconds_total"] = time.perf_counter() - started
    return RolloutPhaseProfileResult(
        overrides=tuple(),
        warmup=int(warmup),
        updates=resolved_updates,
        measured_updates=len(per_update_records),
        per_update_records=tuple(per_update_records),
        breakdown=breakdown,
        seconds_total=float(breakdown["profile_seconds_total"]),
    )


def profile_result_payload(
    result: RolloutPhaseProfileResult,
    *,
    overrides: Sequence[str],
    preset: str | None,
) -> dict[str, object]:
    payload = dict(result.breakdown)
    payload.update(
        {
            "preset": preset,
            "overrides": list(overrides),
            "warmup": result.warmup,
            "updates": result.updates,
            "measured_updates": result.measured_updates,
            "seconds_total": result.seconds_total,
        }
    )
    return payload


def format_profile_report(payload: Mapping[str, object]) -> str:
    geometry_note = payload.get("geometry_mode", "quick")
    lines = [
        "Rollout phase profile (offline; phase shares diagnostic — not JIT throughput)",
        f"  preset: {payload.get('preset')}  geometry: {geometry_note}",
        f"  measured updates: {payload.get('measured_updates')} "
        f"(warmup={payload.get('warmup')}, total={payload.get('updates')})",
        f"  wall time: {float(payload.get('seconds_total', 0.0)):.1f}s",
        "",
        format_rollout_phase_breakdown(payload),
    ]
    return "\n".join(lines)
