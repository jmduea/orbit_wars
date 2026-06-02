"""Production-aligned short training runs with PPO + rollout scalar metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping

import jax.numpy as jnp

import jax
from src.config import TrainConfig, compose_hydra_train_config
from src.jax.benchmark import rollout_group_summary
from src.jax.device import ensure_jax_accelerator_backend
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import concatenate_transition_batches, ppo_update_jax
from src.jax.train import init_rollout_groups, init_train_state
from src.jax.train.metrics import sum_metric_dicts
from src.jax.train.rollout_groups import (
    active_group_indices,
    replace_rollout_group_state,
)
from src.jax.train.snapshots import init_historical_snapshot_pool
from src.training.curriculum import CurriculumController

UPDATE_METRIC_KEYS: tuple[str, ...] = (
    "policy_loss",
    "value_loss",
    "approx_kl",
    "entropy",
    "total_loss",
)
ROLLOUT_METRIC_KEYS: tuple[str, ...] = (
    "mean_active_launches_per_turn",
    "planet_flow_unreachable_demand_rate",
    "planet_flow_held_demand_rate",
    "planet_flow_emitted_ship_mass_rate",
    "planet_flow_small_launch_rate",
    "planet_flow_duplicate_source_target_rate",
    "planet_flow_emitted_launch_count",
    "planet_flow_control_emitted_launch_count",
    "planet_flow_control_unreachable_demand_rate",
    "planet_flow_control_held_demand_rate",
    "planet_flow_control_emitted_ship_mass_rate",
    "planet_flow_control_small_launch_rate",
    "planet_flow_control_duplicate_source_target_rate",
    "planet_flow_emitted_launch_count_delta_vs_control",
    "planet_flow_emitted_ship_mass_delta_vs_control",
    "planet_flow_unreachable_demand_rate_delta_vs_control",
    "planet_flow_held_demand_rate_delta_vs_control",
    "planet_flow_emitted_ship_mass_rate_delta_vs_control",
    "planet_flow_small_launch_rate_delta_vs_control",
    "planet_flow_duplicate_source_target_rate_delta_vs_control",
    "overall_win_rate",
)

WORKSTATION_VALIDATION_OVERRIDES: tuple[str, ...] = (
    "model=transformer_factorized",
    "training=workstation",
    "opponents=self_play_only",
    "curriculum=off",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "seed=42",
)

DEFAULT_BENCHMARK_OVERRIDES: tuple[str, ...] = (
    "model=transformer_factorized",
    "opponents=self_play_only",
    "curriculum=off",
    "seed=42",
)

PLANET_FLOW_P0_BENCHMARK_OVERRIDES: tuple[str, ...] = (
    "model=planet_flow_target_heatmap",
    "training=2p4p_16_split",
    "opponents=random_only",
    "curriculum=off",
    "artifacts=planet_flow_proof",
    "telemetry.wandb.enabled=false",
    "telemetry.metric_groups.action_decision=true",
    "seed=42",
)


@dataclass(frozen=True, slots=True)
class TrainingBenchmarkSnapshot:
    """Per-update metric capture for learning-curve analysis."""

    update: int
    curriculum_stage_id: str | None
    update_metrics: Mapping[str, float | None]
    rollout_metrics: Mapping[str, float | None]
    survival_time: float | None


@dataclass(frozen=True, slots=True)
class TrainingBenchmarkResult:
    """Aggregate outcome from a warmup + measured update training benchmark."""

    label: str
    overrides: tuple[str, ...]
    updates: int
    warmup: int
    measured_updates: int
    seconds_total: float
    seconds_per_update_mean: float
    compile_seconds_to_update_3: float | None
    devices: tuple[str, ...]
    default_backend: str
    num_envs: int
    rollout_steps: int
    update_metric_means: Mapping[str, float]
    rollout_metric_means: Mapping[str, float | None]
    snapshots: tuple[TrainingBenchmarkSnapshot, ...] = field(default_factory=tuple)
    snapshots_all_finite: bool = True
    env_steps_per_sec: float = 0.0


def _scalar_metric(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    return float(jax.device_get(metrics[key]))


def _survival_time_mean(rollout_metrics: dict) -> float | None:
    episodes = _scalar_metric(rollout_metrics, "episode_done")
    survival_sum = _scalar_metric(rollout_metrics, "survival_time_sum")
    if episodes is None or survival_sum is None:
        return None
    return survival_sum / max(episodes, 1e-9)


def _update_snapshot(
    *,
    update: int,
    update_metrics: dict,
    rollout_metrics: dict,
    curriculum: CurriculumController,
) -> TrainingBenchmarkSnapshot:
    update_values: dict[str, float | None] = {
        key: _scalar_metric(update_metrics, key) for key in UPDATE_METRIC_KEYS
    }
    rollout_values: dict[str, float | None] = {
        key: _scalar_metric(rollout_metrics, key) for key in ROLLOUT_METRIC_KEYS
    }
    return TrainingBenchmarkSnapshot(
        update=update,
        curriculum_stage_id=curriculum.current_stage_id(),
        update_metrics=update_values,
        rollout_metrics=rollout_values,
        survival_time=_survival_time_mean(rollout_metrics),
    )


def snapshot_metrics_finite(snapshot: TrainingBenchmarkSnapshot) -> bool:
    for value in (
        *snapshot.update_metrics.values(),
        *snapshot.rollout_metrics.values(),
    ):
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return False
    if snapshot.survival_time is not None and not math.isfinite(snapshot.survival_time):
        return False
    return True


def run_training_benchmark(
    cfg: TrainConfig,
    *,
    label: str,
    overrides: tuple[str, ...],
    warmup: int,
    updates: int,
    snapshot_updates: frozenset[int] | set[int] = frozenset(),
) -> TrainingBenchmarkResult:
    """Run collect + PPO on the production rollout-group path."""

    ensure_jax_accelerator_backend()
    run_started = time.perf_counter()
    devices = tuple(str(device) for device in jax.devices())
    default_backend = jax.default_backend()

    group_specs = rollout_group_summary(cfg)
    total_envs = sum(int(spec["num_envs"]) for spec in group_specs)

    key = jax.random.PRNGKey(cfg.seed)
    _, rollout_init_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    key, rollout_groups = init_rollout_groups(rollout_init_key, cfg, policy)
    historical_pool = init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    update_fn = jax.jit(lambda ts, tr: ppo_update_jax(ts, policy, tr, cfg))

    timings: list[float] = []
    update_sums = {key: 0.0 for key in UPDATE_METRIC_KEYS}
    rollout_sums = {key: 0.0 for key in ROLLOUT_METRIC_KEYS}
    rollout_seen = {key: False for key in ROLLOUT_METRIC_KEYS}
    measured = 0
    compile_seconds_to_update_3: float | None = None
    snapshot_targets = set(snapshot_updates)
    snapshots: list[TrainingBenchmarkSnapshot] = []

    for iteration in range(warmup + updates):
        update = iteration + 1
        t0 = time.perf_counter()
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
        transitions_by_group = []
        rollout_metrics_by_group = []
        next_groups = []
        for group_idx, rollout_key in zip(active_indices, rollout_keys, strict=True):
            group = rollout_groups[group_idx]
            _, env_state, turn_batch, transitions, rollout_metrics = group.collect_fn(
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

        transitions = concatenate_transition_batches(transitions_by_group)
        rollout_metrics = sum_metric_dicts(rollout_metrics_by_group)
        metrics_accum = None
        for _ in range(cfg.training.epochs):
            train_state, update_metrics = update_fn(train_state, transitions)
            metrics_accum = (
                update_metrics
                if metrics_accum is None
                else jax.tree.map(jnp.add, metrics_accum, update_metrics)
            )
        assert metrics_accum is not None
        jax.block_until_ready(metrics_accum["total_loss"])
        elapsed = time.perf_counter() - t0
        if update == 3:
            compile_seconds_to_update_3 = time.perf_counter() - run_started
        if iteration >= warmup:
            measured += 1
            timings.append(elapsed)
            for metric_key in UPDATE_METRIC_KEYS:
                if metric_key in metrics_accum:
                    update_sums[metric_key] += float(
                        jax.device_get(metrics_accum[metric_key])
                    )
            for metric_key in ROLLOUT_METRIC_KEYS:
                if metric_key in rollout_metrics:
                    rollout_seen[metric_key] = True
                    rollout_sums[metric_key] += float(
                        jax.device_get(rollout_metrics[metric_key])
                    )
        if update in snapshot_targets:
            epoch_count = max(int(cfg.training.epochs), 1)
            avg_update_metrics = jax.tree.map(
                lambda value: value / epoch_count,
                metrics_accum,
            )
            snapshots.append(
                _update_snapshot(
                    update=update,
                    update_metrics=avg_update_metrics,
                    rollout_metrics=rollout_metrics,
                    curriculum=curriculum,
                )
            )

    total_seconds = sum(timings)
    update_means = {
        key: update_sums[key] / max(measured, 1) for key in UPDATE_METRIC_KEYS
    }
    rollout_means = {
        key: (rollout_sums[key] / max(measured, 1) if rollout_seen[key] else None)
        for key in ROLLOUT_METRIC_KEYS
    }
    snapshots_all_finite = all(snapshot_metrics_finite(item) for item in snapshots)
    env_steps = measured * int(cfg.training.rollout_steps) * total_envs
    return TrainingBenchmarkResult(
        label=label,
        overrides=overrides,
        updates=updates,
        warmup=warmup,
        measured_updates=measured,
        seconds_total=total_seconds,
        seconds_per_update_mean=total_seconds / max(measured, 1),
        compile_seconds_to_update_3=compile_seconds_to_update_3,
        devices=devices,
        default_backend=default_backend,
        num_envs=total_envs,
        rollout_steps=int(cfg.training.rollout_steps),
        update_metric_means=update_means,
        rollout_metric_means=rollout_means,
        snapshots=tuple(snapshots),
        snapshots_all_finite=snapshots_all_finite,
        env_steps_per_sec=env_steps / max(total_seconds, 1e-9),
    )


def training_benchmark_payload(result: TrainingBenchmarkResult) -> dict[str, object]:
    """JSON-serializable benchmark report."""

    payload: dict[str, object] = {
        "label": result.label,
        "overrides": list(result.overrides),
        "updates": result.updates,
        "warmup": result.warmup,
        "measured_updates": result.measured_updates,
        "devices": list(result.devices),
        "default_backend": result.default_backend,
        "num_envs": result.num_envs,
        "rollout_steps": result.rollout_steps,
        "compile_seconds_to_update_3": result.compile_seconds_to_update_3,
        "seconds_total": result.seconds_total,
        "seconds_per_update_mean": result.seconds_per_update_mean,
        "env_steps_per_sec": result.env_steps_per_sec,
    }
    payload.update(result.update_metric_means)
    payload.update(result.rollout_metric_means)
    if result.snapshots:
        payload["snapshots"] = [
            {
                "update": snapshot.update,
                "curriculum_stage_id": snapshot.curriculum_stage_id,
                **{key: snapshot.update_metrics.get(key) for key in UPDATE_METRIC_KEYS},
                **{
                    key: snapshot.rollout_metrics.get(key)
                    for key in ROLLOUT_METRIC_KEYS
                },
                "survival_time": snapshot.survival_time,
            }
            for snapshot in result.snapshots
        ]
        payload["snapshots_all_finite"] = result.snapshots_all_finite
    return payload


def compose_benchmark_config(overrides: list[str]) -> TrainConfig:
    return compose_hydra_train_config(list(overrides))


def resolve_benchmark_overrides(
    *,
    preset: str | None,
    overrides: list[str] | None,
) -> list[str]:
    preset_overrides = {
        "validation": WORKSTATION_VALIDATION_OVERRIDES,
        "planet_flow_p0": PLANET_FLOW_P0_BENCHMARK_OVERRIDES,
    }
    if preset in preset_overrides:
        resolved = list(preset_overrides[preset])
        if overrides:
            resolved.extend(overrides)
        return resolved
    if overrides is not None:
        return list(overrides)
    return list(DEFAULT_BENCHMARK_OVERRIDES)


def format_profile_name(overrides: list[str]) -> str | None:
    for override in overrides:
        if override.startswith("format="):
            return override.split("=", 1)[1]
        if override.startswith("training="):
            return override.split("=", 1)[1]
    return None
