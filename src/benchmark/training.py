"""Production-aligned short training runs with PPO + rollout scalar metrics."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import jax.numpy as jnp
import jax.profiler
import jax.random
import jax.tree

import jax
from src.benchmark.production import rollout_group_summary
from src.config import TrainConfig, compose_hydra_train_config
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

PRIMARY_E2E_OVERRIDES: tuple[str, ...] = (
    "task=shield_cheap",
    "model=transformer_factorized",
    "opponents=self_play_only",
    "curriculum=off",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "seed=42",
)

E2E_THROUGHPUT_GATE = "launch_hygiene_e2e_throughput"
E2E_THROUGHPUT_METRICS: tuple[str, ...] = (
    "env_steps_per_sec",
    "samples_per_sec",
    "seconds_per_update_mean",
)
DEFAULT_E2E_WITHIN_PCT = 10.0
MIN_BASELINE_RUNS = 3

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
    rollout_metric_means: Mapping[str, float]
    snapshots: tuple[TrainingBenchmarkSnapshot, ...] = field(default_factory=tuple)
    snapshots_all_finite: bool = True
    env_steps: int = 0
    samples: int = 0
    env_steps_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    rollout_collect_seconds_per_update_mean: float | None = None
    ppo_seconds_per_update_mean: float | None = None
    host_overhead_seconds_per_update_mean: float | None = None


def _scalar_metric(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    return float(jax.device_get(metrics[key]))


def _json_number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"expected JSON number, got {type(value).__name__}")


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
    detailed_timing: bool = False,
    profile_dir: Path | None = None,
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
    rollout_timings: list[float] = []
    ppo_timings: list[float] = []
    overhead_timings: list[float] = []
    update_sums = {key: 0.0 for key in UPDATE_METRIC_KEYS}
    rollout_sums = {key: 0.0 for key in ROLLOUT_METRIC_KEYS}
    env_steps_total = 0.0
    samples_total = 0.0
    measured = 0
    compile_seconds_to_update_3: float | None = None
    snapshot_targets = set(snapshot_updates)
    snapshots: list[TrainingBenchmarkSnapshot] = []
    profile_active = False

    try:
        for iteration in range(warmup + updates):
            update = iteration + 1
            if profile_dir is not None and iteration == warmup:
                profile_dir.mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(str(profile_dir))
                profile_active = True

            t0 = time.perf_counter()
            with jax.profiler.TraceAnnotation("ow_benchmark_update"):
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
                rollout_t0 = time.perf_counter()
                transitions_by_group = []
                rollout_metrics_by_group = []
                next_groups = []
                with jax.profiler.TraceAnnotation("ow_rollout_collect"):
                    for group_idx, rollout_key in zip(
                        active_indices, rollout_keys, strict=True
                    ):
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
                    for group_idx, updated_group in zip(
                        active_indices, next_groups, strict=True
                    ):
                        merged_groups[group_idx] = updated_group
                    rollout_groups = merged_groups

                    transitions = concatenate_transition_batches(transitions_by_group)
                    rollout_metrics = sum_metric_dicts(rollout_metrics_by_group)
                    if detailed_timing:
                        jax.block_until_ready(transitions.log_prob)
                rollout_elapsed = time.perf_counter() - rollout_t0

                ppo_t0 = time.perf_counter()
                metrics_accum = None
                with jax.profiler.TraceAnnotation("ow_ppo_update_epochs"):
                    for _ in range(cfg.training.epochs):
                        train_state, update_metrics = update_fn(
                            train_state, transitions
                        )
                        metrics_accum = (
                            update_metrics
                            if metrics_accum is None
                            else jax.tree.map(jnp.add, metrics_accum, update_metrics)
                        )
                assert metrics_accum is not None
                jax.block_until_ready(metrics_accum["total_loss"])
                ppo_elapsed = time.perf_counter() - ppo_t0
            elapsed = time.perf_counter() - t0
            if update == 3:
                compile_seconds_to_update_3 = time.perf_counter() - run_started
            if iteration >= warmup:
                measured += 1
                timings.append(elapsed)
                if detailed_timing:
                    rollout_timings.append(rollout_elapsed)
                    ppo_timings.append(ppo_elapsed)
                    overhead_timings.append(
                        max(elapsed - rollout_elapsed - ppo_elapsed, 0.0)
                    )
                for metric_key in UPDATE_METRIC_KEYS:
                    if metric_key in metrics_accum:
                        update_sums[metric_key] += float(
                            jax.device_get(metrics_accum[metric_key])
                        )
                for metric_key in ROLLOUT_METRIC_KEYS:
                    if metric_key in rollout_metrics:
                        rollout_sums[metric_key] += float(
                            jax.device_get(rollout_metrics[metric_key])
                        )
                if "env_steps" in rollout_metrics:
                    env_steps_total += float(
                        jax.device_get(rollout_metrics["env_steps"])
                    )
                if "samples" in rollout_metrics:
                    samples_total += float(jax.device_get(rollout_metrics["samples"]))
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
    finally:
        if profile_active:
            jax.profiler.stop_trace()

    total_seconds = sum(timings)
    update_means = {
        key: update_sums[key] / max(measured, 1) for key in UPDATE_METRIC_KEYS
    }
    rollout_means = {
        key: rollout_sums[key] / max(measured, 1) for key in ROLLOUT_METRIC_KEYS
    }
    snapshots_all_finite = all(snapshot_metrics_finite(item) for item in snapshots)
    env_steps = int(env_steps_total)
    samples = int(samples_total)
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
        env_steps=env_steps,
        samples=samples,
        env_steps_per_sec=env_steps / max(total_seconds, 1e-9),
        samples_per_sec=samples / max(total_seconds, 1e-9),
        rollout_collect_seconds_per_update_mean=(
            statistics.fmean(rollout_timings) if rollout_timings else None
        ),
        ppo_seconds_per_update_mean=(
            statistics.fmean(ppo_timings) if ppo_timings else None
        ),
        host_overhead_seconds_per_update_mean=(
            statistics.fmean(overhead_timings) if overhead_timings else None
        ),
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
        "env_steps": result.env_steps,
        "samples": result.samples,
        "env_steps_per_sec": result.env_steps_per_sec,
        "samples_per_sec": result.samples_per_sec,
    }
    if result.rollout_collect_seconds_per_update_mean is not None:
        payload["rollout_collect_seconds_per_update_mean"] = (
            result.rollout_collect_seconds_per_update_mean
        )
    if result.ppo_seconds_per_update_mean is not None:
        payload["ppo_seconds_per_update_mean"] = result.ppo_seconds_per_update_mean
    if result.host_overhead_seconds_per_update_mean is not None:
        payload["host_overhead_seconds_per_update_mean"] = (
            result.host_overhead_seconds_per_update_mean
        )
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
    if preset == "admission":
        from src.jax.preflight_gate_loader import admission_gate_train_overrides

        resolved = list(admission_gate_train_overrides(extra=tuple(overrides or ())))
        return resolved
    preset_overrides = {
        "validation": WORKSTATION_VALIDATION_OVERRIDES,
        "primary": PRIMARY_E2E_OVERRIDES,
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


def default_benchmark_updates(*, preset: str | None) -> int:
    if preset == "primary":
        return 20
    return 30


def e2e_throughput_metric_values(payload: Mapping[str, object]) -> dict[str, float]:
    return {
        key: _json_number(payload[key])
        for key in E2E_THROUGHPUT_METRICS
        if key in payload and payload[key] is not None
    }


def resolve_e2e_measured_for_gate(
    *,
    repeats: int,
    run_payloads: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Throughput metrics for baseline gate comparison."""

    if repeats == 1:
        if not run_payloads:
            return {}
        return e2e_throughput_metric_values(run_payloads[0])
    if not isinstance(aggregate, dict):
        return {}
    measured: dict[str, float] = {}
    for key in E2E_THROUGHPUT_METRICS:
        stats = aggregate.get(key)
        if not isinstance(stats, dict) or "mean" not in stats:
            continue
        measured[key] = float(stats["mean"])
    return measured


def aggregate_e2e_run_payloads(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for metric in E2E_THROUGHPUT_METRICS:
        values = [
            _json_number(run[metric])
            for run in runs
            if metric in run and run[metric] is not None
        ]
        if not values:
            continue
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return aggregate


def derive_e2e_pass_band(
    aggregate: Mapping[str, Mapping[str, float]],
    *,
    within_pct: float = DEFAULT_E2E_WITHIN_PCT,
) -> dict[str, object]:
    factor_low = 1.0 - within_pct / 100.0
    factor_high = 1.0 + within_pct / 100.0
    floors: dict[str, float] = {}
    ceilings: dict[str, float] = {}
    env_stats = aggregate.get("env_steps_per_sec")
    if env_stats is not None:
        floors["env_steps_per_sec"] = float(env_stats["mean"]) * factor_low
    seconds_stats = aggregate.get("seconds_per_update_mean")
    if seconds_stats is not None:
        ceilings["seconds_per_update_mean"] = float(seconds_stats["mean"]) * factor_high
    return {
        "within_pct": within_pct,
        "floors": floors,
        "ceilings": ceilings,
    }


def validate_e2e_baseline_artifact(baseline: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if baseline.get("gate") != E2E_THROUGHPUT_GATE:
        errors.append(f"gate must be {E2E_THROUGHPUT_GATE!r}")
    runs = baseline.get("runs")
    if not isinstance(runs, list) or len(runs) < MIN_BASELINE_RUNS:
        errors.append(f"runs must contain at least {MIN_BASELINE_RUNS} entries")
    aggregate = baseline.get("aggregate")
    if not isinstance(aggregate, dict):
        errors.append("aggregate must be an object")
    else:
        required_metrics = E2E_THROUGHPUT_METRICS
        pass_band = baseline.get("pass_band")
        if isinstance(pass_band, dict):
            authority = pass_band.get("gate_authority_metrics")
            if isinstance(authority, list) and authority:
                required_metrics = tuple(str(metric) for metric in authority)
        for metric in required_metrics:
            stats = aggregate.get(metric)
            if not isinstance(stats, dict) or "mean" not in stats:
                errors.append(f"aggregate.{metric}.mean is required")
    pass_band = baseline.get("pass_band")
    if pass_band is not None and not isinstance(pass_band, dict):
        errors.append("pass_band must be an object when present")
    return errors


def load_e2e_baseline(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"baseline artifact not found: {path}")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"baseline artifact must be a JSON object: {path}")
    errors = validate_e2e_baseline_artifact(baseline)
    if errors:
        raise ValueError(f"invalid baseline artifact {path}: {'; '.join(errors)}")
    return baseline


def resolve_e2e_pass_band(
    baseline: Mapping[str, object],
    *,
    within_pct: float | None,
) -> dict[str, object]:
    embedded = baseline.get("pass_band")
    if isinstance(embedded, dict) and within_pct is None:
        return dict(embedded)
    aggregate = baseline.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("baseline aggregate is required to derive pass band")
    pct = (
        float(within_pct)
        if within_pct is not None
        else float(
            embedded.get("within_pct", DEFAULT_E2E_WITHIN_PCT)
            if isinstance(embedded, dict)
            else DEFAULT_E2E_WITHIN_PCT
        )
    )
    return derive_e2e_pass_band(aggregate, within_pct=pct)  # type: ignore[arg-type]


def compare_e2e_throughput_to_baseline(
    measured: Mapping[str, float],
    *,
    pass_band: Mapping[str, object],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    floors = pass_band.get("floors")
    ceilings = pass_band.get("ceilings")
    if isinstance(floors, dict):
        for metric, floor in floors.items():
            observed = measured.get(str(metric))
            if observed is None:
                failures.append(f"missing measured metric {metric}")
                continue
            if float(observed) < float(floor):
                failures.append(
                    f"{metric} {float(observed):.4f} < floor {float(floor):.4f}"
                )
    if isinstance(ceilings, dict):
        for metric, ceiling in ceilings.items():
            observed = measured.get(str(metric))
            if observed is None:
                failures.append(f"missing measured metric {metric}")
                continue
            if float(observed) > float(ceiling):
                failures.append(
                    f"{metric} {float(observed):.4f} > ceiling {float(ceiling):.4f}"
                )
    return (not failures, failures)


def device_fingerprint(devices: Sequence[str], default_backend: str) -> str:
    device_text = ", ".join(devices) if devices else "unknown"
    return f"{default_backend}:{device_text}"


def check_baseline_device_match(
    baseline: Mapping[str, object],
    *,
    devices: Sequence[str],
    default_backend: str,
    mode: str,
    force: bool,
) -> tuple[bool, str | None]:
    device_info = baseline.get("device")
    if not isinstance(device_info, dict):
        return True, None
    baseline_backend = str(device_info.get("default_backend", ""))
    baseline_devices = device_info.get("devices")
    current_backend = default_backend
    current_devices = list(devices)
    mismatch = baseline_backend and baseline_backend != current_backend
    if isinstance(baseline_devices, list) and baseline_devices:
        mismatch = mismatch or list(baseline_devices) != current_devices
    if not mismatch:
        return True, None
    message = (
        f"device mismatch: baseline {baseline_backend}/{baseline_devices} "
        f"vs current {current_backend}/{current_devices}"
    )
    if mode == "strict" and not force:
        return False, message
    if mode == "warn" and not force:
        return True, message
    return True, None


def build_e2e_baseline_artifact(
    *,
    commit_sha: str | None,
    merge_topology_notes: str,
    co_landing_commits: Sequence[str],
    run_date: str,
    device: Mapping[str, object],
    primary_profile: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
    within_pct: float = DEFAULT_E2E_WITHIN_PCT,
    operator_example: str,
    gap_assessment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    aggregate = aggregate_e2e_run_payloads(runs)
    pass_band = derive_e2e_pass_band(aggregate, within_pct=within_pct)
    artifact: dict[str, object] = {
        "gate": E2E_THROUGHPUT_GATE,
        "commit_sha": commit_sha,
        "merge_topology_notes": merge_topology_notes,
        "co_landing_commits": list(co_landing_commits),
        "run_date": run_date,
        "device": dict(device),
        "primary_profile": dict(primary_profile),
        "runs": list(runs),
        "aggregate": aggregate,
        "pass_band": pass_band,
        "operator_example": operator_example,
    }
    if gap_assessment is not None:
        artifact["gap_assessment"] = dict(gap_assessment)
    errors = validate_e2e_baseline_artifact(artifact)
    if errors:
        raise ValueError(f"invalid baseline artifact: {'; '.join(errors)}")
    return artifact


def format_profile_name(overrides: list[str]) -> str | None:
    for override in overrides:
        if override.startswith("format="):
            return override.split("=", 1)[1]
        if override.startswith("training="):
            return override.split("=", 1)[1]
    return None
