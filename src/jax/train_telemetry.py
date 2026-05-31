from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.train_checkpoint import HistoricalSnapshotPool, append_jsonl
from src.telemetry.metric_registry import (
    ROLLOUT_OUTPUT_METRIC_NAMES,
    filter_update_record,
)


class _TelemetryLogger(Protocol):
    def log(self, record: dict[str, object], *, step: int) -> None: ...


def build_per_format_timing_metrics(
    format_stats: dict[int, dict[str, float]],
    *,
    update_seconds: float,
    rollout_seconds: float,
    ppo_seconds: float,
    include_per_format: bool = False,
) -> dict[str, float]:
    metrics = {
        "update_time_rollout_fraction": rollout_seconds / max(update_seconds, 1e-9),
        "update_time_ppo_fraction": ppo_seconds / max(update_seconds, 1e-9),
    }
    if not include_per_format:
        return metrics
    for player_count, suffix in ((2, "2p"), (4, "4p")):
        stats = format_stats.get(player_count, {})
        seconds = float(stats.get("seconds", 0.0))
        env_steps = float(stats.get("env_steps", 0.0))
        samples = float(stats.get("samples", 0.0))
        metrics[f"rollout_seconds_{suffix}"] = seconds
        metrics[f"env_steps_per_sec_{suffix}"] = env_steps / max(update_seconds, 1e-9)
        metrics[f"rollout_env_steps_per_sec_{suffix}"] = env_steps / max(seconds, 1e-9)
        metrics[f"samples_per_sec_{suffix}"] = samples / max(update_seconds, 1e-9)
        metrics[f"rollout_samples_per_sec_{suffix}"] = samples / max(seconds, 1e-9)
    return metrics


def historical_pool_snapshot_telemetry(
    historical_pool: HistoricalSnapshotPool, *, update: int
) -> dict[str, object]:
    """Return valid historical snapshot ids and ages for event records."""

    historical_ids = jax.device_get(historical_pool.snapshot_ids).tolist()
    historical_ages = jax.device_get(
        jnp.where(
            historical_pool.valid_mask,
            jnp.asarray(update, dtype=jnp.int32) - historical_pool.snapshot_updates,
            0,
        )
    ).tolist()
    return {
        "historical_snapshot_ids": historical_ids,
        "historical_snapshot_ages_updates": historical_ages,
    }


def rollout_metrics_for_update_record(
    rollout_scalars: dict[str, float],
    cfg: TrainConfig,
) -> dict[str, float]:
    """Merge optional rollout scalars selected by metric-group filtering."""

    core_keys = frozenset(
        {
            "samples",
            "env_steps",
            "episode_done",
            "average_reward",
            "episode_reward_mean",
            "win_rate_2p",
            "first_place_rate_4p",
            "average_placement_4p",
            "overall_win_rate",
            "survival_time",
            "score_share",
        }
    )
    metrics = {
        key: float(rollout_scalars[key])
        for key in rollout_scalars
        if key in ROLLOUT_OUTPUT_METRIC_NAMES and key not in core_keys
    }
    if "mean_active_launches_per_turn" in rollout_scalars:
        metrics["stop_utilization_ratio"] = float(
            rollout_scalars["mean_active_launches_per_turn"]
        ) / max(float(cfg.model.max_moves_k), 1.0)
    return metrics


def split_debug_update_record(
    record: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split debug/parity keys into a secondary JSONL payload."""

    lean: dict[str, object] = {}
    debug: dict[str, object] = {}
    for name, value in record.items():
        if name.startswith("debug_") or name.startswith("debug/"):
            debug[name] = value
        else:
            lean[name] = value
    return lean, debug


def build_update_record(
    *,
    update: int,
    total_env_steps: int,
    completed_episodes: int,
    rollout_samples: int,
    rollout_scalars: dict[str, float],
    metrics_host: dict[str, float],
    update_seconds: float,
    rollout_seconds: float,
    ppo_seconds: float,
    train_start_time: float,
    per_format_timing_metrics: dict[str, float],
    curriculum_telemetry: dict[str, object],
    reseed_events: list[dict[str, object]],
    update_events: list[dict[str, object]],
    historical_pool: HistoricalSnapshotPool,
    gpu_update_metrics: dict[str, object],
    seed_scheduler_policy: str,
    plateau_metric: str,
    cfg: TrainConfig,
) -> dict[str, object]:
    """Assemble the full per-update telemetry record before metric-group filtering."""

    env_steps = int(rollout_scalars["env_steps"])
    win_rate_2p = float(rollout_scalars["win_rate_2p"])
    first_place_rate_4p = float(rollout_scalars["first_place_rate_4p"])
    average_placement_4p = float(rollout_scalars["average_placement_4p"])
    survival_time = float(rollout_scalars["survival_time"])
    score_share = float(rollout_scalars["score_share"])
    episode_reward_mean = float(rollout_scalars["episode_reward_mean"])
    overall_win_rate = float(rollout_scalars["overall_win_rate"])

    return {
        "update": update,
        "total_env_steps": total_env_steps,
        "completed_episodes": completed_episodes,
        "samples": int(rollout_samples),
        "win_rate_2p": win_rate_2p,
        "first_place_rate_4p": first_place_rate_4p,
        "average_placement_4p": average_placement_4p,
        "overall_win_rate": overall_win_rate,
        "episode_reward_mean": episode_reward_mean,
        **rollout_metrics_for_update_record(rollout_scalars, cfg),
        "survival_time": survival_time,
        "score_share": score_share,
        "update_seconds": update_seconds,
        "elapsed_seconds": time.perf_counter() - train_start_time,
        "rollout_seconds": rollout_seconds,
        "ppo_seconds": ppo_seconds,
        "env_steps_per_sec": env_steps / max(update_seconds, 1e-9),
        "rollout_env_steps_per_sec": env_steps / max(rollout_seconds, 1e-9),
        "samples_per_sec": rollout_samples / max(update_seconds, 1e-9),
        "ppo_samples_per_sec": rollout_samples / max(ppo_seconds, 1e-9),
        **per_format_timing_metrics,
        "seed_scheduler_policy": seed_scheduler_policy,
        "seed_scheduler_plateau_metric": plateau_metric,
        "reseed_events": reseed_events,
        **curriculum_telemetry,
        "historical_pool_size": int(jax.device_get(historical_pool.valid_mask).sum()),
        "historical_pool_capacity": int(historical_pool.valid_mask.shape[0]),
        **{name: float(value) for name, value in metrics_host.items()},
        "curriculum_phase_events": list(update_events),
        **gpu_update_metrics,
    }


def write_filtered_update_records(
    *,
    log_path: Path,
    debug_log_path: Path,
    record: dict[str, object],
    cfg: TrainConfig,
    telemetry: _TelemetryLogger,
    update: int,
) -> None:
    """Apply metric-group filtering and write lean/debug JSONL sinks."""

    lean, debug = split_debug_update_record(record)
    filtered_lean = filter_update_record(lean, cfg)
    append_jsonl(log_path, filtered_lean)
    if debug:
        append_jsonl(debug_log_path, {"update": update, **debug})
    telemetry.log(filtered_lean, step=update)
