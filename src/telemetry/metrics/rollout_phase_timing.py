"""Rollout phase timing metrics (offline benchmarks; opt-in via metric group)."""

from __future__ import annotations

from src.jax.rollout.phase_timing import ROLLOUT_PHASE_TIMING_KEYS
from src.telemetry.metric_definition import MetricDefinition, metric

_ROLLOUT_PHASE_TIMING_BY_NAME: dict[str, MetricDefinition] = {
    key: metric(
        key,
        "rollout_phase_timing",
        f"Rollout collect sub-phase timing for {key}.",
    )
    for key in ROLLOUT_PHASE_TIMING_KEYS
}
