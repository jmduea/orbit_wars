"""Telemetry metric definitions for the timing group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_TIMING_BY_NAME: dict[str, MetricDefinition] = {
    "update_seconds": metric("update_seconds", "timing", "Wall-clock seconds for the full update loop."),
    "elapsed_seconds": metric("elapsed_seconds", "timing", "Wall-clock seconds since training started."),
    "rollout_seconds": metric(
        "rollout_seconds", "timing", "Wall-clock seconds spent collecting rollouts."
    ),
    "ppo_seconds": metric("ppo_seconds", "timing", "Wall-clock seconds spent in PPO optimization."),
    "env_steps_per_sec": metric(
        "env_steps_per_sec",
        "timing",
        "Environment steps processed per second over the full update.",
        protected=True,
    ),
    "rollout_env_steps_per_sec": metric(
        "rollout_env_steps_per_sec",
        "timing",
        "Environment steps processed per second during rollout collection.",
    ),
    "samples_per_sec": metric("samples_per_sec", "timing", "Decision samples processed per second."),
    "ppo_samples_per_sec": metric(
        "ppo_samples_per_sec",
        "timing",
        "Decision samples processed per second during PPO optimization.",
    ),
    "gpu_memory_used_gb": metric(
        "gpu_memory_used_gb",
        "timing",
        "Device memory in use after the update (GiB, driver-reported when available).",
    ),
    "gpu_memory_total_gb": metric(
        "gpu_memory_total_gb",
        "timing",
        "Total device memory for the active GPU (GiB).",
    ),
    "gpu_memory_peak_gb": metric(
        "gpu_memory_peak_gb",
        "timing",
        "Running peak device memory observed since run start (GiB).",
        protected=True,
    ),
}


def timing_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_TIMING_BY_NAME[name] for name in _TIMING_BY_NAME)
