"""Rollout phase timing metric keys (opt-in via telemetry.metric_groups.rollout_phase_timing)."""

from __future__ import annotations

ROLLOUT_PHASE_TIMING_KEYS: tuple[str, ...] = (
    "rollout_phase_policy_seconds",
    "rollout_phase_opponent_seconds",
    "rollout_phase_env_step_seconds",
    "rollout_phase_reset_seconds",
    "rollout_phase_post_step_seconds",
    "rollout_phase_measured_total_seconds",
    "rollout_phase_policy_fraction",
    "rollout_phase_opponent_fraction",
    "rollout_phase_env_step_fraction",
    "rollout_phase_reset_fraction",
    "rollout_phase_post_step_fraction",
)

__all__ = ["ROLLOUT_PHASE_TIMING_KEYS"]
