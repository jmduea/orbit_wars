"""Telemetry metric definitions for the trajectory_shield_debug group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_TRAJECTORY_SHIELD_DEBUG_BY_NAME: dict[str, MetricDefinition] = {
    "trajectory_shield_blocked_count": metric(
        "trajectory_shield_blocked_count",
        "trajectory_shield_debug",
        "Count of actions blocked by the trajectory shield.",
        rollout_scalar_role="base_sum",
    ),
    "trajectory_shield_blocked_sun_count": metric(
        "trajectory_shield_blocked_sun_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to sun collisions.",
        rollout_scalar_role="base_sum",
    ),
    "trajectory_shield_blocked_bounds_count": metric(
        "trajectory_shield_blocked_bounds_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to map bounds.",
        rollout_scalar_role="base_sum",
    ),
    "trajectory_shield_blocked_unintended_hit_count": metric(
        "trajectory_shield_blocked_unintended_hit_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to unintended hits.",
        rollout_scalar_role="base_sum",
    ),
    "trajectory_shield_blocked_horizon_count": metric(
        "trajectory_shield_blocked_horizon_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to shield horizon limits.",
        rollout_scalar_role="base_sum",
    ),
    "trajectory_shield_fallback_noop_count": metric(
        "trajectory_shield_fallback_noop_count",
        "trajectory_shield_debug",
        "Count of shielded decisions that fell back to noop.",
        rollout_scalar_role="base_sum",
    ),
    "trajectory_shield_legal_non_noop_rate": metric(
        "trajectory_shield_legal_non_noop_rate",
        "trajectory_shield_debug",
        "Fraction of originally non-noop decisions that remained legal after shielding.",
        rollout_scalar_role="base_sum",
    ),
}


def trajectory_shield_debug_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_TRAJECTORY_SHIELD_DEBUG_BY_NAME[name] for name in _TRAJECTORY_SHIELD_DEBUG_BY_NAME)
