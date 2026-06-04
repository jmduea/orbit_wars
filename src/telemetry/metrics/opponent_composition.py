"""Telemetry metric definitions for the opponent_composition group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_OPPONENT_COMPOSITION_BY_NAME: dict[str, MetricDefinition] = {
    "opponent_slots_total": metric(
        "opponent_slots_total",
        "opponent_composition",
        "Total opponent slots sampled across the rollout.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_latest": metric(
        "opponent_slots_latest",
        "opponent_composition",
        "Opponent slots filled by the latest learner snapshot.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_historical": metric(
        "opponent_slots_historical",
        "opponent_composition",
        "Opponent slots filled by historical learner snapshots.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_random": metric(
        "opponent_slots_random",
        "opponent_composition",
        "Opponent slots filled by random policy opponents.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_noop": metric(
        "opponent_slots_noop",
        "opponent_composition",
        "Opponent slots filled by no-op opponents.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_nearest_sniper": metric(
        "opponent_slots_nearest_sniper",
        "opponent_composition",
        "Opponent slots filled by nearest-sniper opponents.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_turtle": metric(
        "opponent_slots_turtle",
        "opponent_composition",
        "Opponent slots filled by turtle opponents.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_slots_opportunistic": metric(
        "opponent_slots_opportunistic",
        "opponent_composition",
        "Opponent slots filled by opportunistic opponents.",
        rollout_scalar_role="base_sum",
    ),
    "opponent_historical_fallback_latest_slots": metric(
        "opponent_historical_fallback_latest_slots",
        "opponent_composition",
        "Historical opponent slots that fell back to the latest policy.",
        rollout_scalar_role="base_sum",
    ),
}


def opponent_composition_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_OPPONENT_COMPOSITION_BY_NAME[name] for name in _OPPONENT_COMPOSITION_BY_NAME)
