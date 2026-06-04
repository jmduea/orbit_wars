"""Telemetry metric definitions for the game_state group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_GAME_STATE_BY_NAME: dict[str, MetricDefinition] = {
    "survival_time": metric(
        "survival_time",
        "game_state",
        "Mean survival time for completed episodes.",
        rollout_scalar_role="finalized_rate",
    ),
    "score_share": metric(
        "score_share",
        "game_state",
        "Mean score share for completed episodes.",
        rollout_scalar_role="finalized_rate",
    ),
}


def game_state_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_GAME_STATE_BY_NAME[name] for name in _GAME_STATE_BY_NAME)
