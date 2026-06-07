"""Telemetry metric definitions for the curriculum group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_CURRICULUM_BY_NAME: dict[str, MetricDefinition] = {
    "seed_scheduler_policy": metric(
        "seed_scheduler_policy",
        "curriculum",
        "Seed scheduling policy selected for the next update.",
    ),
    "seed_scheduler_plateau_metric": metric(
        "seed_scheduler_plateau_metric",
        "curriculum",
        "Canonical plateau metric monitored by the seed scheduler.",
    ),
    "curriculum_stage_id": metric("curriculum_stage_id", "curriculum", "Active curriculum stage identifier."),
    "curriculum_stage_index": metric("curriculum_stage_index", "curriculum", "Active curriculum stage index."),
    "curriculum_stage_update": metric(
        "curriculum_stage_update",
        "curriculum",
        "Update index attached to the active curriculum stage snapshot.",
    ),
    "curriculum_stage_dwell_updates": metric(
        "curriculum_stage_dwell_updates",
        "curriculum",
        "Updates spent in the current curriculum stage.",
    ),
}


def curriculum_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_CURRICULUM_BY_NAME[name] for name in _CURRICULUM_BY_NAME)
