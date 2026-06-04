"""Telemetry metric definitions for the events group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_EVENTS_BY_NAME: dict[str, MetricDefinition] = {
    "gpu_name": metric(
        "gpu_name",
        "events",
        "Observed GPU product name for the training run.",
        record_kinds=("update", "event"),
    ),
    "reseed_events": metric(
        "reseed_events",
        "events",
        "Embedded seed reseed events emitted during the update.",
    ),
    "curriculum_phase_events": metric(
        "curriculum_phase_events",
        "events",
        "Embedded curriculum and historical snapshot events emitted during the update.",
    ),
    "event": metric(
        "event",
        "events",
        "Sparse event record type.",
        record_kinds=("event",),
        protected=True,
    ),
    "checkpoint_status": metric(
        "checkpoint_status",
        "events",
        "Checkpoint pipeline status for a checkpoint_result event.",
        record_kinds=("event",),
        protected=True,
    ),
    "checkpoint_final": metric(
        "checkpoint_final",
        "events",
        "Whether the checkpoint_result event corresponds to the final checkpoint.",
        record_kinds=("event",),
        protected=True,
    ),
    "checkpoint_reason": metric(
        "checkpoint_reason",
        "events",
        "Checkpoint pipeline reason string.",
        record_kinds=("event",),
        protected=True,
    ),
    "checkpoint_error": metric(
        "checkpoint_error",
        "events",
        "Checkpoint pipeline error text, if any.",
        record_kinds=("event",),
        protected=True,
    ),
    "snapshot_id": metric(
        "snapshot_id",
        "events",
        "Historical snapshot identifier for a sparse event record.",
        record_kinds=("event",),
    ),
    "snapshot_slot": metric(
        "snapshot_slot",
        "events",
        "Historical snapshot slot for a sparse event record.",
        record_kinds=("event",),
    ),
    "historical_snapshot_evicted": metric(
        "historical_snapshot_evicted",
        "events",
        "Whether a historical snapshot event replaced an existing slot.",
        record_kinds=("event",),
    ),
    "from_stage": metric(
        "from_stage",
        "events",
        "Previous curriculum stage for a sparse event record.",
        record_kinds=("event",),
    ),
    "to_stage": metric(
        "to_stage",
        "events",
        "Next curriculum stage for a sparse event record.",
        record_kinds=("event",),
    ),
    "stage": metric(
        "stage",
        "events",
        "Current curriculum stage for a sparse event record.",
        record_kinds=("event",),
    ),
    "reason": metric(
        "reason",
        "events",
        "Human-readable reason attached to a sparse event record.",
        record_kinds=("event",),
    ),
    "metric": metric(
        "metric",
        "events",
        "Metric name attached to a sparse event record.",
        record_kinds=("event",),
    ),
    "metric_value": metric(
        "metric_value",
        "events",
        "Metric value attached to a sparse event record.",
        record_kinds=("event",),
    ),
    "threshold": metric(
        "threshold",
        "events",
        "Threshold value attached to a sparse event record.",
        record_kinds=("event",),
    ),
    "bracket_training_phase": metric(
        "bracket_training_phase",
        "events",
        "Bracket training phase (qualifier, main, weak_config) for bracket_training profile.",
    ),
    "weak_config": metric(
        "weak_config",
        "events",
        "True when 500M env-step qualifier budget exhausted without qualifier clear.",
    ),
}


def events_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_EVENTS_BY_NAME[name] for name in _EVENTS_BY_NAME)
