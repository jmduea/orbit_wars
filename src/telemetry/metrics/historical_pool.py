"""Telemetry metric definitions for the historical_pool group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_HISTORICAL_POOL_BY_NAME: dict[str, MetricDefinition] = {
    "historical_pool_size": metric(
        "historical_pool_size",
        "historical_pool",
        "Valid historical snapshot count in the pool.",
    ),
    "historical_pool_capacity": metric(
        "historical_pool_capacity",
        "historical_pool",
        "Configured historical snapshot pool capacity.",
    ),
    "historical_snapshot_ids": metric(
        "historical_snapshot_ids",
        "historical_pool",
        "Snapshot identifiers currently stored in the historical pool.",
        record_kinds=("event",),
    ),
    "historical_snapshot_ages_updates": metric(
        "historical_snapshot_ages_updates",
        "historical_pool",
        "Snapshot ages in updates for the historical pool.",
        record_kinds=("event",),
    ),
}


def historical_pool_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_HISTORICAL_POOL_BY_NAME[name] for name in _HISTORICAL_POOL_BY_NAME)
