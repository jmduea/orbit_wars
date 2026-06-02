"""Telemetry registry entries derived from Planet Flow metric descriptors."""

from __future__ import annotations

from src.jax.rollout.planet_flow_metric_descriptors import (
    PLANET_FLOW_CONTROL_COUNT_DESCRIPTORS,
    PLANET_FLOW_CONTROL_PREFIX,
    PLANET_FLOW_CONTROL_RATE_DESCRIPTORS,
    PLANET_FLOW_COUNT_DESCRIPTORS,
    PLANET_FLOW_DELTA_DESCRIPTORS,
    PLANET_FLOW_METRIC_GROUP,
    PLANET_FLOW_PREFIX,
    PLANET_FLOW_RATE_DESCRIPTORS,
    PlanetFlowMetricDescriptor,
)
from src.telemetry.metric_registry import MetricDefinition


def _registry_entry(
    prefix: str,
    descriptor: PlanetFlowMetricDescriptor,
) -> MetricDefinition:
    return MetricDefinition(
        name=f"{prefix}_{descriptor.suffix}",
        group=PLANET_FLOW_METRIC_GROUP,
        description=descriptor.description,
        record_kinds=frozenset({"update"}),
    )


def _registry_entries(
    prefix: str,
    descriptors: tuple[PlanetFlowMetricDescriptor, ...],
) -> tuple[MetricDefinition, ...]:
    return tuple(_registry_entry(prefix, descriptor) for descriptor in descriptors)


def planet_flow_metric_definitions() -> tuple[MetricDefinition, ...]:
    """All Planet Flow telemetry definitions in rollout contract order."""
    return (
        *_registry_entries(PLANET_FLOW_PREFIX, PLANET_FLOW_COUNT_DESCRIPTORS),
        *_registry_entries(PLANET_FLOW_PREFIX, PLANET_FLOW_RATE_DESCRIPTORS),
        *_registry_entries(PLANET_FLOW_CONTROL_PREFIX, PLANET_FLOW_CONTROL_COUNT_DESCRIPTORS),
        *_registry_entries(PLANET_FLOW_CONTROL_PREFIX, PLANET_FLOW_CONTROL_RATE_DESCRIPTORS),
        *(
            MetricDefinition(
                name=descriptor.suffix,
                group=PLANET_FLOW_METRIC_GROUP,
                description=descriptor.description,
                record_kinds=frozenset({"update"}),
            )
            for descriptor in PLANET_FLOW_DELTA_DESCRIPTORS
        ),
    )
