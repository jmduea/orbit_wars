"""Structured Planet Flow metric descriptors shared by rollout, telemetry, and finalize."""

from __future__ import annotations

from dataclasses import dataclass

PLANET_FLOW_PREFIX = "planet_flow"
PLANET_FLOW_CONTROL_PREFIX = "planet_flow_control"
PLANET_FLOW_METRIC_GROUP = "action_decision"


@dataclass(frozen=True, slots=True)
class PlanetFlowMetricDescriptor:
    suffix: str
    description: str


PLANET_FLOW_COUNT_DESCRIPTORS: tuple[PlanetFlowMetricDescriptor, ...] = (
    PlanetFlowMetricDescriptor(
        "demanded_mass_sum",
        "Sum of active target pressure mass sampled by Planet Flow.",
    ),
    PlanetFlowMetricDescriptor(
        "unreachable_demand_mass_sum",
        "Planet Flow target pressure mass with no feasible candidate edge.",
    ),
    PlanetFlowMetricDescriptor(
        "held_demand_mass_sum",
        "Planet Flow target pressure mass not represented by emitted launches.",
    ),
    PlanetFlowMetricDescriptor(
        "requested_ship_mass_sum",
        "Ship mass requested by the Planet Flow compiler before capacity truncation.",
    ),
    PlanetFlowMetricDescriptor(
        "emitted_ship_mass_sum",
        "Ship mass emitted by the Planet Flow compiler.",
    ),
    PlanetFlowMetricDescriptor(
        "capacity_dropped_launch_count",
        "Planet Flow launch intents dropped by per-turn action capacity.",
    ),
    PlanetFlowMetricDescriptor(
        "emitted_launch_count",
        "Valid launches emitted by the Planet Flow compiler.",
    ),
    PlanetFlowMetricDescriptor(
        "small_launch_count",
        "Planet Flow emitted launches with at most one ship.",
    ),
    PlanetFlowMetricDescriptor(
        "duplicate_source_target_count",
        "Mergeable same-source/same-target Planet Flow duplicate launches.",
    ),
)

PLANET_FLOW_RATE_DESCRIPTORS: tuple[PlanetFlowMetricDescriptor, ...] = (
    PlanetFlowMetricDescriptor(
        "unreachable_demand_rate",
        "Unreachable Planet Flow demand mass divided by demanded mass.",
    ),
    PlanetFlowMetricDescriptor(
        "held_demand_rate",
        "Held Planet Flow demand mass divided by demanded mass.",
    ),
    PlanetFlowMetricDescriptor(
        "emitted_ship_mass_rate",
        "Emitted Planet Flow ship mass divided by requested ship mass.",
    ),
    PlanetFlowMetricDescriptor(
        "capacity_drop_rate",
        "Planet Flow capacity-dropped launch intents divided by attempted launches.",
    ),
    PlanetFlowMetricDescriptor(
        "small_launch_rate",
        "Planet Flow one-ship emitted launches divided by emitted launches.",
    ),
    PlanetFlowMetricDescriptor(
        "duplicate_source_target_rate",
        "Duplicate same-source/same-target Planet Flow launches divided by emitted launches.",
    ),
)

PLANET_FLOW_CONTROL_COUNT_DESCRIPTORS: tuple[PlanetFlowMetricDescriptor, ...] = (
    PlanetFlowMetricDescriptor(
        "demanded_mass_sum",
        "Seeded-random control target pressure mass run through the Planet Flow compiler.",
    ),
    PlanetFlowMetricDescriptor(
        "unreachable_demand_mass_sum",
        "Seeded-random control demand mass with no feasible candidate edge.",
    ),
    PlanetFlowMetricDescriptor(
        "held_demand_mass_sum",
        "Seeded-random control demand mass not represented by emitted launches.",
    ),
    PlanetFlowMetricDescriptor(
        "requested_ship_mass_sum",
        "Ship mass requested by the seeded-random Planet Flow compiler control.",
    ),
    PlanetFlowMetricDescriptor(
        "emitted_ship_mass_sum",
        "Ship mass emitted by the seeded-random Planet Flow compiler control.",
    ),
    PlanetFlowMetricDescriptor(
        "capacity_dropped_launch_count",
        "Seeded-random control launch intents dropped by per-turn action capacity.",
    ),
    PlanetFlowMetricDescriptor(
        "emitted_launch_count",
        "Valid launches emitted by the seeded-random Planet Flow compiler control.",
    ),
    PlanetFlowMetricDescriptor(
        "small_launch_count",
        "Seeded-random control emitted launches with at most one ship.",
    ),
    PlanetFlowMetricDescriptor(
        "duplicate_source_target_count",
        "Seeded-random control duplicate same-source/same-target launch count.",
    ),
)

PLANET_FLOW_CONTROL_RATE_DESCRIPTORS: tuple[PlanetFlowMetricDescriptor, ...] = (
    PlanetFlowMetricDescriptor(
        "unreachable_demand_rate",
        "Seeded-random control unreachable demand mass divided by demanded mass.",
    ),
    PlanetFlowMetricDescriptor(
        "held_demand_rate",
        "Seeded-random control held demand mass divided by demanded mass.",
    ),
    PlanetFlowMetricDescriptor(
        "emitted_ship_mass_rate",
        "Seeded-random control emitted ship mass divided by requested ship mass.",
    ),
    PlanetFlowMetricDescriptor(
        "capacity_drop_rate",
        "Seeded-random control capacity-dropped intents divided by attempted launches.",
    ),
    PlanetFlowMetricDescriptor(
        "small_launch_rate",
        "Seeded-random control one-ship launches divided by emitted launches.",
    ),
    PlanetFlowMetricDescriptor(
        "duplicate_source_target_rate",
        "Seeded-random control duplicate launches divided by emitted launches.",
    ),
)

PLANET_FLOW_DELTA_DESCRIPTORS: tuple[PlanetFlowMetricDescriptor, ...] = (
    PlanetFlowMetricDescriptor(
        "planet_flow_emitted_launch_count_delta_vs_control",
        "Learned Planet Flow emitted launch count minus seeded-random control count.",
    ),
    PlanetFlowMetricDescriptor(
        "planet_flow_emitted_ship_mass_delta_vs_control",
        "Learned Planet Flow emitted ship mass minus seeded-random control mass.",
    ),
    PlanetFlowMetricDescriptor(
        "planet_flow_unreachable_demand_rate_delta_vs_control",
        "Learned unreachable demand rate minus seeded-random control rate.",
    ),
    PlanetFlowMetricDescriptor(
        "planet_flow_held_demand_rate_delta_vs_control",
        "Learned held demand rate minus seeded-random control rate.",
    ),
    PlanetFlowMetricDescriptor(
        "planet_flow_emitted_ship_mass_rate_delta_vs_control",
        "Learned emitted ship mass rate minus seeded-random control rate.",
    ),
    PlanetFlowMetricDescriptor(
        "planet_flow_small_launch_rate_delta_vs_control",
        "Learned one-ship launch rate minus seeded-random control rate.",
    ),
    PlanetFlowMetricDescriptor(
        "planet_flow_duplicate_source_target_rate_delta_vs_control",
        "Learned duplicate launch rate minus seeded-random control rate.",
    ),
)

# Rate names whose learned-vs-control delta is derived after both rate families finalize.
PLANET_FLOW_RATE_DELTA_SUFFIXES: tuple[str, ...] = tuple(
    descriptor.suffix for descriptor in PLANET_FLOW_RATE_DESCRIPTORS
)


def _names(
    descriptors: tuple[PlanetFlowMetricDescriptor, ...],
    *,
    prefix: str,
) -> tuple[str, ...]:
    return tuple(f"{prefix}_{descriptor.suffix}" for descriptor in descriptors)


PLANET_FLOW_COUNT_KEYS = _names(PLANET_FLOW_COUNT_DESCRIPTORS, prefix=PLANET_FLOW_PREFIX)
PLANET_FLOW_RATE_KEYS = _names(PLANET_FLOW_RATE_DESCRIPTORS, prefix=PLANET_FLOW_PREFIX)
PLANET_FLOW_CONTROL_COUNT_KEYS = _names(
    PLANET_FLOW_CONTROL_COUNT_DESCRIPTORS, prefix=PLANET_FLOW_CONTROL_PREFIX
)
PLANET_FLOW_CONTROL_RATE_KEYS = _names(
    PLANET_FLOW_CONTROL_RATE_DESCRIPTORS, prefix=PLANET_FLOW_CONTROL_PREFIX
)
PLANET_FLOW_CONTROL_DELTA_KEYS = tuple(
    descriptor.suffix for descriptor in PLANET_FLOW_DELTA_DESCRIPTORS
)