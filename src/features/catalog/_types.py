"""Shared types for single-source feature catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import jax


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Metadata for one observation feature slot."""

    name: str
    size: int = 1
    active: bool = True


class FeatureCompute(Protocol):
    """Compute one feature tensor from a group-specific assembly context."""

    def __call__(self, ctx: Any) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class FeatureCatalogEntry:
    """Single-source feature declaration: metadata plus compute."""

    definition: FeatureDefinition
    compute: FeatureCompute


@dataclass(frozen=True, slots=True)
class PlanetAssemblyContext:
    """Inputs for planet feature assembly."""

    planets: Any
    active_mask: Any
    orbit_radius: Any
    orbit_angle: Any
    owner_slot: Any
    rotating: Any
    incoming_friendly: Any
    ship_delta: Any
    scale: Any


@dataclass(frozen=True, slots=True)
class EdgeRowAssemblyContext:
    """Pre-gathered edge row tensors for catalog assembly."""

    delta_x: Any
    delta_y: Any
    distance: Any
    crosses: Any
    tgt_ships: Any
    owner_slot: Any
    turns: Any
    incoming_friendly: Any
    incoming_enemy: Any
    ordered_valid: Any
    tgt_active: Any
    scale: Any


@dataclass(frozen=True, slots=True)
class GlobalAssemblyContext:
    """Scratch for global feature assembly."""

    step_fraction: Any
    planet_fractions: Any
    ship_fractions: Any
    fleet_ship_fractions: Any
    owner_counts: Any
    owner_ships: Any
    owner_fleets: Any
    active_mask: Any
    player_count_feature: Any
    owner_production: Any
    ship_delta_slots: Any
    planet_delta_slots: Any
    fleet_delta_slots: Any
    production_delta_slots: Any
    angular_velocity: Any
