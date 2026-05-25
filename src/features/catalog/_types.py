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
    description: str = ""

    def __repr__(self) -> str:
        return f"FeatureDefinition(name={self.name}, size={self.size}, active={self.active},\n description={self.description})"


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
    # TODO: add planet_score (planet_total_prod * remaining steps)
    # TODO: add connectedness (number of static edges)


@dataclass(frozen=True, slots=True)
class EdgeRowAssemblyContext:
    """Pre-gathered edge row tensors for catalog assembly.

    Per-anchor tensors are shaped ``(P, K, num_anchors)`` where anchor index 0
    corresponds to the first entry of ``TaskConfig.intercept_anchors`` (e.g.
    ``s=1.0``) and index 1 to the second (e.g. ``s=6.0``). ``crosses_now`` is
    the legality-aligned snapshot sun-crossing field (mirrors the dynamic
    trajectory shield's snapshot-line check); the per-anchor
    ``sun_cross_at_intercept_per_anchor`` is the predictive counterpart
    evaluated against the target's future position at each anchor.
    ``tgt_ships_per_anchor`` holds ``min(ships + production * tau, scale) / scale``
    at each anchor's intercept delay ``tau``.
    """

    intercept_delta_x_per_anchor: Any
    intercept_delta_y_per_anchor: Any
    intercept_distance_per_anchor: Any
    intercept_turns_per_anchor: Any
    sun_cross_at_intercept_per_anchor: Any
    crosses_now: Any
    tgt_ships_per_anchor: Any
    owner_slot: Any
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
    # TODO: add owner_score ((owner_total_prod * remaining steps) + owner_ships)
    # TODO: add owner_score_delta (owner_score - previous_owner_score)
    # TODO: add current_winner_score (max(owner_score))
