"""Planet feature catalog — single-source declarations and assembly.

Adding a planet feature example::

    def _feat_my_signal(ctx: PlanetAssemblyContext) -> jnp.ndarray:
        return (ctx.planets.ships / ctx.scale)[..., None]

    # Append to PLANET_FEATURE_ENTRIES:
    FeatureCatalogEntry(FeatureDefinition("my_signal"), _feat_my_signal)

Rebuild is automatic; ``PLANET_FEATURE_SCHEMA.base_dim`` and encoder assembly
pick up the new slot without editing ``constants.py`` or manual ``jnp.stack``.
"""

from __future__ import annotations

import jax.numpy as jnp

from src.config.schema import TaskConfig
from src.features.catalog._core import FeatureCatalog
from src.features.catalog._types import (
    FeatureCatalogEntry,
    FeatureDefinition,
    PlanetAssemblyContext,
)
from src.game.constants import BOARD_CENTER, BOARD_SIZE, MAX_PRODUCTION
from src.jax.feature_primitives import (
    canonical_angle,
    incoming_fleet_pressure,
    is_rotating_xy,
    target_owner_one_hot,
)


def _feat_active(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return ctx.planets.active.astype(jnp.float32)[..., None]


def _feat_orbit_radius(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return ctx.orbit_radius[..., None]


def _feat_orbit_angle(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return ctx.orbit_angle[..., None]


def _feat_radius(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return (ctx.planets.radius / 5.0)[..., None]


def _feat_ships(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return (jnp.minimum(ctx.planets.ships, ctx.scale) / ctx.scale)[..., None]


def _feat_production(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return (ctx.planets.production / MAX_PRODUCTION)[..., None]


def _feat_rotating_flag(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return ctx.rotating.astype(jnp.float32)[..., None]


def _feat_owner_slot(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return ctx.owner_slot


def _feat_incoming_friendly_pressure(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_friendly / ctx.scale)[..., None]


def _feat_ship_delta(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return ctx.ship_delta[..., None]


PLANET_FEATURE_ENTRIES: tuple[FeatureCatalogEntry, ...] = (
    FeatureCatalogEntry(FeatureDefinition("active"), _feat_active),
    FeatureCatalogEntry(FeatureDefinition("orbit_radius"), _feat_orbit_radius),
    FeatureCatalogEntry(FeatureDefinition("orbit_angle"), _feat_orbit_angle),
    FeatureCatalogEntry(FeatureDefinition("radius"), _feat_radius),
    FeatureCatalogEntry(FeatureDefinition("ships"), _feat_ships),
    FeatureCatalogEntry(FeatureDefinition("production"), _feat_production),
    FeatureCatalogEntry(FeatureDefinition("rotating_flag"), _feat_rotating_flag),
    FeatureCatalogEntry(FeatureDefinition("owner_slot", size=4), _feat_owner_slot),
    FeatureCatalogEntry(
        FeatureDefinition("incoming_friendly_pressure"),
        _feat_incoming_friendly_pressure,
    ),
    FeatureCatalogEntry(FeatureDefinition("ship_delta"), _feat_ship_delta),
)

PLANET_FEATURE_CATALOG = FeatureCatalog(PLANET_FEATURE_ENTRIES)


def build_planet_context(
    planets,
    fleets,
    player,
    env_cfg: TaskConfig,
    scale,
    theta_ref,
    history_planet_ships,
) -> PlanetAssemblyContext:
    """Build shared scratch for planet feature assembly."""

    owner_slot = target_owner_one_hot(planets.owner, player, env_cfg)
    rotating = is_rotating_xy(planets.x, planets.y, planets.radius)
    sun_dx = planets.x - BOARD_CENTER[0]
    sun_dy = planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(sun_dx * sun_dx + sun_dy * sun_dy) / BOARD_SIZE
    orbit_angle = canonical_angle(planets.x, planets.y, theta_ref)
    ship_delta = (planets.ships - history_planet_ships) / scale
    incoming_friendly, _incoming_enemy = incoming_fleet_pressure(
        planets.x, planets.y, planets.radius, fleets, player
    )
    return PlanetAssemblyContext(
        planets=planets,
        active_mask=planets.active,
        orbit_radius=orbit_radius,
        orbit_angle=orbit_angle,
        owner_slot=owner_slot,
        rotating=rotating,
        incoming_friendly=incoming_friendly,
        ship_delta=ship_delta,
        scale=scale,
    )


def assemble_planet_features(context: PlanetAssemblyContext) -> jnp.ndarray:
    """Assemble planet features and zero inactive rows."""

    features = PLANET_FEATURE_CATALOG.assemble(context)
    return jnp.where(context.active_mask[:, None], features, jnp.zeros_like(features))
