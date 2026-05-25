"""Global feature catalog — single-source declarations and frame assembly."""

from __future__ import annotations

import jax.numpy as jnp

from src.config.schema import TaskConfig
from src.features.catalog._core import FeatureCatalog
from src.features.catalog._types import (
    FeatureCatalogEntry,
    FeatureDefinition,
    GlobalAssemblyContext,
)
from src.game.constants import (
    ANGULAR_VELOCITY_NORM,
    MAX_PLANETS,
    MAX_STEPS,
)
from src.jax.feature_primitives import owner_relative_production


def _feat_step_fraction(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.step_fraction


def _feat_planet_fractions(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.planet_fractions


def _feat_ship_fractions(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.ship_fractions


def _feat_fleet_ship_fractions(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.fleet_ship_fractions


def _feat_owner_relative_planet_counts(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.owner_counts


def _feat_owner_relative_ship_totals(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.owner_ships


def _feat_owner_relative_fleet_totals(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.owner_fleets


def _feat_active_owner_mask(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.active_mask


def _feat_player_count(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.player_count_feature


def _feat_owner_relative_production(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.owner_production


def _feat_ship_delta_slots(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.ship_delta_slots


def _feat_planet_delta_slots(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.planet_delta_slots


def _feat_fleet_delta_slots(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.fleet_delta_slots


def _feat_production_delta_slots(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.production_delta_slots


def _feat_angular_velocity(ctx: GlobalAssemblyContext) -> jnp.ndarray:
    return ctx.angular_velocity


GLOBAL_FEATURE_ENTRIES: tuple[FeatureCatalogEntry, ...] = (
    FeatureCatalogEntry(FeatureDefinition("step_fraction"), _feat_step_fraction),
    FeatureCatalogEntry(
        FeatureDefinition("planet_fractions", size=3), _feat_planet_fractions
    ),
    FeatureCatalogEntry(
        FeatureDefinition("ship_fractions", size=2), _feat_ship_fractions
    ),
    FeatureCatalogEntry(
        FeatureDefinition("fleet_ship_fractions", size=2), _feat_fleet_ship_fractions
    ),
    FeatureCatalogEntry(
        FeatureDefinition("owner_relative_planet_counts", size=4),
        _feat_owner_relative_planet_counts,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("owner_relative_ship_totals", size=4),
        _feat_owner_relative_ship_totals,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("owner_relative_fleet_totals", size=4),
        _feat_owner_relative_fleet_totals,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("active_owner_mask", size=4), _feat_active_owner_mask
    ),
    FeatureCatalogEntry(FeatureDefinition("player_count"), _feat_player_count),
    FeatureCatalogEntry(
        FeatureDefinition("owner_relative_production", size=4),
        _feat_owner_relative_production,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("ship_delta_slots", size=4), _feat_ship_delta_slots
    ),
    FeatureCatalogEntry(
        FeatureDefinition("planet_delta_slots", size=4), _feat_planet_delta_slots
    ),
    FeatureCatalogEntry(
        FeatureDefinition("fleet_delta_slots", size=4), _feat_fleet_delta_slots
    ),
    FeatureCatalogEntry(
        FeatureDefinition("production_delta_slots", size=4),
        _feat_production_delta_slots,
    ),
    FeatureCatalogEntry(FeatureDefinition("angular_velocity"), _feat_angular_velocity),
)

GLOBAL_FEATURE_CATALOG = FeatureCatalog(GLOBAL_FEATURE_ENTRIES, concat_axis=0)


def build_global_context(
    game,
    env_cfg: TaskConfig,
    scale,
    previous_global,
    previous_present,
) -> GlobalAssemblyContext:
    """Build shared scratch for one global feature frame."""

    planets = game.planets
    fleets = game.fleets
    player = game.player
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    neutral = planets.active & (planets.owner == -1)
    my_fleet = fleets.active & (fleets.owner == player)
    enemy_fleet = fleets.active & (fleets.owner != player)
    denom = MAX_PLANETS * scale
    owner_production = owner_relative_production(planets, player, env_cfg)

    player_count_int = max(1, min(4, int(env_cfg.player_count)))
    player_count = jnp.asarray(player_count_int, dtype=jnp.int32)
    planet_slots = (
        planets.owner.astype(jnp.int32) - player.astype(jnp.int32)
    ) % player_count
    valid_planets = (
        planets.active & (planets.owner >= 0) & (planets.owner < player_count)
    )
    owner_counts_raw = jnp.bincount(
        planet_slots,
        weights=valid_planets.astype(jnp.float32),
        length=4,
    )[:4]
    owner_ships_raw = jnp.bincount(
        planet_slots,
        weights=jnp.where(valid_planets, planets.ships, 0.0),
        length=4,
    )[:4]
    fleet_slots = (
        fleets.owner.astype(jnp.int32) - player.astype(jnp.int32)
    ) % player_count
    valid_fleets = fleets.active & (fleets.owner >= 0) & (fleets.owner < player_count)
    owner_fleets_raw = jnp.bincount(
        fleet_slots,
        weights=jnp.where(valid_fleets, fleets.ships, 0.0),
        length=4,
    )[:4]
    owner_counts = owner_counts_raw / MAX_PLANETS
    owner_ships = owner_ships_raw / denom
    owner_fleets = owner_fleets_raw / denom
    active_mask = (jnp.arange(4, dtype=jnp.int32) < player_count).astype(jnp.float32)
    player_count_feature = jnp.asarray([player_count_int / 4.0], dtype=jnp.float32)

    ship_totals_slice = GLOBAL_FEATURE_CATALOG.base_slice("owner_relative_ship_totals")
    planet_counts_slice = GLOBAL_FEATURE_CATALOG.base_slice(
        "owner_relative_planet_counts"
    )
    fleet_totals_slice = GLOBAL_FEATURE_CATALOG.base_slice(
        "owner_relative_fleet_totals"
    )
    production_slice = GLOBAL_FEATURE_CATALOG.base_slice("owner_relative_production")

    step_fraction = jnp.asarray(
        [game.step.astype(jnp.float32) / MAX_STEPS],
        dtype=jnp.float32,
    )
    planet_fractions = jnp.asarray(
        [
            mine.astype(jnp.float32).sum() / MAX_PLANETS,
            enemy.astype(jnp.float32).sum() / MAX_PLANETS,
            neutral.astype(jnp.float32).sum() / MAX_PLANETS,
        ],
        dtype=jnp.float32,
    )
    ship_fractions = jnp.asarray(
        [
            jnp.where(mine, planets.ships, 0.0).sum() / denom,
            jnp.where(enemy, planets.ships, 0.0).sum() / denom,
        ],
        dtype=jnp.float32,
    )
    fleet_ship_fractions = jnp.asarray(
        [
            jnp.where(my_fleet, fleets.ships, 0.0).sum() / denom,
            jnp.where(enemy_fleet, fleets.ships, 0.0).sum() / denom,
        ],
        dtype=jnp.float32,
    )
    angular_velocity = jnp.asarray(
        [game.angular_velocity.astype(jnp.float32) / ANGULAR_VELOCITY_NORM],
        dtype=jnp.float32,
    )

    return GlobalAssemblyContext(
        step_fraction=step_fraction,
        planet_fractions=planet_fractions,
        ship_fractions=ship_fractions,
        fleet_ship_fractions=fleet_ship_fractions,
        owner_counts=owner_counts,
        owner_ships=owner_ships,
        owner_fleets=owner_fleets,
        active_mask=active_mask,
        player_count_feature=player_count_feature,
        owner_production=owner_production,
        ship_delta_slots=(owner_ships - previous_global[ship_totals_slice])
        * previous_present,
        planet_delta_slots=(owner_counts - previous_global[planet_counts_slice])
        * previous_present,
        fleet_delta_slots=(owner_fleets - previous_global[fleet_totals_slice])
        * previous_present,
        production_delta_slots=(owner_production - previous_global[production_slice])
        * previous_present,
        angular_velocity=angular_velocity,
    )


def assemble_global_frame(context: GlobalAssemblyContext) -> jnp.ndarray:
    return GLOBAL_FEATURE_CATALOG.assemble(context)
