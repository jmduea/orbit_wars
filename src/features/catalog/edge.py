"""Edge feature catalog — single-source declarations and row assembly."""

from __future__ import annotations

import jax.numpy as jnp

from src.features.catalog._core import FeatureCatalog
from src.features.catalog._types import (
    EdgeRowAssemblyContext,
    FeatureCatalogEntry,
    FeatureDefinition,
)


def _feat_delta_coords(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return jnp.stack([ctx.delta_x, ctx.delta_y], axis=-1)


def _feat_distance(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.distance[..., None]


def _feat_sun_crossing(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.crosses.astype(jnp.float32)[..., None]


def _feat_target_ships(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (jnp.minimum(ctx.tgt_ships, ctx.scale) / ctx.scale)[..., None]


def _feat_target_owner_slot(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.owner_slot


def _feat_turns_to_arrival(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.turns[..., None]


def _feat_target_incoming_friendly(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_friendly / ctx.scale)[..., None]


def _feat_target_incoming_enemy(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_enemy / ctx.scale)[..., None]


EDGE_FEATURE_ENTRIES: tuple[FeatureCatalogEntry, ...] = (
    FeatureCatalogEntry(FeatureDefinition("delta_coords", size=2), _feat_delta_coords),
    FeatureCatalogEntry(FeatureDefinition("distance"), _feat_distance),
    FeatureCatalogEntry(FeatureDefinition("sun_crossing"), _feat_sun_crossing),
    FeatureCatalogEntry(FeatureDefinition("target_ships"), _feat_target_ships),
    FeatureCatalogEntry(
        FeatureDefinition("target_owner_slot", size=4), _feat_target_owner_slot
    ),
    FeatureCatalogEntry(FeatureDefinition("turns_to_arrival"), _feat_turns_to_arrival),
    FeatureCatalogEntry(
        FeatureDefinition("target_incoming_friendly"), _feat_target_incoming_friendly
    ),
    FeatureCatalogEntry(
        FeatureDefinition("target_incoming_enemy"), _feat_target_incoming_enemy
    ),
)

EDGE_FEATURE_CATALOG = FeatureCatalog(EDGE_FEATURE_ENTRIES)


def assemble_edge_rows(context: EdgeRowAssemblyContext) -> jnp.ndarray:
    """Assemble edge rows and zero invalid targets."""

    rows = EDGE_FEATURE_CATALOG.assemble(context)
    rows = jnp.where(context.ordered_valid[..., None], rows, 0.0)
    return jnp.where(context.tgt_active[..., None], rows, 0.0)
