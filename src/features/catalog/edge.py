"""Edge feature catalog — single-source declarations and row assembly."""

from __future__ import annotations

import jax.numpy as jnp

from src.features.catalog._core import FeatureCatalog
from src.features.catalog._types import (
    EdgeRowAssemblyContext,
    FeatureCatalogEntry,
    FeatureDefinition,
)


def _feat_intercept_delta_coords_s1(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return jnp.stack(
        [
            ctx.intercept_delta_x_per_anchor[..., 0],
            ctx.intercept_delta_y_per_anchor[..., 0],
        ],
        axis=-1,
    )


def _feat_intercept_distance_s1(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.intercept_distance_per_anchor[..., 0:1]


def _feat_intercept_turns_s1(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.intercept_turns_per_anchor[..., 0:1]


def _feat_sun_cross_at_intercept_s1(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.sun_cross_at_intercept_per_anchor[..., 0:1].astype(jnp.float32)


def _feat_intercept_delta_coords_s6(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return jnp.stack(
        [
            ctx.intercept_delta_x_per_anchor[..., 1],
            ctx.intercept_delta_y_per_anchor[..., 1],
        ],
        axis=-1,
    )


def _feat_intercept_distance_s6(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.intercept_distance_per_anchor[..., 1:2]


def _feat_intercept_turns_s6(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.intercept_turns_per_anchor[..., 1:2]


def _feat_sun_cross_at_intercept_s6(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.sun_cross_at_intercept_per_anchor[..., 1:2].astype(jnp.float32)


def _feat_crosses_now(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.crosses_now.astype(jnp.float32)[..., None]


# TODO(M5): forward-projected target ships per anchor.
def _feat_target_ships(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (jnp.minimum(ctx.tgt_ships, ctx.scale) / ctx.scale)[..., None]


def _feat_target_owner_slot(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.owner_slot


def _feat_target_incoming_friendly(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_friendly / ctx.scale)[..., None]


def _feat_target_incoming_enemy(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_enemy / ctx.scale)[..., None]


EDGE_FEATURE_ENTRIES: tuple[FeatureCatalogEntry, ...] = (
    FeatureCatalogEntry(
        FeatureDefinition("intercept_delta_coords_s1", size=2),
        _feat_intercept_delta_coords_s1,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("intercept_distance_s1"), _feat_intercept_distance_s1
    ),
    FeatureCatalogEntry(
        FeatureDefinition("intercept_turns_s1"), _feat_intercept_turns_s1
    ),
    FeatureCatalogEntry(
        FeatureDefinition("sun_cross_at_intercept_s1"),
        _feat_sun_cross_at_intercept_s1,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("intercept_delta_coords_s6", size=2),
        _feat_intercept_delta_coords_s6,
    ),
    FeatureCatalogEntry(
        FeatureDefinition("intercept_distance_s6"), _feat_intercept_distance_s6
    ),
    FeatureCatalogEntry(
        FeatureDefinition("intercept_turns_s6"), _feat_intercept_turns_s6
    ),
    FeatureCatalogEntry(
        FeatureDefinition("sun_cross_at_intercept_s6"),
        _feat_sun_cross_at_intercept_s6,
    ),
    FeatureCatalogEntry(FeatureDefinition("crosses_now"), _feat_crosses_now),
    FeatureCatalogEntry(FeatureDefinition("target_ships"), _feat_target_ships),
    FeatureCatalogEntry(
        FeatureDefinition("target_owner_slot", size=4), _feat_target_owner_slot
    ),
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
