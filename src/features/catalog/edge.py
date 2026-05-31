"""Edge feature catalog — parametric intercept anchors and row assembly."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from src.config.schema import TaskConfig
from src.features.catalog._core import FeatureCatalog
from src.features.catalog._types import (
    EdgeRowAssemblyContext,
    FeatureCatalogEntry,
    FeatureDefinition,
)

DEFAULT_INTERCEPT_ANCHORS: tuple[float, ...] = (1.0, 3.0, 6.0)


def intercept_anchor_label(speed: float) -> str:
    speed_f = float(speed)
    if speed_f.is_integer():
        return f"s{int(speed_f)}"
    return "s" + f"{speed_f:g}".replace(".", "p")


def expected_edge_feature_dim(num_anchors: int) -> int:
    """Edge width: five intercept fields plus target_ships per anchor, plus static tail."""

    return 6 * num_anchors + 7


def _feat_crosses_now(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.crosses_now.astype(jnp.float32)[..., None]


def _feat_target_owner_slot(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return ctx.owner_slot


def _feat_target_incoming_friendly(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_friendly / ctx.scale)[..., None]


def _feat_target_incoming_enemy(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
    return (ctx.incoming_enemy / ctx.scale)[..., None]


def _make_delta_coords(anchor_index: int):
    def compute(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
        return jnp.stack(
            [
                ctx.intercept_delta_x_per_anchor[..., anchor_index],
                ctx.intercept_delta_y_per_anchor[..., anchor_index],
            ],
            axis=-1,
        )

    return compute


def _make_per_anchor_field(field_name: str, anchor_index: int):
    def compute(ctx: EdgeRowAssemblyContext) -> jnp.ndarray:
        tensor = getattr(ctx, field_name)
        return tensor[..., anchor_index : anchor_index + 1]

    return compute


def build_edge_feature_catalog(
    intercept_anchors: Sequence[float],
) -> FeatureCatalog:
    entries: list[FeatureCatalogEntry] = []
    for anchor_index, speed in enumerate(intercept_anchors):
        label = intercept_anchor_label(float(speed))
        entries.extend(
            [
                FeatureCatalogEntry(
                    FeatureDefinition(f"intercept_delta_coords_{label}", size=2),
                    _make_delta_coords(anchor_index),
                ),
                FeatureCatalogEntry(
                    FeatureDefinition(f"intercept_distance_{label}"),
                    _make_per_anchor_field(
                        "intercept_distance_per_anchor", anchor_index
                    ),
                ),
                FeatureCatalogEntry(
                    FeatureDefinition(f"intercept_turns_{label}"),
                    _make_per_anchor_field("intercept_turns_per_anchor", anchor_index),
                ),
                FeatureCatalogEntry(
                    FeatureDefinition(f"sun_cross_at_intercept_{label}"),
                    _make_per_anchor_field(
                        "sun_cross_at_intercept_per_anchor", anchor_index
                    ),
                ),
                FeatureCatalogEntry(
                    FeatureDefinition(f"target_ships_{label}"),
                    _make_per_anchor_field("tgt_ships_per_anchor", anchor_index),
                ),
            ]
        )
    entries.extend(
        [
            FeatureCatalogEntry(FeatureDefinition("crosses_now"), _feat_crosses_now),
            FeatureCatalogEntry(
                FeatureDefinition("target_owner_slot", size=4), _feat_target_owner_slot
            ),
            FeatureCatalogEntry(
                FeatureDefinition("target_incoming_friendly"),
                _feat_target_incoming_friendly,
            ),
            FeatureCatalogEntry(
                FeatureDefinition("target_incoming_enemy"), _feat_target_incoming_enemy
            ),
        ]
    )
    return FeatureCatalog(tuple(entries))


def edge_feature_catalog_for(env_cfg: TaskConfig | None = None) -> FeatureCatalog:
    if env_cfg is None:
        return EDGE_FEATURE_CATALOG
    anchors = tuple(float(s) for s in env_cfg.intercept_anchors)
    if anchors == DEFAULT_INTERCEPT_ANCHORS:
        return EDGE_FEATURE_CATALOG
    return build_edge_feature_catalog(anchors)


EDGE_FEATURE_CATALOG = build_edge_feature_catalog(DEFAULT_INTERCEPT_ANCHORS)
assert EDGE_FEATURE_CATALOG.base_dim == expected_edge_feature_dim(
    len(DEFAULT_INTERCEPT_ANCHORS)
)


def assemble_edge_rows(
    context: EdgeRowAssemblyContext,
    *,
    catalog: FeatureCatalog | None = None,
) -> jnp.ndarray:
    """Assemble edge rows and zero invalid targets."""

    active_catalog = catalog or EDGE_FEATURE_CATALOG
    rows = active_catalog.assemble(context)
    rows = jnp.where(context.ordered_valid[..., None], rows, 0.0)
    return jnp.where(context.tgt_active[..., None], rows, 0.0)
