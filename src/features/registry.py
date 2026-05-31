"""Feature schema views and dimension helpers derived from catalogs."""

from __future__ import annotations

from src.config.schema import TaskConfig
from src.features.catalog import (
    EDGE_FEATURE_CATALOG,
    GLOBAL_FEATURE_CATALOG,
    PLANET_FEATURE_CATALOG,
)
from src.features.schema_api import CatalogView

PLANET_FEATURE_SCHEMA = CatalogView(PLANET_FEATURE_CATALOG)
EDGE_FEATURE_SCHEMA = CatalogView(EDGE_FEATURE_CATALOG)
GLOBAL_FEATURE_SCHEMA = CatalogView(GLOBAL_FEATURE_CATALOG)


def feature_history_steps(env_cfg: TaskConfig | None = None) -> int:
    if env_cfg is None:
        return 1
    return max(1, int(getattr(env_cfg, "feature_history_steps", 1)))


def planet_feature_schema(env_cfg: TaskConfig | None = None) -> CatalogView:
    return PLANET_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def planet_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    """Return per-planet vector width for encoders and ``TurnBatch``.

    Planet tensors are a single current frame; temporal history is stacked only
    on ``global_features`` (see ``_stack_global_history`` in ``src/jax/features.py``).
    """
    return planet_feature_schema(env_cfg).base_dim


def edge_feature_schema(env_cfg: TaskConfig | None = None) -> CatalogView:
    from src.features.catalog.edge import edge_feature_catalog_for

    return CatalogView(edge_feature_catalog_for(env_cfg))


def edge_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    return edge_feature_schema(env_cfg).base_dim


def edge_k(env_cfg: TaskConfig) -> int:
    return max(0, int(env_cfg.candidate_count) - 1)


def global_feature_schema(env_cfg: TaskConfig | None = None) -> CatalogView:
    return GLOBAL_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def global_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    return global_feature_schema(env_cfg).total_dim
