from src.features.catalog import (
    EDGE_FEATURE_CATALOG,
    GLOBAL_FEATURE_CATALOG,
    PLANET_FEATURE_CATALOG,
)
from src.features.registry import (
    EDGE_FEATURE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    PLANET_FEATURE_SCHEMA,
    edge_k,
)


def test_schema_base_dims_match_catalogs() -> None:
    assert PLANET_FEATURE_SCHEMA.base_dim == PLANET_FEATURE_CATALOG.base_dim
    assert EDGE_FEATURE_SCHEMA.base_dim == EDGE_FEATURE_CATALOG.base_dim
    assert GLOBAL_FEATURE_SCHEMA.base_dim == GLOBAL_FEATURE_CATALOG.base_dim


def test_global_schema_includes_angular_velocity_slice() -> None:
    angular_slice = GLOBAL_FEATURE_SCHEMA.base_slice("angular_velocity")
    assert angular_slice == GLOBAL_FEATURE_CATALOG.base_slice("angular_velocity")


def test_edge_k_from_candidate_count() -> None:
    from src.config import TaskConfig

    assert edge_k(TaskConfig(candidate_count=4)) == 3
    assert edge_k(TaskConfig(candidate_count=1)) == 0
