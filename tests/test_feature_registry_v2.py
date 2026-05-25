from src.features.registry_v2 import (
    EDGE_FEATURE_SCHEMA,
    GLOBAL_V2_FEATURE_SCHEMA,
    PLANET_FEATURE_SCHEMA,
    edge_k,
)
from src.game.constants import (
    BASE_EDGE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_V2_DIM,
    BASE_PLANET_FEATURE_DIM,
)


def test_v2_schema_base_dims_match_constants() -> None:
    assert PLANET_FEATURE_SCHEMA.base_dim == BASE_PLANET_FEATURE_DIM
    assert EDGE_FEATURE_SCHEMA.base_dim == BASE_EDGE_FEATURE_DIM
    assert GLOBAL_V2_FEATURE_SCHEMA.base_dim == BASE_GLOBAL_FEATURE_V2_DIM


def test_v2_global_schema_includes_angular_velocity_slice() -> None:
    assert GLOBAL_V2_FEATURE_SCHEMA.slice("angular_velocity") == slice(45, 46)


def test_edge_k_from_candidate_count() -> None:
    from src.config import TaskConfig

    assert edge_k(TaskConfig(candidate_count=4)) == 3
    assert edge_k(TaskConfig(candidate_count=1)) == 0
