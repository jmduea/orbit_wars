import pytest

from src.constants import (
    BASE_CANDIDATE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_DIM,
    BASE_SELF_FEATURE_DIM,
)
from src.feature_registry import (
    CANDIDATE_FEATURE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    SELF_FEATURE_SCHEMA,
)


def test_schema_base_dims_match_constants() -> None:
    assert SELF_FEATURE_SCHEMA.base_dim == BASE_SELF_FEATURE_DIM
    assert CANDIDATE_FEATURE_SCHEMA.base_dim == BASE_CANDIDATE_FEATURE_DIM
    assert GLOBAL_FEATURE_SCHEMA.base_dim == BASE_GLOBAL_FEATURE_DIM


def test_self_schema_critical_slices() -> None:
    assert SELF_FEATURE_SCHEMA.slice("source_ships") == slice(4, 5)
    assert SELF_FEATURE_SCHEMA.slice("owner_relative_ship_totals") == slice(15, 19)


def test_candidate_schema_critical_slices() -> None:
    assert CANDIDATE_FEATURE_SCHEMA.slice("target_coords") == slice(4, 6)
    assert CANDIDATE_FEATURE_SCHEMA.slice("target_ships") == slice(9, 10)
    assert CANDIDATE_FEATURE_SCHEMA.slice("relative_owner_slots") == slice(14, 18)


def test_global_schema_critical_slices() -> None:
    assert GLOBAL_FEATURE_SCHEMA.slice("owner_relative_planet_counts") == slice(8, 12)
    assert GLOBAL_FEATURE_SCHEMA.slice("owner_relative_ship_totals") == slice(12, 16)
    assert GLOBAL_FEATURE_SCHEMA.slice("owner_relative_fleet_totals") == slice(16, 20)
    assert GLOBAL_FEATURE_SCHEMA.slice("owner_relative_production") == slice(25, 29)


def test_unknown_feature_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="not active"):
        SELF_FEATURE_SCHEMA.slice("does_not_exist")


def test_self_schema_history_slices_are_frame_offset() -> None:
    schema = SELF_FEATURE_SCHEMA.with_history(history_steps=3)

    assert schema.base_dim == BASE_SELF_FEATURE_DIM
    assert schema.total_dim == BASE_SELF_FEATURE_DIM * 3

    assert schema.base_slice("source_ships") == slice(4, 5)
    assert schema.slice("source_ships", frame=0) == slice(4, 5)
    assert schema.slice("source_ships", frame=1) == slice(34, 35)
    assert schema.slice("source_ships", frame=-1) == slice(64, 65)

    assert schema.history_slices("source_ships") == (
        slice(4, 5),
        slice(34, 35),
        slice(64, 65),
    )
