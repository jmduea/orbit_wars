from src.features.catalog import GLOBAL_FEATURE_CATALOG, PLANET_FEATURE_CATALOG
from src.features.catalog.edge import (
    DEFAULT_INTERCEPT_ANCHORS,
    EDGE_FEATURE_CATALOG,
    expected_edge_feature_dim,
)
from src.features.registry import (
    EDGE_FEATURE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    PLANET_FEATURE_SCHEMA,
    edge_feature_dim,
    edge_k,
)


def test_schema_base_dims_match_catalogs() -> None:
    assert PLANET_FEATURE_SCHEMA.base_dim == PLANET_FEATURE_CATALOG.base_dim
    assert EDGE_FEATURE_SCHEMA.base_dim == EDGE_FEATURE_CATALOG.base_dim
    assert GLOBAL_FEATURE_SCHEMA.base_dim == GLOBAL_FEATURE_CATALOG.base_dim


def test_edge_feature_dim_matches_parametric_catalog() -> None:
    from src.config import TaskConfig

    cfg = TaskConfig()
    assert edge_feature_dim(cfg) == expected_edge_feature_dim(
        len(cfg.intercept_anchors)
    )
    assert EDGE_FEATURE_CATALOG.base_dim == expected_edge_feature_dim(
        len(DEFAULT_INTERCEPT_ANCHORS)
    )


def test_edge_catalog_ordered_field_list_pin() -> None:
    expected_names_and_sizes = (
        ("intercept_delta_coords_s1", 2),
        ("intercept_distance_s1", 1),
        ("intercept_turns_s1", 1),
        ("sun_cross_at_intercept_s1", 1),
        ("target_ships_s1", 1),
        ("intercept_delta_coords_s3", 2),
        ("intercept_distance_s3", 1),
        ("intercept_turns_s3", 1),
        ("sun_cross_at_intercept_s3", 1),
        ("target_ships_s3", 1),
        ("intercept_delta_coords_s6", 2),
        ("intercept_distance_s6", 1),
        ("intercept_turns_s6", 1),
        ("sun_cross_at_intercept_s6", 1),
        ("target_ships_s6", 1),
        ("crosses_now", 1),
        ("target_owner_slot", 4),
        ("target_incoming_friendly", 1),
        ("target_incoming_enemy", 1),
    )
    actual = tuple(
        (entry.definition.name, entry.definition.size)
        for entry in EDGE_FEATURE_CATALOG.entries
        if entry.definition.active
    )
    assert actual == expected_names_and_sizes


def test_edge_schema_slice_lookup_for_new_feature_names() -> None:
    for name in (
        "intercept_delta_coords_s1",
        "intercept_distance_s1",
        "intercept_turns_s1",
        "sun_cross_at_intercept_s1",
        "intercept_delta_coords_s3",
        "intercept_distance_s3",
        "intercept_turns_s3",
        "sun_cross_at_intercept_s3",
        "intercept_delta_coords_s6",
        "intercept_distance_s6",
        "intercept_turns_s6",
        "sun_cross_at_intercept_s6",
        "crosses_now",
    ):
        sl = EDGE_FEATURE_SCHEMA.base_slice(name)
        assert sl.stop > sl.start


def test_global_schema_includes_angular_velocity_slice() -> None:
    angular_slice = GLOBAL_FEATURE_SCHEMA.base_slice("angular_velocity")
    assert angular_slice == GLOBAL_FEATURE_CATALOG.base_slice("angular_velocity")


def test_catalog_view_slices_match_catalog() -> None:
    for schema, catalog in (
        (PLANET_FEATURE_SCHEMA, PLANET_FEATURE_CATALOG),
        (EDGE_FEATURE_SCHEMA, EDGE_FEATURE_CATALOG),
        (GLOBAL_FEATURE_SCHEMA, GLOBAL_FEATURE_CATALOG),
    ):
        assert schema.names == tuple(d.name for d in catalog.definitions)
        assert schema.dim == schema.base_dim
        for name in schema.names:
            assert schema.base_slice(name) == catalog.base_slice(name)


def test_catalog_view_with_history_expands_dim() -> None:
    expanded = GLOBAL_FEATURE_SCHEMA.with_history(3)
    assert expanded.dim == GLOBAL_FEATURE_CATALOG.base_dim * 3
    base = GLOBAL_FEATURE_SCHEMA.base_slice("angular_velocity")
    assert expanded.slice("angular_velocity", frame=0) == base
    assert expanded.slice("angular_velocity", frame=1) == slice(
        base.start + expanded.base_dim,
        base.stop + expanded.base_dim,
    )


def test_edge_k_from_candidate_count() -> None:
    from src.config import TaskConfig

    assert edge_k(TaskConfig(candidate_count=4)) == 3
    assert edge_k(TaskConfig(candidate_count=1)) == 0
