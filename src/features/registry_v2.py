from src.config.schema import TaskConfig
from src.features.registry import FeatureGroupRegistry, FeatureItem
from src.game.constants import (
    BASE_EDGE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_V2_DIM,
    BASE_PLANET_FEATURE_DIM,
)

PLANET_FEATURE_REGISTRY = [
    FeatureItem("active", size=1, active=True),
    FeatureItem("orbit_radius", size=1, active=True),
    FeatureItem("orbit_angle", size=1, active=True),
    FeatureItem("radius", size=1, active=True),
    FeatureItem("ships", size=1, active=True),
    FeatureItem("production", size=1, active=True),
    FeatureItem("rotating_flag", size=1, active=True),
    FeatureItem("owner_slot", size=4, active=True),
    FeatureItem("incoming_friendly_pressure", size=1, active=True),
    FeatureItem("ship_delta", size=1, active=True),
]
PLANET_FEATURE_SCHEMA = FeatureGroupRegistry(PLANET_FEATURE_REGISTRY)

EDGE_FEATURE_REGISTRY = [
    FeatureItem("delta_coords", size=2, active=True),
    FeatureItem("distance", size=1, active=True),
    FeatureItem("sun_crossing", size=1, active=True),
    FeatureItem("target_ships", size=1, active=True),
    FeatureItem("target_owner_slot", size=4, active=True),
    FeatureItem("turns_to_arrival", size=1, active=True),
    FeatureItem("target_incoming_friendly", size=1, active=True),
    FeatureItem("target_incoming_enemy", size=1, active=True),
]
EDGE_FEATURE_SCHEMA = FeatureGroupRegistry(EDGE_FEATURE_REGISTRY)

GLOBAL_V2_FEATURE_REGISTRY = [
    FeatureItem("step_fraction", size=1, active=True),
    FeatureItem("planet_fractions", size=3, active=True),
    FeatureItem("ship_fractions", size=2, active=True),
    FeatureItem("fleet_ship_fractions", size=2, active=True),
    FeatureItem("owner_relative_planet_counts", size=4, active=True),
    FeatureItem("owner_relative_ship_totals", size=4, active=True),
    FeatureItem("owner_relative_fleet_totals", size=4, active=True),
    FeatureItem("active_owner_mask", size=4, active=True),
    FeatureItem("player_count", size=1, active=True),
    FeatureItem("owner_relative_production", size=4, active=True),
    FeatureItem("ship_delta_slots", size=4, active=True),
    FeatureItem("planet_delta_slots", size=4, active=True),
    FeatureItem("fleet_delta_slots", size=4, active=True),
    FeatureItem("production_delta_slots", size=4, active=True),
    FeatureItem("angular_velocity", size=1, active=True),
]
GLOBAL_V2_FEATURE_SCHEMA = FeatureGroupRegistry(GLOBAL_V2_FEATURE_REGISTRY)


def _validate_schema_dim(
    name: str, schema: FeatureGroupRegistry, expected_dim: int
) -> None:
    if schema.base_dim != expected_dim:
        raise ValueError(
            f"{name} schema base_dim {schema.base_dim} does not match expected {expected_dim}"
        )


_validate_schema_dim(
    "planet", PLANET_FEATURE_SCHEMA, expected_dim=BASE_PLANET_FEATURE_DIM
)
_validate_schema_dim("edge", EDGE_FEATURE_SCHEMA, expected_dim=BASE_EDGE_FEATURE_DIM)
_validate_schema_dim(
    "global_v2", GLOBAL_V2_FEATURE_SCHEMA, expected_dim=BASE_GLOBAL_FEATURE_V2_DIM
)


def feature_history_steps(env_cfg: TaskConfig | None = None) -> int:
    if env_cfg is None:
        return 1
    return max(1, int(getattr(env_cfg, "feature_history_steps", 1)))


def planet_feature_schema(env_cfg: TaskConfig | None = None) -> FeatureGroupRegistry:
    return PLANET_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def planet_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    return planet_feature_schema(env_cfg).total_dim


def edge_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    return EDGE_FEATURE_SCHEMA.base_dim


def edge_k(env_cfg: TaskConfig) -> int:
    return max(0, int(env_cfg.candidate_count) - 1)


def global_v2_feature_schema(env_cfg: TaskConfig | None = None) -> FeatureGroupRegistry:
    return GLOBAL_V2_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def global_v2_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    return global_v2_feature_schema(env_cfg).total_dim
