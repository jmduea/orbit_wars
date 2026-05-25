from dataclasses import dataclass
from typing import Sequence

from src.config.schema import TaskConfig
from src.game.constants import (
    BASE_EDGE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_V2_DIM,
    BASE_PLANET_FEATURE_DIM,
)


@dataclass(frozen=True)
class FeatureItem:
    """Metadata for a single feature."""

    name: str
    size: int = 1
    active: bool = True


class FeatureGroupRegistry:
    """Ordered feature schema"""

    def __init__(self, features: Sequence[FeatureItem], history_steps: int = 1):
        self.all_features = tuple(features)
        self.history_steps = max(1, int(history_steps))
        self._active_features = tuple(
            feature for feature in self.all_features if feature.active
        )
        self._base_slices = self._build_slices(self._active_features)

    @staticmethod
    def _build_slices(features: Sequence[FeatureItem]) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        start = 0
        for feature in features:
            if feature.name in slices:
                raise ValueError(f"Duplicate feature name: {feature.name}")
            stop = start + feature.size
            slices[feature.name] = slice(start, stop)
            start = stop
        return slices

    @property
    def base_dim(self) -> int:
        return sum(feature.size for feature in self._active_features)

    @property
    def total_dim(self) -> int:
        return self.base_dim * self.history_steps

    def base_slice(self, feature_name: str) -> slice:
        """Return the feature's slice within a single base frame."""
        try:
            return self._base_slices[feature_name]
        except KeyError as exc:
            raise ValueError(
                f"Feature '{feature_name}' is not active in the registry"
            ) from exc

    def frame_slice(self, frame: int = -1) -> slice:
        """Return the full slice for one history frame.

        frame=0 is oldest, frame=-1 is current.
        """
        frame_index = self._normalize_frame(frame)
        start = frame_index * self.base_dim
        return slice(start, start + self.base_dim)

    def slice(self, feature_name: str, frame: int = -1) -> slice:
        """Return a feature slice in the history-expanded vector.

        Defaults to the current frame, matching how most callers inspect features.
        """
        base = self.base_slice(feature_name)
        frame_index = self._normalize_frame(frame)
        offset = frame_index * self.base_dim
        return slice(offset + base.start, offset + base.stop)

    def history_slices(self, feature_name: str) -> tuple[slice, ...]:  # ty: ignore
        """Return one slice per history frame for a feature."""
        return tuple(
            self.slice(feature_name, frame=frame) for frame in range(self.history_steps)
        )

    def with_history(self, history_steps: int) -> "FeatureGroupRegistry":
        """Return a copy of this registry with the given number of history steps."""
        return FeatureGroupRegistry(self.all_features, history_steps=history_steps)

    def _normalize_frame(self, frame: int) -> int:
        if frame < 0:
            frame += self.history_steps
            if frame < 0 or frame >= self.history_steps:
                raise IndexError(
                    f"History frame {frame} out of range for {self.history_steps} frames"
                )
        return frame


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

GLOBAL_FEATURE_REGISTRY = [
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
GLOBAL_FEATURE_SCHEMA = FeatureGroupRegistry(GLOBAL_FEATURE_REGISTRY)


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
    "global", GLOBAL_FEATURE_SCHEMA, expected_dim=BASE_GLOBAL_FEATURE_V2_DIM
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


def global_feature_schema(env_cfg: TaskConfig | None = None) -> FeatureGroupRegistry:
    return GLOBAL_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def global_feature_dim(env_cfg: TaskConfig | None = None) -> int:
    return global_feature_schema(env_cfg).total_dim
