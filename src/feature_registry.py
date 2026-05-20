from dataclasses import dataclass
from typing import Sequence

from src.conf_schema import EnvConfig
from src.constants import (
    BASE_CANDIDATE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_DIM,
    BASE_SELF_FEATURE_DIM,
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


# Self Features Schema (Base: 30 elements)
SELF_FEATURE_REGISTRY = [
    FeatureItem("bias", size=1, active=True),
    FeatureItem(
        "source_coords", size=2, active=True
    ),  # Combines x and y position fields
    FeatureItem("source_radius", size=1, active=True),
    FeatureItem("source_ships", size=1, active=True),
    FeatureItem("source_production", size=1, active=True),
    FeatureItem("rotating_planet_flag", size=1, active=True),
    FeatureItem("friendly_planet_count", size=1, active=True),
    FeatureItem("enemy_planet_count", size=1, active=True),
    FeatureItem("total_friendly_ships", size=1, active=True),
    FeatureItem("total_enemy_ships", size=1, active=True),
    FeatureItem("owner_relative_planet_counts", size=4, active=True),
    FeatureItem("owner_relative_ship_totals", size=4, active=True),
    FeatureItem("active_owner_mask", size=4, active=True),
    FeatureItem("player_count", size=1, active=True),
    FeatureItem("ship_delta", size=1, active=True),
    FeatureItem("history_present_flag", size=1, active=True),
    FeatureItem("ownership_stable_flag", size=1, active=True),
    FeatureItem("outgoing_friendly_ships", size=1, active=True),
    FeatureItem("incoming_friendly_pressure", size=1, active=True),
    FeatureItem("incoming_enemy_pressure", size=1, active=True),
]
SELF_FEATURE_SCHEMA = FeatureGroupRegistry(SELF_FEATURE_REGISTRY)

# Candidate Features Schema (Base: 24 elements)
CANDIDATE_FEATURE_REGISTRY = [
    FeatureItem("bias", size=1, active=True),
    FeatureItem(
        "target_ownership_flags", size=3, active=True
    ),  # Neutral, Friendly, Enemy flags
    FeatureItem("target_coords", size=2, active=True),  # Target x, y positions
    FeatureItem("delta_coords", size=2, active=True),  # Delta x, y vectors
    FeatureItem("distance_to_target", size=1, active=True),
    FeatureItem("target_ships", size=1, active=True),
    FeatureItem("target_production", size=1, active=True),
    FeatureItem("target_is_rotating", size=1, active=True),
    FeatureItem("shot_crosses_sun", size=1, active=True),
    FeatureItem("source_ships", size=1, active=True),
    FeatureItem("relative_owner_slots", size=4, active=True),
    FeatureItem("turns_to_arrival", size=1, active=True),
    FeatureItem("incoming_friendly_pressure", size=1, active=True),
    FeatureItem("incoming_enemy_pressure", size=1, active=True),
    FeatureItem("target_ship_delta", size=1, active=True),
    FeatureItem("owner_changed_flag", size=1, active=True),
    FeatureItem("always_on_marker", size=1, active=True),
]
CANDIDATE_FEATURE_SCHEMA = FeatureGroupRegistry(CANDIDATE_FEATURE_REGISTRY)

# Global Features Schema (Base: 45 elements)
GLOBAL_FEATURE_REGISTRY = [
    FeatureItem("step_fraction", size=1, active=True),
    FeatureItem(
        "planet_fractions", size=3, active=True
    ),  # Friendly, Enemy, Neutral fractions
    FeatureItem(
        "ship_fractions", size=2, active=True
    ),  # Friendly, Enemy ship fractions
    FeatureItem(
        "fleet_ship_fractions", size=2, active=True
    ),  # Friendly, Enemy fleet fractions
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
]
GLOBAL_FEATURE_SCHEMA = FeatureGroupRegistry(GLOBAL_FEATURE_REGISTRY)


def _validate_schema_dim(
    name: str, schema: FeatureGroupRegistry, expected_dim: int
) -> None:
    if schema.base_dim != expected_dim:
        raise ValueError(
            f"{name} schema base_dim {schema.base_dim} does not match expected {expected_dim}"
        )


_validate_schema_dim("self", SELF_FEATURE_SCHEMA, expected_dim=BASE_SELF_FEATURE_DIM)
_validate_schema_dim(
    "candidate", CANDIDATE_FEATURE_SCHEMA, expected_dim=BASE_CANDIDATE_FEATURE_DIM
)
_validate_schema_dim(
    "global", GLOBAL_FEATURE_SCHEMA, expected_dim=BASE_GLOBAL_FEATURE_DIM
)


# -- env-aware factories --
def feature_history_steps(env_cfg: EnvConfig | None = None) -> int:
    if env_cfg is None:
        return 1
    return max(1, int(getattr(env_cfg, "feature_history_steps", 1)))


def self_feature_schema(env_cfg: EnvConfig | None = None) -> FeatureGroupRegistry:
    return SELF_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def self_feature_dim(env_cfg: EnvConfig | None = None) -> int:
    return self_feature_schema(env_cfg).total_dim


def candidate_feature_schema(env_cfg: EnvConfig | None = None) -> FeatureGroupRegistry:
    return CANDIDATE_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def candidate_feature_dim(env_cfg: EnvConfig | None = None) -> int:
    return candidate_feature_schema(env_cfg).total_dim


def global_feature_schema(env_cfg: EnvConfig | None = None) -> FeatureGroupRegistry:
    return GLOBAL_FEATURE_SCHEMA.with_history(feature_history_steps(env_cfg))


def global_feature_dim(env_cfg: EnvConfig | None = None) -> int:
    return global_feature_schema(env_cfg).total_dim
