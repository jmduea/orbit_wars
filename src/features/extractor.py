"""Python-side feature extraction wrapping JAX encode_turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.schema import TaskConfig
from src.features.registry import (
    edge_feature_dim,
    edge_k,
    global_feature_dim,
    planet_feature_dim,
)
from src.jax.features import (
    FeatureHistory,
    TurnBatch,
    append_feature_history,
    empty_feature_history,
    encode_turn,
)
from src.jax.submission_runtime import jax_game_from_observation


@dataclass(frozen=True)
class FeatureSchema:
    planet_feature_dim: int
    edge_feature_dim: int
    global_feature_dim: int
    edge_k: int


@dataclass(frozen=True)
class ExtractedFeatures:
    batch: TurnBatch
    schema: FeatureSchema
    game: Any


def coerce_to_jax_game(observation: Any, *, max_fleet_slots: int | None = None):
    """Convert an observation payload into a JAX game state."""

    return jax_game_from_observation(observation, max_fleet_slots=max_fleet_slots)


class FeatureExtractor:
    """Encode observations through the canonical planet-edge JAX feature path."""

    def __init__(self, env_cfg: TaskConfig):
        self.env_cfg = env_cfg
        self.schema = FeatureSchema(
            planet_feature_dim=planet_feature_dim(env_cfg),
            edge_feature_dim=edge_feature_dim(env_cfg),
            global_feature_dim=global_feature_dim(env_cfg),
            edge_k=edge_k(env_cfg),
        )

    def empty_history(self) -> FeatureHistory:
        return empty_feature_history(self.env_cfg)

    def extract(
        self,
        observation: Any,
        *,
        history: FeatureHistory | None = None,
        max_fleet_slots: int | None = None,
    ) -> ExtractedFeatures:
        game = coerce_to_jax_game(observation, max_fleet_slots=max_fleet_slots)
        batch = encode_turn(game, self.env_cfg, history)
        return ExtractedFeatures(batch=batch, schema=self.schema, game=game)

    def append_history(
        self,
        history: FeatureHistory | None,
        observation: Any,
        *,
        max_fleet_slots: int | None = None,
    ) -> FeatureHistory:
        game = coerce_to_jax_game(observation, max_fleet_slots=max_fleet_slots)
        return append_feature_history(history, game, self.env_cfg)
