import numpy as np

from src.config import EnvConfig
from src.features import (
    BASE_SELF_FEATURE_DIM,
    FeatureHistoryBuffer,
    build_feature_snapshot,
    encode_turn,
    self_feature_dim,
)
from src.game_types import GameState, PlanetState


def _state(step: int, ships: int) -> GameState:
    return GameState(
        step=step,
        player=0,
        planets=[
            PlanetState(0, 0, 10.0, 10.0, 2.0, ships, 1),
            PlanetState(1, 1, 30.0, 10.0, 2.0, 20, 1),
        ],
        fleets=[],
    )


def test_python_feature_history_stacks_chronological_rows():
    cfg = EnvConfig(candidate_count=2, feature_history_steps=2)
    history = FeatureHistoryBuffer(max_steps=cfg.feature_history_steps - 1)

    first = encode_turn(_state(0, 10), cfg, feature_history=history)
    history.append(build_feature_snapshot(first))
    second = encode_turn(_state(1, 30), cfg, feature_history=history)

    assert second.self_features.shape == (1, self_feature_dim(cfg))
    previous_slice = second.self_features[0, :BASE_SELF_FEATURE_DIM]
    current_slice = second.self_features[0, BASE_SELF_FEATURE_DIM:]
    np.testing.assert_allclose(
        previous_slice, first.self_features[0, BASE_SELF_FEATURE_DIM:]
    )
    np.testing.assert_allclose(
        current_slice,
        encode_turn(_state(1, 30), EnvConfig(candidate_count=2)).self_features[0],
    )
