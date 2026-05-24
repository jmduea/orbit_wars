from __future__ import annotations

import numpy as np

from src.features import TurnBatch
from src.game.types import GameState
from src.features.normalization import ObservationNormalizer


def _batch(scale: float) -> TurnBatch:
    return TurnBatch(
        self_features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32) * scale,
        candidate_features=np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, 2.0]],
            ],
            dtype=np.float32,
        )
        * scale,
        global_features=np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32) * scale,
        candidate_mask=np.asarray([[True, False], [True, True]], dtype=bool),
        contexts=[],
        state=GameState(player=0, step=0, planets=[], fleets=[]),
    )


def test_observation_normalizer_round_trip_and_normalization() -> None:
    normalizer = ObservationNormalizer(clip=5.0)
    normalizer.update(_batch(scale=1.0))
    normalizer.update(_batch(scale=2.0))

    normalized = normalizer.normalize_batch(_batch(scale=1.5))
    assert normalized.self_features.shape == (2, 2)
    assert np.isfinite(normalized.self_features).all()
    assert np.max(np.abs(normalized.self_features)) <= 5.0 + 1e-6

    clone = ObservationNormalizer()
    clone.load_state_dict(normalizer.state_dict())
    cloned_norm = clone.normalize_batch(_batch(scale=1.5))
    np.testing.assert_allclose(normalized.self_features, cloned_norm.self_features)
    np.testing.assert_allclose(
        normalized.candidate_features, cloned_norm.candidate_features
    )
    np.testing.assert_allclose(normalized.global_features, cloned_norm.global_features)
