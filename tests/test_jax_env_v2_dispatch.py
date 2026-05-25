import jax
import pytest

from src.config import RewardConfig, TaskConfig
from src.features.registry_v2 import edge_k, global_v2_feature_dim
from src.game.constants import BASE_EDGE_FEATURE_DIM, BASE_PLANET_FEATURE_DIM, MAX_PLANETS
from src.jax.env import empty_action, reset, step
from src.jax.features_v2 import JaxTurnBatchV2


@pytest.mark.jax
def test_jax_reset_and_step_shapes_with_v2_encoding():
    cfg = TaskConfig(
        max_fleets=16,
        candidate_count=4,
        encoding_version="v2",
        ship_feature_scale=1000.0,
    )
    k = edge_k(cfg)
    state, batch = reset(jax.random.PRNGKey(21), cfg)
    assert isinstance(batch, JaxTurnBatchV2)
    assert batch.planet_features.shape == (MAX_PLANETS, BASE_PLANET_FEATURE_DIM)
    assert batch.edge_features.shape == (MAX_PLANETS, k, BASE_EDGE_FEATURE_DIM)
    assert batch.global_features.shape == (global_v2_feature_dim(cfg),)

    action = empty_action(cfg)
    next_state, result = step(state, action, action, cfg, RewardConfig())
    assert isinstance(result.batch, JaxTurnBatchV2)
    assert next_state.feature_history is not None
    assert next_state.game.step == state.game.step + 1
