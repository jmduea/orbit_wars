import jax
import jax.numpy as jnp
import numpy as np

from src.config import RewardConfig, TaskConfig
from src.features.registry import edge_k, global_feature_dim, planet_feature_dim
from src.features.registry import edge_feature_dim
from src.game.constants import MAX_PLANETS
from src.jax.env import batched_reset, empty_action, reset, step
from src.jax.features import TurnBatch


def test_jax_reset_is_deterministic_for_identical_key():
    cfg = TaskConfig(max_fleets=32, candidate_count=6)
    key = jax.random.PRNGKey(123)

    state_a, batch_a = reset(key, cfg)
    state_b, batch_b = reset(key, cfg)

    np.testing.assert_allclose(
        np.asarray(state_a.game.planets.x), np.asarray(state_b.game.planets.x)
    )
    np.testing.assert_array_equal(
        np.asarray(batch_a.planet_features), np.asarray(batch_b.planet_features)
    )


def test_jax_batched_reset_and_step_shapes():
    cfg = TaskConfig(max_fleets=16, candidate_count=5)
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    k = edge_k(cfg)

    states, batches = batched_reset(keys, cfg)
    assert states.game.planets.x.shape == (3, MAX_PLANETS)
    assert batches.planet_features.shape == (3, MAX_PLANETS, planet_feature_dim(cfg))
    assert batches.edge_features.shape == (3, MAX_PLANETS, k, edge_feature_dim(cfg))
    assert batches.global_features.shape == (3, global_feature_dim(cfg))

    action = empty_action(cfg)
    batched_action = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), action)
    next_states, results = batched_step(
        states, batched_action, batched_action, cfg, RewardConfig()
    )
    assert next_states.game.step.shape == (3,)
    assert results.reward.shape == (3,)
    assert isinstance(results.batch, TurnBatch)
