import numpy as np
import jax
import jax.numpy as jnp

from src.config import EnvConfig
from src.features import (
    NO_OP_CANDIDATE_INDEX,
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)
from src.jax_env import (
    JaxAction,
    batched_reset,
    batched_step,
    empty_action,
    reset,
    step,
)


def test_jax_reset_is_deterministic_for_identical_key():
    cfg = EnvConfig(max_planets=16, max_fleets=32, candidate_count=6)
    key = jax.random.PRNGKey(123)

    state_a, batch_a = reset(key, cfg)
    state_b, batch_b = reset(key, cfg)

    np.testing.assert_allclose(
        np.asarray(state_a.game.planets.x), np.asarray(state_b.game.planets.x)
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.game.planets.owner), np.asarray(state_b.game.planets.owner)
    )
    np.testing.assert_allclose(
        np.asarray(batch_a.self_features), np.asarray(batch_b.self_features)
    )


def test_jax_batched_reset_and_step_shapes():
    cfg = EnvConfig(max_planets=12, max_fleets=16, candidate_count=5)
    keys = jax.random.split(jax.random.PRNGKey(7), 3)

    states, batches = batched_reset(keys, cfg)
    assert states.game.planets.x.shape == (3, cfg.max_planets)
    assert batches.self_features.shape == (3, cfg.max_planets, self_feature_dim())
    assert batches.candidate_features.shape == (
        3,
        cfg.max_planets,
        cfg.candidate_count,
        candidate_feature_dim(),
    )
    assert batches.global_features.shape == (3, cfg.max_planets, global_feature_dim())
    assert batches.candidate_mask.shape == (3, cfg.max_planets, cfg.candidate_count)

    action = empty_action(cfg)
    batched_action = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), action)
    next_states, results = batched_step(states, batched_action, batched_action, cfg)
    assert next_states.game.step.shape == (3,)
    assert results.reward.shape == (3,)
    assert results.batch.self_features.shape == (3, cfg.max_planets, self_feature_dim())


def test_noop_candidate_slot_is_valid_for_owned_planets_only():
    cfg = EnvConfig(max_planets=12, max_fleets=16, candidate_count=4)
    _state, batch = reset(jax.random.PRNGKey(0), cfg)
    masks = np.asarray(batch.candidate_mask)
    decision_mask = np.asarray(batch.decision_mask).astype(bool)

    assert masks[decision_mask, NO_OP_CANDIDATE_INDEX].all()
    assert not masks[~decision_mask, NO_OP_CANDIDATE_INDEX].any()


def test_launch_and_production_match_core_orbit_wars_mechanics():
    cfg = EnvConfig(max_planets=12, max_fleets=16, candidate_count=4)
    state, _batch = reset(jax.random.PRNGKey(0), cfg)
    owners = np.asarray(state.game.planets.owner)
    source_id = int(np.flatnonzero(owners == 0)[0])
    ships_before = float(np.asarray(state.game.planets.ships)[source_id])
    production = float(np.asarray(state.game.planets.production)[source_id])

    action = empty_action(cfg)
    action = JaxAction(
        source_id=action.source_id.at[0].set(source_id),
        angle=action.angle.at[0].set(0.0),
        ships=action.ships.at[0].set(3.0),
        valid=action.valid.at[0].set(True),
    )
    next_state, _result = step(state, action, empty_action(cfg), cfg)

    ships_after = float(np.asarray(next_state.game.planets.ships)[source_id])
    assert ships_after == ships_before - 3.0 + production
    assert int(np.asarray(next_state.game.fleets.active).sum()) == 1
