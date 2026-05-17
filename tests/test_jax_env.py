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
from src.jax_features import encode_turn
from src.jax_env import (
    JaxAction,
    JaxFleetState,
    JaxGameState,
    JaxPlanetState,
    batched_reset,
    batched_step,
    empty_action,
    max_fleets,
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


def test_jax_candidates_are_sorted_by_distance_before_id_tiebreaker():
    cfg = EnvConfig(max_planets=16, max_fleets=16, candidate_count=3)
    planet_ids = jnp.arange(cfg.max_planets, dtype=jnp.int32)
    owner = jnp.full((cfg.max_planets,), -1, dtype=jnp.int32)
    owner = owner.at[0].set(0)
    owner = owner.at[1].set(1)
    owner = owner.at[15].set(1)
    active = jnp.zeros((cfg.max_planets,), dtype=bool)
    active = active.at[jnp.array([0, 1, 15])].set(True)
    x = jnp.zeros((cfg.max_planets,), dtype=jnp.float32)
    y = jnp.zeros((cfg.max_planets,), dtype=jnp.float32)
    x = x.at[0].set(10.0)
    y = y.at[0].set(10.0)
    x = x.at[15].set(11.1)
    y = y.at[15].set(10.0)
    x = x.at[1].set(11.9)
    y = y.at[1].set(10.0)
    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=jnp.ones((cfg.max_planets,), dtype=jnp.float32),
        ships=jnp.ones((cfg.max_planets,), dtype=jnp.float32) * 10.0,
        production=jnp.ones((cfg.max_planets,), dtype=jnp.float32),
        active=active,
    )
    fleet_count = max_fleets(cfg)
    fleets = JaxFleetState(
        id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        owner=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        x=jnp.zeros((fleet_count,), dtype=jnp.float32),
        y=jnp.zeros((fleet_count,), dtype=jnp.float32),
        angle=jnp.zeros((fleet_count,), dtype=jnp.float32),
        from_planet_id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        ships=jnp.zeros((fleet_count,), dtype=jnp.float32),
        active=jnp.zeros((fleet_count,), dtype=bool),
    )
    game = JaxGameState(
        step=jnp.array(0, dtype=jnp.int32),
        player=jnp.array(0, dtype=jnp.int32),
        angular_velocity=jnp.array(0.0, dtype=jnp.float32),
        next_fleet_id=jnp.array(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=fleets,
    )

    encoded = encode_turn(game, cfg)

    np.testing.assert_array_equal(
        np.asarray(encoded.candidate_ids[0]), np.array([-1, 15, 1])
    )


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
