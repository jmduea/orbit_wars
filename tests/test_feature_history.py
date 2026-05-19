import jax.numpy as jnp
import numpy as np

from src.config import EnvConfig
from src.features import (
    BASE_CANDIDATE_FEATURE_DIM,
    BASE_SELF_FEATURE_DIM,
    FeatureHistoryBuffer,
    build_feature_snapshot,
    candidate_feature_dim,
    encode_turn,
    self_feature_dim,
)
from src.game_types import GameState, PlanetState
from src.jax_env import JaxFleetState, JaxGameState, JaxPlanetState, max_fleets
from src.jax_features import append_feature_history, empty_feature_history
from src.jax_features import encode_turn as encode_jax_turn


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
    no_history_current = encode_turn(
        _state(1, 30), EnvConfig(candidate_count=2)
    ).self_features[0]
    np.testing.assert_allclose(current_slice[:24], no_history_current[:24])
    # The stacked current slice also exposes temporal planning signals derived
    # from the retained prior source row.
    np.testing.assert_allclose(current_slice[24], (30 - 10) / cfg.max_ships)
    assert current_slice[25] == 1.0
    assert current_slice[26] == 1.0
    np.testing.assert_allclose(current_slice[27:], no_history_current[27:])


def _three_planet_state(step: int, target_positions: dict[int, float]) -> GameState:
    return GameState(
        step=step,
        player=0,
        planets=[
            PlanetState(0, 0, 10.0, 10.0, 2.0, 50, 1),
            PlanetState(1, 1, target_positions[1], 10.0, 2.0, 20, 1),
            PlanetState(2, 1, target_positions[2], 10.0, 2.0, 30, 1),
        ],
        fleets=[],
    )


def test_python_candidate_history_aligns_by_source_and_target_id_after_reorder():
    cfg = EnvConfig(candidate_count=3, feature_history_steps=2)
    history = FeatureHistoryBuffer(max_steps=cfg.feature_history_steps - 1)

    first = encode_turn(
        _three_planet_state(0, {1: 20.0, 2: 30.0}), cfg, feature_history=history
    )
    assert first.contexts[0].candidate_ids == [-1, 1, 2]
    history.append(build_feature_snapshot(first))

    second = encode_turn(
        _three_planet_state(1, {1: 40.0, 2: 15.0}), cfg, feature_history=history
    )
    assert second.contexts[0].candidate_ids == [-1, 2, 1]
    assert second.candidate_features.shape == (
        1,
        cfg.candidate_count,
        candidate_feature_dim(cfg),
    )

    prior_target_2 = second.candidate_features[0, 1, :BASE_CANDIDATE_FEATURE_DIM]
    np.testing.assert_allclose(
        prior_target_2,
        first.candidate_features[0, 2, -BASE_CANDIDATE_FEATURE_DIM:],
    )
    np.testing.assert_allclose(
        second.candidate_features[0, 0, :BASE_CANDIDATE_FEATURE_DIM],
        np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32),
    )
    assert prior_target_2[-1] == 1.0


def test_python_candidate_history_zeros_missing_prior_targets():
    cfg = EnvConfig(candidate_count=2, feature_history_steps=2)
    history = FeatureHistoryBuffer(max_steps=cfg.feature_history_steps - 1)

    first = encode_turn(
        _three_planet_state(0, {1: 20.0, 2: 30.0}), cfg, feature_history=history
    )
    assert first.contexts[0].candidate_ids == [-1, 1]
    history.append(build_feature_snapshot(first))

    second = encode_turn(
        _three_planet_state(1, {1: 40.0, 2: 15.0}), cfg, feature_history=history
    )
    assert second.contexts[0].candidate_ids == [-1, 2]
    np.testing.assert_allclose(
        second.candidate_features[0, 1, :BASE_CANDIDATE_FEATURE_DIM],
        np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32),
    )


def _jax_three_planet_game(
    cfg: EnvConfig, step: int, target_positions: dict[int, float]
):
    active = jnp.array([True, True, True, False], dtype=bool)
    planets = JaxPlanetState(
        id=jnp.arange(cfg.max_planets, dtype=jnp.int32),
        owner=jnp.array([0, 1, 1, -1], dtype=jnp.int32),
        x=jnp.array(
            [10.0, target_positions[1], target_positions[2], 0.0], dtype=jnp.float32
        ),
        y=jnp.array([10.0, 10.0, 10.0, 0.0], dtype=jnp.float32),
        radius=jnp.ones((cfg.max_planets,), dtype=jnp.float32) * 2.0,
        ships=jnp.array([50.0, 20.0, 30.0, 0.0], dtype=jnp.float32),
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
    return JaxGameState(
        step=jnp.array(step, dtype=jnp.int32),
        player=jnp.array(0, dtype=jnp.int32),
        angular_velocity=jnp.array(0.0, dtype=jnp.float32),
        next_fleet_id=jnp.array(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=fleets,
    )


def test_jax_candidate_history_aligns_by_source_and_target_id_after_reorder():
    cfg = EnvConfig(
        max_planets=4, max_fleets=4, candidate_count=3, feature_history_steps=2
    )
    empty_history = empty_feature_history(cfg)
    first_game = _jax_three_planet_game(cfg, 0, {1: 20.0, 2: 30.0})
    first = encode_jax_turn(first_game, cfg, empty_history)
    np.testing.assert_array_equal(np.asarray(first.candidate_ids[0]), [-1, 1, 2])

    history = append_feature_history(empty_history, first_game, cfg)
    second_game = _jax_three_planet_game(cfg, 1, {1: 40.0, 2: 15.0})
    second = encode_jax_turn(second_game, cfg, history)
    np.testing.assert_array_equal(np.asarray(second.candidate_ids[0]), [-1, 2, 1])

    prior_target_2 = np.asarray(
        second.candidate_features[0, 1, :BASE_CANDIDATE_FEATURE_DIM]
    )
    np.testing.assert_allclose(
        prior_target_2,
        np.asarray(first.candidate_features[0, 2, -BASE_CANDIDATE_FEATURE_DIM:]),
    )
    np.testing.assert_allclose(
        np.asarray(second.candidate_features[0, 0, :BASE_CANDIDATE_FEATURE_DIM]),
        np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32),
    )
    assert prior_target_2[-1] == 1.0


def test_jax_candidate_history_zeros_missing_prior_targets():
    cfg = EnvConfig(
        max_planets=4, max_fleets=4, candidate_count=2, feature_history_steps=2
    )
    empty_history = empty_feature_history(cfg)
    first_game = _jax_three_planet_game(cfg, 0, {1: 20.0, 2: 30.0})
    first = encode_jax_turn(first_game, cfg, empty_history)
    np.testing.assert_array_equal(np.asarray(first.candidate_ids[0]), [-1, 1])

    history = append_feature_history(empty_history, first_game, cfg)
    second_game = _jax_three_planet_game(cfg, 1, {1: 40.0, 2: 15.0})
    second = encode_jax_turn(second_game, cfg, history)
    np.testing.assert_array_equal(np.asarray(second.candidate_ids[0]), [-1, 2])
    np.testing.assert_allclose(
        np.asarray(second.candidate_features[0, 1, :BASE_CANDIDATE_FEATURE_DIM]),
        np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32),
    )
