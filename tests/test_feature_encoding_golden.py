import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TaskConfig, compose_hydra_train_config
from src.features.registry import (
    edge_feature_dim,
    edge_k,
    global_feature_dim,
    planet_feature_dim,
)
from src.game.constants import MAX_PLANETS
from src.jax.env import (
    JaxFleetState,
    JaxGameState,
    JaxPlanetState,
    batched_reset,
    reset,
)
from src.jax.features import (
    append_feature_history,
    empty_feature_history,
    encode_turn,
)
from src.jax.map_pool.comets import empty_comet_state


def _cfg(**kwargs) -> TaskConfig:
    base = dict(
        max_fleets=32,
        candidate_count=4,
        player_count=2,
        feature_history_steps=1,
        ship_feature_scale=1000.0,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def test_compose_hydra_includes_ship_feature_scale() -> None:
    cfg = compose_hydra_train_config()
    assert cfg.task.ship_feature_scale == 1000.0


def test_encode_v2_shapes_on_2p_reset() -> None:
    cfg = _cfg(player_count=2)
    state, _ = reset(jax.random.PRNGKey(11), cfg)
    batch = encode_turn(state.game, cfg)
    k = edge_k(cfg)

    assert batch.planet_features.shape == (MAX_PLANETS, planet_feature_dim(cfg))
    assert batch.planet_mask.shape == (MAX_PLANETS,)
    assert batch.edge_features.shape == (MAX_PLANETS, k, edge_feature_dim(cfg))
    assert batch.edge_mask.shape == (MAX_PLANETS, k)
    assert batch.edge_src_ids.shape == (MAX_PLANETS,)
    assert batch.edge_tgt_ids.shape == (MAX_PLANETS, k)
    assert batch.global_features.shape == (global_feature_dim(cfg),)
    assert batch.theta_ref.shape == ()


def test_encode_v2_shapes_on_4p_reset() -> None:
    cfg = _cfg(player_count=4)
    state, _ = reset(jax.random.PRNGKey(42), cfg)
    batch = encode_turn(state.game, cfg)
    assert batch.planet_features.shape == (MAX_PLANETS, planet_feature_dim(cfg))
    assert batch.global_features.shape == (global_feature_dim(cfg),)


def test_encode_v2_sun_crossing_targets_are_masked() -> None:
    cfg = _cfg(candidate_count=4)
    planet_ids = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32).at[0].set(0)
    active = jnp.zeros((MAX_PLANETS,), dtype=bool).at[0].set(True).at[1].set(True)
    x = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    y = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    x = x.at[0].set(20.0)
    y = y.at[0].set(50.0)
    x = x.at[1].set(80.0)
    y = y.at[1].set(50.0)
    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=jnp.full((MAX_PLANETS,), 1.0, dtype=jnp.float32),
        ships=jnp.full((MAX_PLANETS,), 100.0, dtype=jnp.float32),
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=active,
    )
    fleets = JaxFleetState(
        id=jnp.zeros((1,), dtype=jnp.int32),
        owner=jnp.zeros((1,), dtype=jnp.int32),
        x=jnp.zeros((1,), dtype=jnp.float32),
        y=jnp.zeros((1,), dtype=jnp.float32),
        angle=jnp.zeros((1,), dtype=jnp.float32),
        from_planet_id=jnp.zeros((1,), dtype=jnp.int32),
        ships=jnp.zeros((1,), dtype=jnp.float32),
        active=jnp.zeros((1,), dtype=bool),
    )
    from src.jax.env import empty_comet_state

    game = JaxGameState(
        step=jnp.asarray(0, dtype=jnp.int32),
        player=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(0.03, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        episode_seed=jnp.asarray(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=fleets,
        comets=empty_comet_state(),
    )
    batch = encode_turn(game, cfg)
    assert not bool(np.asarray(batch.edge_mask[0]).any())


def test_encode_v2_global_history_expands_dim() -> None:
    cfg = _cfg(feature_history_steps=3)
    state, _ = reset(jax.random.PRNGKey(7), cfg)
    history = empty_feature_history(cfg)
    history = append_feature_history(history, state.game, cfg)
    history = append_feature_history(history, state.game, cfg)
    batch = encode_turn(state.game, cfg, history=history)
    assert batch.global_features.shape == (global_feature_dim(cfg),)


def test_encode_v2_planet_dim_ignores_history_steps() -> None:
    cfg = _cfg(feature_history_steps=5)
    state, _ = reset(jax.random.PRNGKey(9), cfg)
    batch = encode_turn(state.game, cfg)
    assert planet_feature_dim(cfg) == 13
    assert batch.planet_features.shape == (MAX_PLANETS, planet_feature_dim(cfg))
    assert batch.global_features.shape == (global_feature_dim(cfg),)


@pytest.mark.jax
def test_encode_v2_jit_vmap_smoke() -> None:
    cfg = _cfg()
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    states, _ = batched_reset(keys, cfg)

    def encode_game(game):
        return encode_turn(game, cfg)

    vmapped = jax.jit(jax.vmap(encode_game))
    batch = vmapped(states.game)
    assert batch.planet_features.shape == (4, MAX_PLANETS, planet_feature_dim(cfg))
    assert batch.global_features.shape == (4, global_feature_dim(cfg))
