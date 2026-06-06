"""Characterization tests for pool-gather reset and round-robin map selection."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import RewardConfig, TaskConfig
from src.jax.env import (
    assign_learner_players,
    batched_reset_with_pool,
    empty_action,
    reset_with_pool,
    step,
)
from src.jax.map_pool.bake import bake_one_entry, stack_entries
from src.jax.map_pool.load import load_map_pool, map_pool_constants_from_numpy


@pytest.fixture(scope="module")
def tiny_pool():
    entries = [bake_one_entry(seed) for seed in (0, 1, 2)]
    stacked = stack_entries(entries)
    return map_pool_constants_from_numpy(stacked)


def test_reset_with_pool_produces_valid_group_count(tiny_pool):
    cfg = TaskConfig(player_count=2, map_pool_path="unused")
    key = jax.random.PRNGKey(0)
    state, batch = reset_with_pool(key, cfg, tiny_pool, jnp.array(0, dtype=jnp.int32))
    active = int(state.game.planets.active.sum())
    group_count = active // 4
    assert 5 <= group_count <= 10
    assert batch.planet_mask.ndim >= 1


def test_consecutive_resets_rotate_map_ids(tiny_pool):
    cfg = TaskConfig(player_count=2)
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    map_ids = jnp.array([0, 1], dtype=jnp.int32)
    states, _ = batched_reset_with_pool(keys, cfg, tiny_pool, map_ids)
    av0 = float(states.game.angular_velocity[0])
    av1 = float(states.game.angular_velocity[1])
    assert av0 != av1


def test_episode_map_id_wraps_mod_pool_size(tiny_pool):
    cfg = TaskConfig(player_count=2)
    env_indices = jnp.array([0, 0], dtype=jnp.int32)
    episode_counts = jnp.array([3, 6], dtype=jnp.int32)
    map_ids = (episode_counts + env_indices) % tiny_pool.pool_size
    assert int(map_ids[0]) == int(map_ids[1]) == 0
    keys = jax.random.split(jax.random.PRNGKey(2), 2)
    states, _ = batched_reset_with_pool(keys, cfg, tiny_pool, map_ids)
    av = [float(x) for x in states.game.angular_velocity]
    assert av[0] == av[1]


def test_comet_wave_activates_at_spawn_step():
    pool = load_map_pool("data/jax_map_pool/default_v1.npz")
    cfg = TaskConfig(player_count=2)
    reward_cfg = RewardConfig()
    key = jax.random.PRNGKey(99)
    state, batch = reset_with_pool(key, cfg, pool, jnp.array(0, dtype=jnp.int32))
    del batch
    noop = empty_action(cfg)
    for target_step in (49, 50):
        while int(state.game.step) < target_step:
            state, _ = step(state, noop, noop, cfg, reward_cfg)
        active_groups = int(state.game.comets.group_active.sum())
        if target_step == 50:
            assert active_groups >= 1


def test_default_pool_loader_shapes():
    pool = load_map_pool("data/jax_map_pool/default_v1.npz")
    assert pool.pool_size == 500
    assert pool.comet_paths_x.shape == (500, 5, 4, 40)


def test_initial_rollout_style_assign_learner_players(tiny_pool):
    cfg = TaskConfig(player_count=2)
    keys = jax.random.split(jax.random.PRNGKey(5), 2)
    env_indices = jnp.arange(2, dtype=jnp.int32)
    episode_counts = jnp.zeros((2,), dtype=jnp.int32)
    map_ids = (episode_counts + env_indices) % tiny_pool.pool_size
    state, batch = batched_reset_with_pool(keys, cfg, tiny_pool, map_ids)
    state, batch = assign_learner_players(
        state, env_indices, episode_counts, cfg, alternate_player_sides=True
    )
    assert batch.planet_features.shape[0] == 2
    assert int(state.learner_player[0]) in (0, 1)
