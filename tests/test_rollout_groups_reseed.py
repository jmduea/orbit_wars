"""Rollout group env reset on seed scheduler reseed."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig
from src.jax.policy import build_jax_policy
from src.jax.train.rollout_groups import init_rollout_groups, reset_rollout_groups_envs


def _tiny_rollout_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.task.player_count = 2
    cfg.task.max_fleets = 8
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.format_weights = {2: 1.0}
    cfg.training.rollout_microbatch_envs = 2
    return cfg


@pytest.mark.jax
def test_reset_rollout_groups_envs_changes_planet_layout() -> None:
    cfg = _tiny_rollout_cfg()
    policy = build_jax_policy(cfg=cfg)
    key = jax.random.PRNGKey(0)
    _, rollout_groups = init_rollout_groups(key, cfg, policy)
    before = jax.device_get(rollout_groups[0].env_state.game.planets.x)

    reset_key = jax.random.PRNGKey(99)
    _, reset_groups = reset_rollout_groups_envs(reset_key, rollout_groups)
    after = jax.device_get(reset_groups[0].env_state.game.planets.x)

    assert before.shape == after.shape
    assert not jnp.allclose(before, after)
