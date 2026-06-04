"""Rollout collect with JAX training opponent=noop / no_op."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state
from src.opponents.jax_actions.builders import build_noop_action_from_edge_batch


def _noop_rollout_cfg(*, opponent: str) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.candidate_count = 3
    cfg.task.edge_rank_mode = "intercept_min"
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 4
    cfg.training.update_chunk_rows = 16
    cfg.opponents.mode.opponent = opponent
    return cfg


@pytest.mark.jax
@pytest.mark.parametrize("opponent", ["noop", "no_op"])
def test_collect_rollout_noop_opponent_finite(opponent: str) -> None:
    cfg = _noop_rollout_cfg(opponent=opponent)
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    key = jax.random.PRNGKey(2)

    key, env_state, turn_batch, transitions, metrics = collect_rollout_jax(
        key, env_state, turn_batch, train_state, policy, cfg
    )

    assert (
        float(metrics["env_steps"])
        == cfg.training.rollout_steps * cfg.training.num_envs
    )
    assert jnp.all(jnp.isfinite(transitions.returns))
    assert jnp.all(jnp.isfinite(transitions.advantages))

    noop_action = build_noop_action_from_edge_batch(env_state.game, turn_batch, cfg)
    assert not jnp.any(noop_action.valid)
    assert jnp.all(noop_action.ships == 0)
