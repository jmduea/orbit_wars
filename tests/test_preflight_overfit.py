"""Gate 1 optimization sanity: overfit a single rollout batch."""

from __future__ import annotations

import pytest

import jax
from src.config import TrainConfig
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import ppo_update_jax
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state


def _overfit_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.hidden_size = 16
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.model.max_moves_k = 2
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 4
    cfg.training.update_chunk_rows = 4
    cfg.opponents.mode.opponent = "random"
    return cfg


@pytest.mark.slow
@pytest.mark.jax
def test_overfit_single_rollout_batch() -> None:
    cfg = _overfit_cfg()
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(0), policy, cfg)
    reset_keys = jax.random.split(jax.random.PRNGKey(1), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    _, env_state, turn_batch, transitions, _ = collect_rollout_jax(
        jax.random.PRNGKey(2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
    )
    first_loss = None
    last_loss = None
    state = train_state
    for _ in range(40):
        state, metrics = ppo_update_jax(state, policy, transitions, cfg)
        loss = float(metrics["total_loss"])
        if first_loss is None:
            first_loss = loss
        last_loss = loss
    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss
    assert all(bool(jax.numpy.isfinite(value)) for value in metrics.values())
