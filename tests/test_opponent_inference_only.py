from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.action_sampling import (
    _sample_opponent_policy_action_with_params,
    _sample_shielded_factored_sequence_with_params,
)
from src.jax.env import batched_reset
from src.jax.policy import build_planet_graph_transformer_policy


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(**kwargs) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 32
    cfg.model.max_moves_k = 2
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.task = _task_cfg(**kwargs.pop("task", {}))
    for key, value in kwargs.pop("model", {}).items():
        setattr(cfg.model, key, value)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.jax
def test_inference_only_skips_critic_and_replay_logprob() -> None:
    cfg = _train_cfg(task={"trajectory_shield_mode": "cheap"})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(11), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(12), batch)
    key = jax.random.PRNGKey(13)

    learner = _sample_shielded_factored_sequence_with_params(
        key,
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
        inference_only=False,
    )
    opponent = _sample_shielded_factored_sequence_with_params(
        key,
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
        inference_only=True,
    )

    assert float(np.asarray(opponent.value).sum()) == 0.0
    assert float(np.asarray(learner.log_prob).sum()) != 0.0
    assert float(np.asarray(opponent.log_prob).sum()) == 0.0
    assert np.isfinite(np.asarray(opponent.target_index)).all()


@pytest.mark.jax
def test_opponent_policy_wrapper_uses_inference_only_path() -> None:
    cfg = _train_cfg(task={"trajectory_shield_mode": "cheap"})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(21), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(22), batch)

    action, _hidden = _sample_opponent_policy_action_with_params(
        jax.random.PRNGKey(23),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
    )
    assert jnp.isfinite(action.ships).all()
    assert jnp.isfinite(action.angle).all()
