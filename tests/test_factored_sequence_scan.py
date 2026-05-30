from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.env import batched_reset
from src.jax.factored_sequence_scan import replay_factored_sequence_logprob
from src.jax.policy import build_planet_graph_transformer_policy
from src.opponents.jax_actions.builders import _sample_shielded_factored_sequence_with_params


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
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.jax
def test_rollout_replay_logprob_parity_with_stepwise_scan() -> None:
    cfg = _train_cfg(task={"trajectory_shield_enabled": False})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(1), batch)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(2),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
    )
    delta = replay.log_prob - sample.log_prob
    assert jnp.all(jnp.isfinite(delta))
    assert float(jnp.mean(jnp.abs(delta))) < 1e-4




@pytest.mark.jax
def test_continuous_ship_logprob_depends_on_policy_loc() -> None:
    from src.jax.ship_action import continuous_fraction_log_prob_at_action

    policy_loc = jnp.array([0.0, 1.5], dtype=jnp.float32)
    fraction = jnp.array([0.5, 0.7], dtype=jnp.float32)

    def log_prob_sum(loc: jax.Array) -> jax.Array:
        return continuous_fraction_log_prob_at_action(loc, fraction).sum()

    grad = jax.grad(log_prob_sum)(policy_loc)
    assert jnp.any(jnp.abs(grad) > 1e-6)


@pytest.mark.jax
def test_rollout_replay_logprob_parity_continuous_fraction() -> None:
    cfg = _train_cfg(task={"ship_action_mode": "continuous_fraction"})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(11), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(12), batch)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(13),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=False,
        deterministic_eval=False,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
    )
    active = sample.step_mask > 0.0
    delta = (replay.log_prob - sample.log_prob) * active
    assert jnp.all(jnp.isfinite(delta))
    assert float(jnp.sum(jnp.abs(delta)) / jnp.maximum(active.sum(), 1.0)) < 0.05

@pytest.mark.jax
def test_rollout_replay_logprob_parity_with_decoder_carry() -> None:
    cfg = _train_cfg(task={"trajectory_shield_enabled": False})
    cfg.model.decoder_carry = True
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(3), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(4), batch)
    carry_in = jnp.full((1, cfg.model.hidden_size), 0.5, dtype=jnp.float32)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(5),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
        decoder_hidden_in=carry_in,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
        decoder_hidden=carry_in,
    )
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )
