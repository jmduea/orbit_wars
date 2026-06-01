from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.action_sampling import _sample_shielded_sequence_with_params
from src.jax.decoder_carry import empty_decoder_hidden, reset_decoder_hidden_on_done
from src.jax.env import batched_reset
from src.jax.policy import (
    build_planet_graph_transformer_policy,
    make_synthetic_turn_batch,
)
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=8, max_fleets=16)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, decoder_carry: bool) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 64
    cfg.model.max_moves_k = 2
    cfg.model.decoder_carry = decoder_carry
    cfg.task = _task_cfg()
    return cfg


def _sampler_cfg(*, pointer_decoder: str) -> TrainConfig:
    cfg = _train_cfg(decoder_carry=True)
    cfg.model.pointer_decoder = pointer_decoder
    cfg.model.hidden_size = 32
    cfg.model.max_moves_k = 2
    cfg.task = _task_cfg(ship_bucket_count=4, max_fleets=8)
    cfg.task.trajectory_shield_mode = "off"
    return cfg


def test_reset_decoder_hidden_on_done() -> None:
    hidden = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    fresh = jnp.zeros_like(hidden)
    done = jnp.array([True, False], dtype=bool)
    reset = reset_decoder_hidden_on_done(hidden, done, fresh)
    np.testing.assert_allclose(np.asarray(reset[0]), np.zeros(2), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(reset[1]), np.asarray(hidden[1]), rtol=0, atol=0
    )


def test_policy_returns_decoder_hidden_when_carry_enabled() -> None:
    cfg = _train_cfg(decoder_carry=True)
    policy = build_planet_graph_transformer_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(0))
    params = policy.init(jax.random.PRNGKey(1), batch)
    carry_in = empty_decoder_hidden(2, cfg.model.hidden_size)
    output = policy.apply(
        params,
        batch,
        deterministic=True,
        decoder_hidden=carry_in,
    )
    assert output.decoder_hidden is not None
    assert output.decoder_hidden.shape == (2, cfg.model.hidden_size)


def test_policy_decoder_hidden_none_when_carry_disabled() -> None:
    cfg = _train_cfg(decoder_carry=False)
    policy = build_planet_graph_transformer_policy(cfg)
    batch = make_synthetic_turn_batch(1, cfg.task, key=jax.random.PRNGKey(2))
    params = policy.init(jax.random.PRNGKey(3), batch)
    output = policy.apply(params, batch, deterministic=True)
    assert output.decoder_hidden is None


def test_factorized_sampler_carry_matches_replay_from_incoming_hidden() -> None:
    cfg = _sampler_cfg(pointer_decoder="factorized_topk")
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(4), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(5), batch)
    carry_in = jnp.full((1, cfg.model.hidden_size), 0.25, dtype=jnp.float32)

    sample = _sample_shielded_sequence_with_params(
        jax.random.PRNGKey(6),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        decoder_hidden_in=carry_in,
    )
    replay = policy.apply(
        params,
        batch,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_sequence=sample.source_index,
        target_slot_sequence=sample.target_slot,
        decoder_hidden=carry_in,
        deterministic=True,
    )

    np.testing.assert_allclose(
        np.asarray(sample.decoder_hidden_out),
        np.asarray(replay.decoder_hidden),
        rtol=1e-6,
        atol=1e-6,
    )



def test_rollout_initializes_env_state_decoder_hidden_for_scan_structure() -> None:
    cfg = _sampler_cfg(pointer_decoder="factorized_topk")
    cfg.opponents.mode.opponent = "random"
    cfg.training.num_envs = 1
    cfg.training.rollout_steps = 1
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(10), 1), cfg.task)
    assert state.decoder_hidden is None
    policy = build_planet_graph_transformer_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)

    _key, next_state, _batch, transitions, _metrics = collect_rollout_jax(
        jax.random.PRNGKey(12),
        state,
        batch,
        train_state,
        policy,
        cfg,
    )

    assert next_state.decoder_hidden is not None
    assert next_state.decoder_hidden.shape == (1, cfg.model.hidden_size)
    assert transitions.decoder_hidden is not None
    assert transitions.decoder_hidden.shape == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        cfg.model.hidden_size,
    )
