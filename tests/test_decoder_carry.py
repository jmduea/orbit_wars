from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.decoder_carry import empty_decoder_hidden, reset_decoder_hidden_on_done
from src.jax.policy import build_gnn_pointer_policy, make_synthetic_turn_batch


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=8, max_fleets=16)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, decoder_carry: bool) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "gnn_pointer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 64
    cfg.model.max_moves_k = 2
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.model.decoder_carry = decoder_carry
    cfg.task = _task_cfg()
    return cfg


def test_reset_decoder_hidden_on_done() -> None:
    hidden = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    fresh = jnp.zeros_like(hidden)
    done = jnp.array([True, False], dtype=bool)
    reset = reset_decoder_hidden_on_done(hidden, done, fresh)
    np.testing.assert_allclose(np.asarray(reset[0]), np.zeros(2), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(reset[1]), np.asarray(hidden[1]), rtol=0, atol=0)


def test_policy_returns_decoder_hidden_when_carry_enabled() -> None:
    cfg = _train_cfg(decoder_carry=True)
    policy = build_gnn_pointer_policy(cfg)
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
    policy = build_gnn_pointer_policy(cfg)
    batch = make_synthetic_turn_batch(1, cfg.task, key=jax.random.PRNGKey(2))
    params = policy.init(jax.random.PRNGKey(3), batch)
    output = policy.apply(params, batch, deterministic=True)
    assert output.decoder_hidden is None
