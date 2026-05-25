"""Unit tests for C51-style distributional value support."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig, compose_hydra_train_config
from src.jax.distributional_value import (
    categorical_value_cross_entropy,
    expected_value_from_logits,
    project_returns_to_two_hot,
    value_support,
)
from src.jax.policy import (
    CategoricalValueHead,
    build_value_head,
    is_distributional_value_head,
)


def test_value_support_spans_symmetric_range() -> None:
    support = value_support(51, 1.0)
    assert support.shape == (51,)
    assert float(support[0]) == pytest.approx(-1.0)
    assert float(support[-1]) == pytest.approx(1.0)


def test_project_returns_to_two_hot_is_probability_mass() -> None:
    support = value_support(5, 1.0)
    targets = project_returns_to_two_hot(jnp.array([0.25, -0.5, 1.0]), support)
    assert targets.shape == (3, 5)
    assert jnp.all(targets >= 0.0)
    assert jnp.allclose(targets.sum(axis=-1), 1.0)


def test_expected_value_from_logits_matches_support_mean_at_uniform() -> None:
    support = value_support(5, 2.0)
    logits = jnp.zeros((2, 5), dtype=jnp.float32)
    expected = expected_value_from_logits(logits, support)
    assert expected.shape == (2,)
    assert float(expected[0]) == pytest.approx(0.0, abs=1e-6)


def test_categorical_value_cross_entropy_is_non_negative() -> None:
    support = value_support(11, 1.0)
    logits = jnp.zeros((4, 11), dtype=jnp.float32)
    returns = jnp.array([-0.2, 0.0, 0.5, 0.9], dtype=jnp.float32)
    loss = categorical_value_cross_entropy(logits, returns, support)
    assert loss.shape == (4,)
    assert jnp.all(loss >= 0.0)


def test_build_value_head_distributional_and_forward_shape() -> None:
    cfg = TrainConfig()
    cfg.model.value_head = "distributional"
    cfg.model.value_bins = 11
    cfg.model.value_max = 2.0
    cfg.model.hidden_size = 16

    assert is_distributional_value_head(cfg)
    module = build_value_head(cfg)
    assert isinstance(module, CategoricalValueHead)

    params = module.init(jax.random.PRNGKey(0), jnp.zeros((3, 16), dtype=jnp.float32))
    out = module.apply(params, jnp.zeros((3, 16), dtype=jnp.float32))
    assert out.value.shape == (3,)
    assert out.value_logits is not None
    assert out.value_logits.shape == (3, 11)


def test_compose_hydra_train_config_accepts_distributional_value_head() -> None:
    cfg = compose_hydra_train_config(["model=distributional_value"])
    assert cfg.model.value_head == "distributional"
    assert cfg.model.value_bins == 51
    assert cfg.model.value_max == 1.0


def test_value_loss_per_step_uses_cross_entropy_for_distributional_head() -> None:
    from src.jax.ppo_update import _value_loss_per_step

    cfg = TrainConfig()
    cfg.model.value_head = "distributional"
    cfg.model.value_bins = 11
    cfg.model.value_max = 1.0
    returns = jnp.array([[0.2, -0.1]], dtype=jnp.float32)
    logits = jnp.zeros((1, 11), dtype=jnp.float32)
    value = jnp.array([0.0], dtype=jnp.float32)
    loss = _value_loss_per_step(cfg, value, logits, returns)
    assert loss.shape == (1, 2)
    assert jnp.all(jnp.isfinite(loss))
    assert float(loss.mean()) >= 0.0
