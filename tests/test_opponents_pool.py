"""Tests for categorical opponent family sampling in src.opponents.pool."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.opponents.constants import OPPONENT_NOOP, OPPONENT_RANDOM
from src.opponents.pool import OPPONENT_FAMILY_IDS, sample_opponent_type_ids_jax


@pytest.mark.jax
def test_sample_opponent_type_ids_respects_categorical_mixture() -> None:
    ids = jnp.asarray([OPPONENT_NOOP, OPPONENT_RANDOM], dtype=jnp.int32)
    probs = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    sampled = sample_opponent_type_ids_jax(
        jax.random.PRNGKey(0),
        env_count=4,
        player_count=2,
        ids=ids,
        probs=probs,
    )
    assert sampled.shape == (4, 2)
    assert jnp.all(sampled == OPPONENT_NOOP)


@pytest.mark.jax
def test_opponent_family_ids_matches_constants_order() -> None:
    assert OPPONENT_FAMILY_IDS.shape == (7,)
    assert int(OPPONENT_FAMILY_IDS[-1]) == OPPONENT_NOOP


@pytest.mark.jax
def test_sample_opponent_type_ids_handles_near_zero_probabilities() -> None:
    ids = jnp.asarray([OPPONENT_RANDOM, OPPONENT_NOOP], dtype=jnp.int32)
    probs = jnp.asarray([1e-15, 1.0], dtype=jnp.float32)
    sampled = sample_opponent_type_ids_jax(
        jax.random.PRNGKey(1),
        env_count=2,
        player_count=1,
        ids=ids,
        probs=probs,
    )
    assert jnp.all(sampled == OPPONENT_NOOP)
