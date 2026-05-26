"""CPU-light unit tests for safe masked categorical helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.jax.action_codec import (
    _factored_step_log_prob_entropy,
    _safe_categorical_entropy,
    _safe_categorical_log_prob,
)


def test_safe_categorical_all_false_mask_stays_finite() -> None:
    logits = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=jnp.float32)
    mask = jnp.zeros((2, 3), dtype=bool)
    active = jnp.ones((2,), dtype=jnp.float32)

    logp = _safe_categorical_log_prob(
        logits, mask, jnp.zeros((2,), dtype=jnp.int32), active=active
    )
    entropy = _safe_categorical_entropy(logits, mask, active=active)

    assert np.isfinite(np.asarray(logp)).all()
    assert np.isfinite(np.asarray(entropy)).all()
    np.testing.assert_allclose(np.asarray(logp), 0.0)
    np.testing.assert_allclose(np.asarray(entropy), 0.0)


def test_factored_step_stop_masks_move_terms_without_nan() -> None:
    batch_size = 4
    source_logits = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    target_logits = jnp.zeros((batch_size, 3), dtype=jnp.float32)
    stop_logit = jnp.array([5.0, -5.0, 0.0, 0.0], dtype=jnp.float32)
    ship_logits = jnp.zeros((batch_size, 3, 4), dtype=jnp.float32)
    source_mask = jnp.array(
        [
            [False, False],
            [True, False],
            [False, False],
            [True, True],
        ],
        dtype=bool,
    )
    ship_bucket_mask = jnp.zeros((batch_size, 2, 3, 4), dtype=bool)
    source_index = jnp.zeros((batch_size,), dtype=jnp.int32)
    target_slot = jnp.zeros((batch_size,), dtype=jnp.int32)
    ship_bucket = jnp.zeros((batch_size,), dtype=jnp.int32)
    stop_flag = jnp.array([1, 0, 1, 0], dtype=jnp.float32)

    log_prob, entropy, stop_entropy, move_entropy = _factored_step_log_prob_entropy(
        source_logits,
        target_logits,
        stop_logit,
        ship_logits,
        source_mask,
        ship_bucket_mask,
        source_index,
        target_slot,
        ship_bucket,
        stop_flag,
    )

    for arr in (log_prob, entropy, stop_entropy, move_entropy):
        assert np.isfinite(np.asarray(arr)).all()
    assert float(np.asarray(move_entropy)[0]) == pytest.approx(0.0)
    assert float(np.asarray(move_entropy)[2]) == pytest.approx(0.0)
