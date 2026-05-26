import jax.numpy as jnp
import numpy as np

from src.jax.action_codec import (
    FactoredPolicyOutput,
    JaxPolicyOutput,
    action_log_prob_and_entropy,
    decode_flat_edge,
    factored_action_log_prob_and_entropy,
    factored_action_log_prob_with_shield,
    flat_edge_index,
    noop_edge_index,
)


def test_flat_edge_index_roundtrip() -> None:
    k = 7
    for src_row in (0, 5, 59):
        for slot in (0, 3, k - 1):
            flat = int(flat_edge_index(jnp.int32(src_row), jnp.int32(slot), k))
            decoded_src, decoded_slot = decode_flat_edge(jnp.int32(flat), k)
            np.testing.assert_array_equal(np.asarray(decoded_src), src_row)
            np.testing.assert_array_equal(np.asarray(decoded_slot), slot)


def test_noop_edge_index() -> None:
    assert noop_edge_index(7, max_planets=60) == 420


def test_joint_continuous_ship_log_prob_uses_width_one_head() -> None:
    output = JaxPolicyOutput(
        target_logits=jnp.array([[0.0, 1.0]], dtype=jnp.float32),
        ship_logits=jnp.array([[[[0.25], [-0.5]]]], dtype=jnp.float32),
        value=jnp.zeros((1,), dtype=jnp.float32),
        decoded_target_sequence=jnp.full((1, 1), -1, dtype=jnp.int32),
    )

    log_prob, entropy = action_log_prob_and_entropy(
        output,
        target_index=jnp.array([1], dtype=jnp.int32),
        ship_bucket=jnp.array([1], dtype=jnp.int32),
    )

    expected_target_lp = jax_log_softmax(np.array([0.0, 1.0], dtype=np.float32))[1]
    expected_ship_lp = -np.logaddexp(0.0, 0.5) - np.logaddexp(0.0, -0.5)
    np.testing.assert_allclose(
        np.asarray(log_prob), expected_target_lp + expected_ship_lp, rtol=1e-6
    )
    assert np.isfinite(np.asarray(entropy)).all()


def test_factored_continuous_ship_log_prob_uses_width_one_head() -> None:
    output = FactoredPolicyOutput(
        source_logits=jnp.array([[0.0, 0.5]], dtype=jnp.float32),
        target_logits=jnp.array([[0.25, 0.0]], dtype=jnp.float32),
        stop_logits=jnp.array([[-2.0]], dtype=jnp.float32),
        ship_logits=jnp.array([[[0.75], [-0.25]]], dtype=jnp.float32),
        value=jnp.zeros((1,), dtype=jnp.float32),
        decoded_source_sequence=jnp.full((1, 1), -1, dtype=jnp.int32),
        decoded_target_slot_sequence=jnp.full((1, 1), -1, dtype=jnp.int32),
        decoded_stop_sequence=jnp.zeros((1, 1), dtype=jnp.int32),
    )

    log_prob, entropy = factored_action_log_prob_and_entropy(
        output,
        source_index=jnp.array([1], dtype=jnp.int32),
        target_slot=jnp.array([0], dtype=jnp.int32),
        ship_bucket=jnp.array([1], dtype=jnp.int32),
        stop_flag=jnp.array([0], dtype=jnp.int32),
        step_mask=jnp.array([1.0], dtype=jnp.float32),
    )

    assert np.isfinite(np.asarray(log_prob)).all()
    assert np.isfinite(np.asarray(entropy)).all()


def test_factored_shield_continuous_ship_log_prob_batched_targets() -> None:
    """Regression: continuous ship replay must index per-target legality, not (B, T)."""

    batch_size = 8
    num_targets = 7
    output = FactoredPolicyOutput(
        source_logits=jnp.zeros((batch_size, 1, 2), dtype=jnp.float32),
        target_logits=jnp.zeros((batch_size, 1, num_targets), dtype=jnp.float32),
        stop_logits=jnp.full((batch_size, 1), -2.0, dtype=jnp.float32),
        ship_logits=jnp.zeros((batch_size, 1, num_targets, 1), dtype=jnp.float32),
        value=jnp.zeros((batch_size,), dtype=jnp.float32),
        decoded_source_sequence=jnp.full((batch_size, 1), -1, dtype=jnp.int32),
        decoded_target_slot_sequence=jnp.full((batch_size, 1), -1, dtype=jnp.int32),
        decoded_stop_sequence=jnp.zeros((batch_size, 1), dtype=jnp.int32),
    )
    source_index = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    target_slot = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    ship_bucket = jnp.ones((batch_size, 1), dtype=jnp.int32)
    stop_flag = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    step_mask = jnp.ones((batch_size, 1), dtype=jnp.float32)
    source_mask = jnp.ones((batch_size, 1, 2), dtype=bool)
    ship_bucket_mask = jnp.zeros((batch_size, 1, 2, num_targets, 8), dtype=bool)
    ship_bucket_mask = ship_bucket_mask.at[:, :, 0, 0, 1].set(True)

    log_prob, entropy, stop_entropy, move_entropy = (
        factored_action_log_prob_with_shield(
            output,
            source_index,
            target_slot,
            ship_bucket,
            stop_flag,
            step_mask,
            source_mask,
            ship_bucket_mask,
        )
    )

    for arr in (log_prob, entropy, stop_entropy, move_entropy):
        assert np.isfinite(np.asarray(arr)).all()


def test_factored_shield_replay_teacher_forces_stored_actions() -> None:
    """Stored rollout actions stay legal during replay even when coarse masks omit them."""

    batch_size = 4
    num_targets = 7
    output = FactoredPolicyOutput(
        source_logits=jnp.zeros((batch_size, 1, 2), dtype=jnp.float32),
        target_logits=jnp.zeros((batch_size, 1, num_targets), dtype=jnp.float32),
        stop_logits=jnp.full((batch_size, 1), -1.0, dtype=jnp.float32),
        ship_logits=jnp.zeros((batch_size, 1, num_targets, 1), dtype=jnp.float32),
        value=jnp.zeros((batch_size,), dtype=jnp.float32),
        decoded_source_sequence=jnp.full((batch_size, 1), -1, dtype=jnp.int32),
        decoded_target_slot_sequence=jnp.full((batch_size, 1), -1, dtype=jnp.int32),
        decoded_stop_sequence=jnp.zeros((batch_size, 1), dtype=jnp.int32),
    )
    source_index = jnp.ones((batch_size, 1), dtype=jnp.int32)
    target_slot = jnp.ones((batch_size, 1), dtype=jnp.int32)
    ship_bucket = jnp.ones((batch_size, 1), dtype=jnp.int32)
    stop_flag = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    step_mask = jnp.ones((batch_size, 1), dtype=jnp.float32)
    source_mask = jnp.zeros((batch_size, 1, 2), dtype=bool)
    ship_bucket_mask = jnp.zeros((batch_size, 1, 2, num_targets, 8), dtype=bool)
    ship_bucket_mask = ship_bucket_mask.at[:, :, 0, 0, 1].set(True)

    log_prob, entropy, stop_entropy, move_entropy = (
        factored_action_log_prob_with_shield(
            output,
            source_index,
            target_slot,
            ship_bucket,
            stop_flag,
            step_mask,
            source_mask,
            ship_bucket_mask,
        )
    )

    for arr in (log_prob, entropy, stop_entropy, move_entropy):
        assert np.isfinite(np.asarray(arr)).all()
    assert float(np.asarray(log_prob).mean()) > -1e10


def jax_log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())
