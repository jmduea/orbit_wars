"""Neutral policy action contracts and joint-flat index helpers.

Shared by policy decoders, trajectory shield, rollout, and PPO so ``game/`` and
shield code do not import ``src/jax/policy.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax


class JaxPolicyOutput(NamedTuple):
    """Unified joint-flat policy output structure.

    Fields
    ------
    target_logits: jax.Array
        Shape: (batch, sequence_k, candidates)
    ship_logits: jax.Array
        Shape: (batch, sequence_k, candidates, ship_buckets)
    value: jax.Array
        Shape: (batch,)
    decoded_target_sequence: jax.Array
        Shape: (batch, sequence_k). Target path used by autoregressive decoders,
        or -1 for decoders whose steps can be sampled from logits directly.
    """

    target_logits: jax.Array
    ship_logits: jax.Array
    value: jax.Array
    decoded_target_sequence: jax.Array


class FactoredPolicyOutput(NamedTuple):
    """Factorized top-K pointer policy output."""

    source_logits: jax.Array
    target_logits: jax.Array
    stop_logits: jax.Array
    ship_logits: jax.Array
    value: jax.Array
    decoded_source_sequence: jax.Array
    decoded_target_slot_sequence: jax.Array
    decoded_stop_sequence: jax.Array


def ensure_policy_sequence(value: jax.Array) -> jax.Array:
    """Represent policy logits with an explicit sequence axis."""

    if value.ndim == 2:
        return value[:, None, :]
    if value.ndim == 3:
        return value
    return value


def ensure_action_sequence(value: jax.Array) -> jax.Array:
    """Represent sampled action ids with an explicit sequence axis."""

    if value.ndim == 1:
        return value[:, None]
    return value


def flat_edge_index(src_row: jax.Array, slot: jax.Array, k: int) -> jax.Array:
    """Encode ``(source row, target slot)`` into a flat edge index."""

    return src_row * k + slot


def decode_flat_edge(flat_idx: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Decode a flat edge index into ``(source row, target slot)``."""

    return flat_idx // k, flat_idx % k


def noop_edge_index(k: int, *, max_planets: int) -> int:
    """Flat index of the always-legal NO_OP slot."""

    return max_planets * k


def factored_action_log_prob_and_entropy(
    output: FactoredPolicyOutput,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute factorized log-probability and entropy for one launch sequence."""

    squeeze_sequence = source_index.ndim == 1
    source_logits = ensure_policy_sequence(output.source_logits)
    target_logits = ensure_policy_sequence(output.target_logits)
    stop_logits = output.stop_logits
    if stop_logits.ndim == 1:
        stop_logits = stop_logits[:, None]
    ship_logits = output.ship_logits
    if ship_logits.ndim == 3:
        ship_logits = ship_logits[:, None, :, :]
    source_index = ensure_action_sequence(source_index)
    target_slot = ensure_action_sequence(target_slot)
    ship_bucket = ensure_action_sequence(ship_bucket)
    stop_flag = ensure_action_sequence(stop_flag.astype(jnp.float32))
    step_mask = ensure_action_sequence(step_mask.astype(jnp.float32))

    stop_log_probs = jax.nn.log_sigmoid(stop_logits)
    stop_log_probs_neg = jax.nn.log_sigmoid(-stop_logits)
    stop_lp = stop_flag * stop_log_probs + (1.0 - stop_flag) * stop_log_probs_neg
    stop_probs = jax.nn.sigmoid(stop_logits)
    stop_entropy = -(
        stop_probs * stop_log_probs + (1.0 - stop_probs) * stop_log_probs_neg
    )

    source_log_probs = jax.nn.log_softmax(source_logits, axis=-1)
    source_probs = jax.nn.softmax(source_logits, axis=-1)
    source_lp = jnp.take_along_axis(
        source_log_probs, source_index[..., None], axis=-1
    ).squeeze(-1)
    source_entropy = -(source_probs * source_log_probs).sum(axis=-1)

    target_log_probs = jax.nn.log_softmax(target_logits, axis=-1)
    target_probs = jax.nn.softmax(target_logits, axis=-1)
    target_lp = jnp.take_along_axis(
        target_log_probs, target_slot[..., None], axis=-1
    ).squeeze(-1)
    target_entropy = -(target_probs * target_log_probs).sum(axis=-1)

    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target_slot[..., None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=2,
    ).squeeze(axis=2)
    ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
    ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
    ship_lp = jnp.take_along_axis(
        ship_log_probs, ship_bucket[..., None], axis=-1
    ).squeeze(-1)
    ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)

    head_active = step_mask * (1.0 - stop_flag)
    log_prob = stop_lp + head_active * (source_lp + target_lp + ship_lp)
    entropy = stop_entropy + head_active * (
        source_entropy + target_entropy + ship_entropy
    )
    if squeeze_sequence:
        return log_prob[:, 0], entropy[:, 0]
    return log_prob, entropy


def _factored_step_log_prob_entropy(
    source_logits: jax.Array,
    target_logits: jax.Array,
    stop_logit: jax.Array,
    ship_logits: jax.Array,
    source_mask: jax.Array,
    ship_bucket_mask: jax.Array,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Match rollout ``_sample_factored_step_from_logits`` log-prob math (batched)."""

    illegal_logit = jnp.finfo(jnp.float32).min
    stop = stop_flag.astype(jnp.float32)
    stop_log_probs = jax.nn.log_sigmoid(stop_logit)
    stop_log_probs_neg = jax.nn.log_sigmoid(-stop_logit)
    stop_lp = stop * stop_log_probs + (1.0 - stop) * stop_log_probs_neg
    stop_prob = jax.nn.sigmoid(stop_logit)
    stop_entropy = -(
        stop_prob * stop_log_probs + (1.0 - stop_prob) * stop_log_probs_neg
    )

    masked_source_logits = jnp.where(source_mask, source_logits, illegal_logit)
    source_log_probs = jax.nn.log_softmax(masked_source_logits, axis=-1)
    source_probs = jax.nn.softmax(masked_source_logits, axis=-1)
    source_lp = jnp.take_along_axis(
        source_log_probs, source_index[..., None], axis=-1
    ).squeeze(-1)
    source_entropy = -(source_probs * source_log_probs).sum(axis=-1)

    batch_size = source_logits.shape[0]
    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)
    row_bucket_mask = ship_bucket_mask[batch_idx, source_index]
    target_mask = row_bucket_mask.any(axis=-1)
    masked_target_logits = jnp.where(target_mask, target_logits, illegal_logit)
    target_log_probs = jax.nn.log_softmax(masked_target_logits, axis=-1)
    target_probs = jax.nn.softmax(masked_target_logits, axis=-1)
    target_lp = jnp.take_along_axis(
        target_log_probs, target_slot[..., None], axis=-1
    ).squeeze(-1)
    target_entropy = -(target_probs * target_log_probs).sum(axis=-1)

    selected_bucket_mask = row_bucket_mask[batch_idx, target_slot]
    selected_ship_logits = ship_logits[batch_idx, target_slot]
    selected_ship_logits = jnp.where(
        selected_bucket_mask, selected_ship_logits, illegal_logit
    )
    ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
    ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
    ship_lp = jnp.take_along_axis(
        ship_log_probs, ship_bucket[..., None], axis=-1
    ).squeeze(-1)
    ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)

    head_active = 1.0 - stop
    move_entropy = head_active * (source_entropy + target_entropy + ship_entropy)
    log_prob = stop_lp + head_active * (source_lp + target_lp + ship_lp)
    entropy = stop_entropy + move_entropy
    return log_prob, entropy, stop_entropy, move_entropy


def factored_action_log_prob_with_shield(
    output: FactoredPolicyOutput,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    source_mask: jax.Array,
    ship_bucket_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Factorized log-prob replay using rollout-equivalent per-step shield masks."""

    squeeze_sequence = source_index.ndim == 1
    source_logits = ensure_policy_sequence(output.source_logits)
    target_logits = ensure_policy_sequence(output.target_logits)
    stop_logits = output.stop_logits
    if stop_logits.ndim == 1:
        stop_logits = stop_logits[:, None]
    ship_logits = output.ship_logits
    if ship_logits.ndim == 3:
        ship_logits = ship_logits[:, None, :, :]
    source_index = ensure_action_sequence(source_index.astype(jnp.int32))
    target_slot = ensure_action_sequence(target_slot.astype(jnp.int32))
    ship_bucket = ensure_action_sequence(ship_bucket.astype(jnp.int32))
    stop_flag = ensure_action_sequence(stop_flag.astype(jnp.float32))
    step_mask = ensure_action_sequence(step_mask.astype(jnp.float32))

    if source_mask.ndim == 2:
        source_mask = source_mask[:, None, :]
    if ship_bucket_mask.ndim == 4:
        ship_bucket_mask = ship_bucket_mask[:, None, ...]

    def one_step(
        src_logits,
        tgt_logits,
        st_logit,
        sh_logits,
        src_mask,
        bucket_mask,
        src,
        tgt,
        bkt,
        stop,
        mask,
    ):
        log_prob, entropy, stop_entropy, move_entropy = _factored_step_log_prob_entropy(
            src_logits,
            tgt_logits,
            st_logit,
            sh_logits,
            src_mask,
            bucket_mask,
            src,
            tgt,
            bkt,
            stop,
        )
        return (
            mask * log_prob,
            mask * entropy,
            mask * stop_entropy,
            mask * move_entropy,
        )

    log_prob, entropy, stop_entropy, move_entropy = jax.vmap(
        one_step, in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), out_axes=1
    )(
        source_logits,
        target_logits,
        stop_logits,
        ship_logits,
        source_mask,
        ship_bucket_mask,
        source_index,
        target_slot,
        ship_bucket,
        stop_flag,
        step_mask,
    )
    if squeeze_sequence:
        return (
            log_prob[:, 0],
            entropy[:, 0],
            stop_entropy[:, 0],
            move_entropy[:, 0],
        )
    return log_prob, entropy, stop_entropy, move_entropy


def action_log_prob_and_entropy(
    output: JaxPolicyOutput,
    target_index: jax.Array,
    ship_bucket: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute joint log-probability and entropy for target/bucket actions."""

    squeeze_sequence = target_index.ndim == 1
    target_logits = ensure_policy_sequence(output.target_logits)
    ship_logits = ensure_policy_sequence(output.ship_logits)
    target_index = ensure_action_sequence(target_index)
    ship_bucket = ensure_action_sequence(ship_bucket)
    target_log_probs = jax.nn.log_softmax(target_logits, axis=-1)
    target_probs = jax.nn.softmax(target_logits, axis=-1)
    target_lp = jnp.take_along_axis(
        target_log_probs, target_index[..., None], axis=-1
    ).squeeze(-1)
    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target_index[..., None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=2,
    ).squeeze(axis=2)
    ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
    ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
    ship_lp = jnp.take_along_axis(
        ship_log_probs, ship_bucket[..., None], axis=-1
    ).squeeze(-1)
    target_entropy = -(target_probs * target_log_probs).sum(axis=-1)
    ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)
    log_prob = target_lp + ship_lp
    entropy = target_entropy + ship_entropy
    if squeeze_sequence:
        return log_prob[:, 0], entropy[:, 0]
    return log_prob, entropy
