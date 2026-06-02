"""Neutral policy action contracts and joint-flat index helpers.

Shared by policy decoders, trajectory shield, rollout, and PPO so ``game/`` and
shield code do not import ``src/jax/policy.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax


def source_mask_from_bucket_mask_and_ships(
    ship_bucket_mask: jax.Array,
    remaining_ships: jax.Array,
) -> jax.Array:
    """Owned planets with ships and a shielded non-noop bucket on any slot."""

    has_real_bucket = ship_bucket_mask[..., 1:].any(axis=-1)
    row_has_legal = has_real_bucket
    return (remaining_ships > 0.0) & row_has_legal.any(axis=-1)


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
    value_logits: jax.Array | None
        Shape: (batch, value_bins) when ``model.value_head=distributional``; else None.
    """

    target_logits: jax.Array
    ship_logits: jax.Array
    value: jax.Array
    decoded_target_sequence: jax.Array
    value_logits: jax.Array | None = None
    decoder_hidden: jax.Array | None = None


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
    value_logits: jax.Array | None = None
    decoder_hidden: jax.Array | None = None


class PlanetFlowPolicyOutput(NamedTuple):
    """Planet Flow target-demand policy output.

    ``target_demand_logits`` has shape ``(batch, MAX_PLANETS, pressure_buckets)``.
    Each active planet samples one normalized pressure bucket; inactive planets
    are forced to bucket zero by the pressure-action contract.
    """

    target_demand_logits: jax.Array
    value: jax.Array
    value_logits: jax.Array | None = None


class PlanetFlowPressureAction(NamedTuple):
    """Sampled Planet Flow pressure action stored for rollout/PPO replay."""

    target_bucket: jax.Array
    target_pressure: jax.Array
    log_prob: jax.Array
    entropy: jax.Array
    target_mask: jax.Array


def _continuous_fraction_log_prob(logit: jax.Array) -> jax.Array:
    """Log density of a logistic ship-fraction draw at ``logit``."""

    return -jax.nn.softplus(-logit) - jax.nn.softplus(logit)


def _logit_from_fraction(fraction: jax.Array) -> jax.Array:
    """Invert a clipped launch fraction for continuous ship replay."""

    clipped = jnp.clip(fraction.astype(jnp.float32), 1e-6, 1.0 - 1e-6)
    return jnp.log(clipped) - jnp.log1p(-clipped)


_SAFE_NEG_LOGIT = -1.0e9


def _safe_masked_logits(
    logits: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mask logits for categorical ops without all-false ``log_softmax`` NaNs."""

    mask = mask.astype(bool)
    any_valid = mask.any(axis=-1)
    safe_mask = jnp.where(any_valid[..., None], mask, jnp.ones_like(mask, dtype=bool))
    safe_logits = jnp.where(safe_mask, logits, _SAFE_NEG_LOGIT)
    return safe_logits, any_valid.astype(jnp.float32)


def _safe_categorical_log_prob(
    logits: jax.Array,
    mask: jax.Array,
    action: jax.Array,
    *,
    active: jax.Array,
) -> jax.Array:
    """Categorical log-prob that returns 0 for inactive or all-false rows."""

    safe_logits, any_valid = _safe_masked_logits(logits, mask)
    log_probs = jax.nn.log_softmax(safe_logits, axis=-1)
    action = jnp.clip(action, 0, logits.shape[-1] - 1)
    selected = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
    return jnp.where((active > 0.0) & (any_valid > 0.0), selected, 0.0)


def _safe_categorical_entropy(
    logits: jax.Array,
    mask: jax.Array,
    *,
    active: jax.Array,
) -> jax.Array:
    """Categorical entropy that returns 0 for inactive or all-false rows."""

    safe_logits, any_valid = _safe_masked_logits(logits, mask)
    log_probs = jax.nn.log_softmax(safe_logits, axis=-1)
    probs = jax.nn.softmax(safe_logits, axis=-1)
    probs = jnp.where(mask, probs, 0.0)
    log_probs = jnp.where(mask, log_probs, 0.0)
    entropy = -(probs * log_probs).sum(axis=-1)
    return jnp.where((active > 0.0) & (any_valid > 0.0), entropy, 0.0)


def _masked_categorical_entropy(probs: jax.Array, log_probs: jax.Array) -> jax.Array:
    """Sum entropy terms while ignoring zero-mass buckets (``0 * -inf``)."""

    return -(probs * jnp.where(probs > 0.0, log_probs, 0.0)).sum(axis=-1)


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


def planet_flow_action_log_prob_entropy(
    output: PlanetFlowPolicyOutput,
    target_bucket: jax.Array,
    target_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Replay log-probability and entropy over sampled pressure buckets."""

    logits = output.target_demand_logits
    target_mask = target_mask.astype(bool)
    target_bucket = jnp.where(target_mask, target_bucket, jnp.zeros_like(target_bucket))
    bucket_mask = jnp.ones_like(logits, dtype=bool)
    active = target_mask.astype(jnp.float32)
    log_prob = _safe_categorical_log_prob(
        logits,
        bucket_mask,
        target_bucket.astype(jnp.int32),
        active=active,
    )
    entropy = _safe_categorical_entropy(logits, bucket_mask, active=active)
    return log_prob.sum(axis=-1), entropy.sum(axis=-1)


def planet_flow_categorical_kl(
    old_output: PlanetFlowPolicyOutput,
    new_output: PlanetFlowPolicyOutput,
    target_mask: jax.Array,
) -> jax.Array:
    """Exact categorical KL over active Planet Flow pressure heads."""

    old_logits = old_output.target_demand_logits
    new_logits = new_output.target_demand_logits
    old_log_probs = jax.nn.log_softmax(old_logits, axis=-1)
    new_log_probs = jax.nn.log_softmax(new_logits, axis=-1)
    old_probs = jax.nn.softmax(old_logits, axis=-1)
    per_target_kl = (old_probs * (old_log_probs - new_log_probs)).sum(axis=-1)
    return jnp.where(target_mask, per_target_kl, 0.0).sum(axis=-1)


def planet_flow_invalid_bucket_count(
    target_bucket: jax.Array,
    bucket_count: int,
    target_mask: jax.Array,
) -> jax.Array:
    """Count active target positions whose stored bucket cannot be replayed."""

    invalid = (target_bucket < 0) | (target_bucket >= int(bucket_count))
    return (invalid & target_mask.astype(bool)).sum()


def sample_planet_flow_pressure_action(
    key: jax.Array,
    output: PlanetFlowPolicyOutput,
    pressure_bucket_values: jax.Array,
    target_mask: jax.Array,
    *,
    deterministic: bool,
) -> PlanetFlowPressureAction:
    """Sample the target-demand pressure action for Planet Flow P0."""

    logits = output.target_demand_logits
    sampled = jnp.where(
        deterministic,
        jnp.argmax(logits, axis=-1),
        jax.random.categorical(key, logits, axis=-1),
    ).astype(jnp.int32)
    target_mask = target_mask.astype(bool)
    target_bucket = jnp.where(target_mask, sampled, jnp.zeros_like(sampled))
    bucket_values = jnp.asarray(pressure_bucket_values, dtype=jnp.float32)
    target_pressure = jnp.take(bucket_values, target_bucket, axis=0)
    target_pressure = jnp.where(target_mask, target_pressure, 0.0)
    log_prob, entropy = planet_flow_action_log_prob_entropy(
        output, target_bucket, target_mask
    )
    return PlanetFlowPressureAction(
        target_bucket=target_bucket,
        target_pressure=target_pressure,
        log_prob=log_prob,
        entropy=entropy,
        target_mask=target_mask,
    )


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
    ship_fraction: jax.Array | None = None,
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
    if ship_fraction is not None:
        ship_fraction = ensure_action_sequence(ship_fraction.astype(jnp.float32))

    stop_log_probs = jax.nn.log_sigmoid(stop_logits)
    stop_log_probs_neg = jax.nn.log_sigmoid(-stop_logits)
    stop_probs = jax.nn.sigmoid(stop_logits)

    head_active = step_mask * (1.0 - stop_flag)
    launch_active = head_active
    source_lp = _safe_categorical_log_prob(
        source_logits,
        jnp.ones_like(source_logits, dtype=bool),
        source_index,
        active=launch_active,
    )
    target_lp = _safe_categorical_log_prob(
        target_logits,
        jnp.ones_like(target_logits, dtype=bool),
        target_slot,
        active=launch_active,
    )
    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target_slot[..., None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=2,
    ).squeeze(axis=2)
    if selected_ship_logits.shape[-1] == 1:
        policy_loc = selected_ship_logits[..., 0]
        if ship_fraction is not None:
            from src.jax.ship_action import continuous_fraction_log_prob_at_action

            ship_lp = continuous_fraction_log_prob_at_action(policy_loc, ship_fraction)
        else:
            ship_lp = _continuous_fraction_log_prob(policy_loc)
        ship_entropy = jnp.zeros_like(ship_lp)
        ship_lp = jnp.where(launch_active > 0.0, ship_lp, 0.0)
    else:
        ship_lp = _safe_categorical_log_prob(
            selected_ship_logits,
            jnp.ones_like(selected_ship_logits, dtype=bool),
            ship_bucket,
            active=launch_active,
        )
        ship_entropy = _safe_categorical_entropy(
            selected_ship_logits,
            jnp.ones_like(selected_ship_logits, dtype=bool),
            active=launch_active,
        )
    source_entropy = _safe_categorical_entropy(
        source_logits,
        jnp.ones_like(source_logits, dtype=bool),
        active=launch_active,
    )
    target_entropy = _safe_categorical_entropy(
        target_logits,
        jnp.ones_like(target_logits, dtype=bool),
        active=launch_active,
    )
    stop_lp = stop_flag * stop_log_probs + (1.0 - stop_flag) * stop_log_probs_neg
    stop_entropy = -(
        stop_probs * stop_log_probs + (1.0 - stop_probs) * stop_log_probs_neg
    )
    stop_lp = jnp.where(step_mask > 0.0, stop_lp, 0.0)
    stop_entropy = jnp.where(step_mask > 0.0, stop_entropy, 0.0)
    move_lp = jnp.where(launch_active > 0.0, source_lp + target_lp + ship_lp, 0.0)
    move_entropy = jnp.where(
        launch_active > 0.0, source_entropy + target_entropy + ship_entropy, 0.0
    )
    log_prob = stop_lp + move_lp
    entropy = stop_entropy + move_entropy
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
    ship_fraction: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Match rollout ``_sample_factored_step_from_logits`` log-prob math (batched)."""

    stop = stop_flag.astype(jnp.float32)
    stop_log_probs = jax.nn.log_sigmoid(stop_logit)
    stop_log_probs_neg = jax.nn.log_sigmoid(-stop_logit)
    stop_lp = stop * stop_log_probs + (1.0 - stop) * stop_log_probs_neg
    stop_prob = jax.nn.sigmoid(stop_logit)
    stop_entropy = -(
        stop_prob * stop_log_probs + (1.0 - stop_prob) * stop_log_probs_neg
    )

    batch_size = source_logits.shape[0]
    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)
    source_mask = source_mask.astype(bool)
    source_mask = source_mask.at[batch_idx, source_index].set(True)

    launch_active = 1.0 - stop
    source_lp = _safe_categorical_log_prob(
        source_logits,
        source_mask,
        source_index,
        active=launch_active,
    )
    source_entropy = _safe_categorical_entropy(
        source_logits,
        source_mask,
        active=launch_active,
    )

    row_bucket_mask = ship_bucket_mask[batch_idx, source_index]
    target_mask = row_bucket_mask.any(axis=-1)
    target_mask = target_mask.at[batch_idx, target_slot].set(True)
    target_lp = _safe_categorical_log_prob(
        target_logits,
        target_mask,
        target_slot,
        active=launch_active,
    )
    target_entropy = _safe_categorical_entropy(
        target_logits,
        target_mask,
        active=launch_active,
    )

    selected_bucket_mask = row_bucket_mask[batch_idx, target_slot]
    selected_bucket_mask = selected_bucket_mask.at[batch_idx, ship_bucket].set(True)
    selected_ship_logits = ship_logits[batch_idx, target_slot]
    if selected_ship_logits.shape[-1] == 1:
        policy_loc = jnp.squeeze(selected_ship_logits, axis=-1)
        selected_target_legal = target_mask[batch_idx, target_slot]
        if ship_fraction is not None:
            from src.jax.ship_action import continuous_fraction_log_prob_at_action

            policy_loc = jnp.where(
                selected_target_legal, policy_loc, _SAFE_NEG_LOGIT
            )
            ship_lp = continuous_fraction_log_prob_at_action(policy_loc, ship_fraction)
        else:
            policy_loc = jnp.where(selected_target_legal, policy_loc, _SAFE_NEG_LOGIT)
            ship_lp = _continuous_fraction_log_prob(policy_loc)
        ship_entropy = jnp.zeros_like(ship_lp)
        ship_lp = jnp.where(launch_active > 0.0, ship_lp, 0.0)
    else:
        ship_lp = _safe_categorical_log_prob(
            selected_ship_logits,
            selected_bucket_mask,
            ship_bucket,
            active=launch_active,
        )
        ship_entropy = _safe_categorical_entropy(
            selected_ship_logits,
            selected_bucket_mask,
            active=launch_active,
        )

    move_lp = jnp.where(launch_active > 0.0, source_lp + target_lp + ship_lp, 0.0)
    move_entropy = jnp.where(
        launch_active > 0.0, source_entropy + target_entropy + ship_entropy, 0.0
    )
    log_prob = stop_lp + move_lp
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
    ship_fraction: jax.Array | None = None,
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
    if ship_fraction is not None:
        ship_fraction = ensure_action_sequence(ship_fraction.astype(jnp.float32))

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
        frac,
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
            ship_fraction=frac,
        )
        active = mask.astype(jnp.float32)
        return (
            jnp.where(active > 0.0, log_prob, 0.0),
            jnp.where(active > 0.0, entropy, 0.0),
            jnp.where(active > 0.0, stop_entropy, 0.0),
            jnp.where(active > 0.0, move_entropy, 0.0),
        )

    frac_seq = ship_fraction
    if frac_seq is None:
        frac_seq = jnp.zeros_like(step_mask)

    log_prob, entropy, stop_entropy, move_entropy = jax.vmap(
        one_step, in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), out_axes=1
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
        frac_seq,
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
    ship_fraction: jax.Array | None = None,
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
    if selected_ship_logits.shape[-1] == 1:
        policy_loc = selected_ship_logits[..., 0]
        if ship_fraction is not None:
            from src.jax.ship_action import continuous_fraction_log_prob_at_action

            ship_lp = continuous_fraction_log_prob_at_action(policy_loc, ship_fraction)
        else:
            ship_lp = _continuous_fraction_log_prob(policy_loc)
        ship_entropy = jnp.zeros_like(ship_lp)
    else:
        ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
        ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
        ship_lp = jnp.take_along_axis(
            ship_log_probs, ship_bucket[..., None], axis=-1
        ).squeeze(-1)
        ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)
    target_entropy = -(target_probs * target_log_probs).sum(axis=-1)
    log_prob = target_lp + ship_lp
    entropy = target_entropy + ship_entropy
    if squeeze_sequence:
        return log_prob[:, 0], entropy[:, 0]
    return log_prob, entropy
