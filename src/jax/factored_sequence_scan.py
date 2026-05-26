"""Shared stepwise factorized sequence scan for rollout sampling and PPO replay.

Rollout and PPO must use the same prefix-growing ``policy.apply`` loop so stored
actions receive logits identical to those seen during sampling.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry import PLANET_FEATURE_SCHEMA
from src.jax.action_codec import (
    _factored_step_log_prob_entropy,
    source_mask_from_bucket_mask_and_ships,
)
from src.jax.decoder_carry import decoder_carry_enabled
from src.jax.features import TurnBatch, ship_feature_scale
from src.jax.ship_action import is_continuous_ship_mode, ship_count_for_action


class FactoredSequenceLogProbResult(NamedTuple):
    """Per-step log-prob replay aligned with rollout sampling."""

    log_prob: jax.Array
    entropy: jax.Array
    stop_entropy: jax.Array
    move_entropy: jax.Array


def owned_planet_ships_from_turn_batch(
    batch: TurnBatch,
    task_cfg,
) -> jax.Array:
    """Reconstruct learner-owned planet ship counts from encoded planet features."""

    ships_slice = PLANET_FEATURE_SCHEMA.base_slice("ships")
    owner_slice = PLANET_FEATURE_SCHEMA.base_slice("owner_slot")
    scale = ship_feature_scale(task_cfg)
    ships = batch.planet_features[..., ships_slice].squeeze(-1) * scale
    owner_slot = batch.planet_features[..., owner_slice]
    owned = batch.planet_mask & (owner_slot[..., 0] > 0.5)
    return jnp.where(owned, ships, 0.0)


def replay_factored_sequence_logprob(
    params: dict,
    policy: object,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    player_count: jax.Array,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_bucket_mask: jax.Array,
    ship_fraction: jax.Array | None = None,
    decoder_hidden: jax.Array | None = None,
    initial_remaining_ships: jax.Array | None = None,
) -> FactoredSequenceLogProbResult:
    """Replay stored factorized actions with rollout-matching prefix decoding."""

    env_count = batch.planet_features.shape[0]
    sequence_k = source_index.shape[1]
    continuous = is_continuous_ship_mode(cfg)
    carry_enabled = decoder_carry_enabled(cfg)

    from src.opponents.jax_actions.builders import ship_count_for_bucket_jax

    source_prefix = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    slot_prefix = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    if initial_remaining_ships is not None:
        remaining_ships = initial_remaining_ships.astype(jnp.float32)
    else:
        remaining_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)
    sequence_active = jnp.ones((env_count,), dtype=bool)

    log_prob_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    entropy_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    stop_entropy_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    move_entropy_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)

    def scan_step(carry, step_idx):
        (
            source_prefix,
            slot_prefix,
            remaining_ships,
            sequence_active,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
        ) = carry

        step_active = sequence_active & (step_mask[:, step_idx] > 0.0)
        apply_kwargs = {
            "player_count": player_count,
            "source_sequence": source_prefix,
            "target_slot_sequence": slot_prefix,
            "deterministic": True,
        }
        if carry_enabled and decoder_hidden is not None:
            apply_kwargs["decoder_hidden"] = decoder_hidden
        step_output = policy.apply(params, batch, **apply_kwargs)

        step_bucket_mask = ship_bucket_mask[:, step_idx]
        source_mask = jax.vmap(source_mask_from_bucket_mask_and_ships, in_axes=(0, 0))(
            step_bucket_mask, remaining_ships
        )
        stored_stop = stop_flag[:, step_idx]
        stored_source = source_index[:, step_idx]
        stored_target = target_slot[:, step_idx]
        stored_bucket = ship_bucket[:, step_idx]
        fraction_arg = None
        if continuous and ship_fraction is not None:
            fraction_arg = ship_fraction[:, step_idx]

        step_lp, step_ent, stop_ent, move_ent = _factored_step_log_prob_entropy(
            step_output.source_logits[:, step_idx, :],
            step_output.target_logits[:, step_idx, :],
            step_output.stop_logits[:, step_idx],
            step_output.ship_logits[:, step_idx, :, :],
            source_mask,
            step_bucket_mask,
            stored_source,
            stored_target,
            stored_bucket,
            stored_stop.astype(jnp.float32),
            ship_fraction=fraction_arg,
        )
        active_f = step_active.astype(jnp.float32)
        step_lp = step_lp * active_f
        step_ent = step_ent * active_f
        stop_ent = stop_ent * active_f
        move_ent = move_ent * active_f

        log_prob_out = log_prob_out.at[:, step_idx].set(step_lp)
        entropy_out = entropy_out.at[:, step_idx].set(step_ent)
        stop_entropy_out = stop_entropy_out.at[:, step_idx].set(stop_ent)
        move_entropy_out = move_entropy_out.at[:, step_idx].set(move_ent)

        stop_bool = stored_stop.astype(bool) & step_active
        src_rows = stored_source
        batch_idx = jnp.arange(env_count, dtype=jnp.int32)
        current_source_ships = remaining_ships[batch_idx, src_rows]
        if continuous and fraction_arg is not None:
            launched = jax.vmap(
                lambda ships, bucket, fraction: ship_count_for_action(
                    ships, bucket, fraction, cfg
                )
            )(current_source_ships, stored_bucket, fraction_arg)
            launch_valid = (
                step_active
                & jnp.logical_not(stop_bool)
                & (launched > 0.0)
                & (fraction_arg > 0.0)
            )
        else:
            launched = jax.vmap(
                lambda ships, bucket: ship_count_for_bucket_jax(
                    ships, bucket, cfg.task.ship_bucket_count
                )
            )(current_source_ships, stored_bucket)
            launch_valid = (
                step_active
                & jnp.logical_not(stop_bool)
                & (launched > 0.0)
                & (stored_bucket > 0)
            )
        remaining_ships = remaining_ships.at[batch_idx, src_rows].set(
            jnp.where(
                launch_valid,
                jnp.maximum(current_source_ships - launched, 0.0),
                current_source_ships,
            )
        )
        source_prefix = source_prefix.at[:, step_idx].set(stored_source)
        slot_prefix = slot_prefix.at[:, step_idx].set(stored_target)
        sequence_active = sequence_active & jnp.logical_not(stop_bool)

        return (
            source_prefix,
            slot_prefix,
            remaining_ships,
            sequence_active,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
        ), None

    (
        (
            _source_prefix,
            _slot_prefix,
            _remaining_ships,
            _sequence_active,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
        ),
        _,
    ) = jax.lax.scan(
        scan_step,
        (
            source_prefix,
            slot_prefix,
            remaining_ships,
            sequence_active,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
        ),
        jnp.arange(sequence_k, dtype=jnp.int32),
    )
    return FactoredSequenceLogProbResult(
        log_prob=log_prob_out,
        entropy=entropy_out,
        stop_entropy=stop_entropy_out,
        move_entropy=move_entropy_out,
    )


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean over elements where ``mask > 0``."""

    denom = jnp.maximum(mask.sum(), 1.0)
    return (x * mask).sum() / denom


def factored_logprob_parity_metrics(
    params: dict,
    policy: object,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    player_count: jax.Array,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_bucket_mask: jax.Array,
    old_log_prob: jax.Array,
    ship_fraction: jax.Array | None = None,
    decoder_hidden: jax.Array | None = None,
    initial_remaining_ships: jax.Array | None = None,
    advantages: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Pre-update rollout↔replay parity diagnostics for factorized PPO."""

    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_index=source_index,
        target_slot=target_slot,
        ship_bucket=ship_bucket,
        stop_flag=stop_flag,
        step_mask=step_mask,
        ship_bucket_mask=ship_bucket_mask,
        ship_fraction=ship_fraction,
        decoder_hidden=decoder_hidden,
        initial_remaining_ships=initial_remaining_ships,
    )
    new_log_prob = replay.log_prob
    delta = new_log_prob - old_log_prob
    mask = step_mask.astype(jnp.float32)
    log_ratio = delta
    ratio20 = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
    seq_valid = mask
    approx_kl_v1 = masked_mean(old_log_prob - new_log_prob, mask)
    approx_kl_v2 = masked_mean((ratio20 - 1.0) - log_ratio, mask)
    metrics: dict[str, jax.Array] = {
        "debug/logprob_delta_mean": masked_mean(delta, mask),
        "debug/logprob_delta_abs_mean": masked_mean(jnp.abs(delta), mask),
        "debug/logprob_delta_abs_max": jnp.max(
            jnp.where(mask > 0.0, jnp.abs(delta), 0.0)
        ),
        "debug/ratio_pre_update_mean": masked_mean(ratio20, mask),
        "debug/approx_kl_v1": approx_kl_v1,
        "debug/approx_kl_v2": approx_kl_v2,
        "debug/new_log_prob_finite": jnp.all(jnp.isfinite(new_log_prob)).astype(
            jnp.float32
        ),
        "debug/entropy_finite": jnp.all(jnp.isfinite(replay.entropy)).astype(
            jnp.float32
        ),
        "debug/old_log_prob_min": jnp.min(old_log_prob),
        "debug/old_log_prob_max": jnp.max(old_log_prob),
        "debug/new_log_prob_min": jnp.min(new_log_prob),
        "debug/new_log_prob_max": jnp.max(new_log_prob),
        "debug/old_log_prob_at_neg100_frac": masked_mean(
            (old_log_prob <= -100.0).astype(jnp.float32), mask
        ),
        "debug/old_log_prob_at_pos100_frac": masked_mean(
            (old_log_prob >= 100.0).astype(jnp.float32), mask
        ),
        "debug/log_ratio_min": jnp.min(log_ratio),
        "debug/log_ratio_max": jnp.max(log_ratio),
        "debug/log_ratio_abs_mean": masked_mean(jnp.abs(log_ratio), mask),
        "debug/log_ratio_gt_1_frac": masked_mean(
            (jnp.abs(log_ratio) > 1.0).astype(jnp.float32), mask
        ),
        "debug/log_ratio_gt_5_frac": masked_mean(
            (jnp.abs(log_ratio) > 5.0).astype(jnp.float32), mask
        ),
        "debug/log_ratio_gt_20_frac": masked_mean(
            (jnp.abs(log_ratio) > 20.0).astype(jnp.float32), mask
        ),
        "debug/ratio20_mean": masked_mean(ratio20, mask),
        "debug/ratio20_max": jnp.max(jnp.where(mask > 0.0, ratio20, 0.0)),
    }
    if advantages is not None:
        metrics.update(
            {
                "debug/adv_min": jnp.min(advantages),
                "debug/adv_max": jnp.max(advantages),
                "debug/adv_abs_mean": masked_mean(jnp.abs(advantages), mask),
                "debug/neg_adv_ratio20_objective_mean": masked_mean(
                    jnp.where(advantages < 0.0, ratio20 * advantages, 0.0),
                    mask,
                ),
            }
        )
    return metrics
