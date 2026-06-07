"""Shared stepwise factorized sequence scan for rollout sampling and PPO replay.

Rollout sampling and PPO replay both score sub-steps with the same shield-prefix
decoder semantics: teacher columns ``0..step_idx-1`` committed, column
``step_idx`` zeroed (see :func:`build_shield_prefix_teacher_sequences`). Each path
encodes ``TurnBatch`` once and reuses ``PlanetEdgeEncoderOutput`` for decoder-only
prefix forwards.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry import PLANET_FEATURE_SCHEMA, edge_k
from src.game.constants import MAX_PLANETS
from src.jax.action_codec import (
    FactoredPolicyOutput,
    _factored_step_log_prob_entropy,
    source_mask_from_bucket_mask_and_ships,
)
from src.jax.array_ops import masked_mean
from src.jax.decoder_carry import decoder_carry_enabled
from src.jax.features import TurnBatch, ship_feature_scale
from src.jax.launch_hygiene import (
    apply_cumulative_forbidden_to_shield,
    apply_launch_to_cumulative_forbidden,
    build_hygiene_lookups,
    compose_hygiene_with_shield_mask,
    empty_cumulative_forbidden,
)
from src.jax.ship_action import is_continuous_ship_mode, ship_count_for_action
from src.opponents.jax_actions.builders import ship_count_for_bucket_jax


class FactoredSequenceLogProbResult(NamedTuple):
    """Per-step log-prob replay aligned with rollout sampling."""

    log_prob: jax.Array
    entropy: jax.Array
    stop_entropy: jax.Array
    move_entropy: jax.Array
    value: jax.Array | None = None
    value_logits: jax.Array | None = None


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


def _replay_logprobs_with_prefix_forwards(
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
    ship_fraction: jax.Array | None,
    initial_remaining_ships: jax.Array | None,
    decoder_hidden: jax.Array | None = None,
    encoder_out=None,
) -> FactoredSequenceLogProbResult:
    """Replay log-probs with one shield-prefix decoder forward per sub-step."""

    env_count = source_index.shape[0]
    sequence_k = source_index.shape[1]
    continuous = is_continuous_ship_mode(cfg)

    if initial_remaining_ships is not None:
        remaining_ships = initial_remaining_ships.astype(jnp.float32)
    else:
        remaining_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)

    log_prob_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    entropy_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    stop_entropy_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    move_entropy_out = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    use_distributional = cfg.model.value_head.strip().lower() == "distributional"
    value_logits_out: jax.Array | None = (
        jnp.zeros((env_count, cfg.model.value_bins), dtype=jnp.float32)
        if use_distributional
        else None
    )

    if encoder_out is None:
        encoder_out = forward_factorized_encode(params, policy, batch)
    value_out = forward_factorized_critic(
        params, policy, encoder_out, player_count=player_count
    )
    value_out_scalar = value_out.value
    if value_out.value_logits is not None:
        value_logits_out = value_out.value_logits

    hygiene_lookups = build_hygiene_lookups(batch)
    k = edge_k(cfg.task)
    cumulative_forbidden = empty_cumulative_forbidden(
        env_count,
        num_planets=MAX_PLANETS,
        max_k=k,
        buckets=cfg.task.ship_bucket_count,
        max_launches=sequence_k,
    )

    def scan_step(carry, step_idx):
        (
            remaining_ships,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
            cumulative_forbidden,
        ) = carry

        source_prefix, target_prefix = build_shield_prefix_teacher_sequences(
            source_index, target_slot, step_idx
        )
        step_output = forward_factored_policy(
            params,
            policy,
            batch,
            cfg,
            player_count=player_count,
            source_sequence=source_prefix,
            target_slot_sequence=target_prefix,
            decoder_hidden=decoder_hidden,
            deterministic=True,
            encoder_out=encoder_out,
        )
        step_active = step_mask[:, step_idx] > 0.0
        step_bucket_mask = apply_cumulative_forbidden_to_shield(
            ship_bucket_mask[:, step_idx],
            cumulative_forbidden,
        )
        fraction_arg = None
        if continuous and ship_fraction is not None:
            fraction_arg = ship_fraction[:, step_idx]
        source_mask = jax.vmap(source_mask_from_bucket_mask_and_ships, in_axes=(0, 0))(
            step_bucket_mask, remaining_ships
        )
        step_lp, step_ent, stop_ent, move_ent = _factored_step_log_prob_entropy(
            step_output.source_logits[:, step_idx, :],
            step_output.target_logits[:, step_idx, :],
            step_output.stop_logits[:, step_idx],
            step_output.ship_logits[:, step_idx, :, :],
            source_mask,
            step_bucket_mask,
            source_index[:, step_idx],
            target_slot[:, step_idx],
            ship_bucket[:, step_idx],
            stop_flag[:, step_idx],
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

        stored_stop = stop_flag[:, step_idx]
        stored_source = source_index[:, step_idx]
        stored_bucket = ship_bucket[:, step_idx]
        stop_bool = stored_stop.astype(bool) & step_active
        batch_idx = jnp.arange(env_count, dtype=jnp.int32)
        src_rows = stored_source
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
        cumulative_forbidden = apply_launch_to_cumulative_forbidden(
            cumulative_forbidden,
            batch=batch,
            lookups=hygiene_lookups,
            src_row=stored_source,
            slot=target_slot[:, step_idx],
            active=launch_valid,
        )
        remaining_ships = remaining_ships.at[batch_idx, src_rows].set(
            jnp.where(
                launch_valid,
                jnp.maximum(current_source_ships - launched, 0.0),
                current_source_ships,
            )
        )

        return (
            remaining_ships,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
            cumulative_forbidden,
        ), None

    (
        (
            _remaining_ships,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
            _cumulative_forbidden,
        ),
        _,
    ) = jax.lax.scan(
        scan_step,
        (
            remaining_ships,
            log_prob_out,
            entropy_out,
            stop_entropy_out,
            move_entropy_out,
            cumulative_forbidden,
        ),
        jnp.arange(sequence_k, dtype=jnp.int32),
    )
    return FactoredSequenceLogProbResult(
        log_prob=log_prob_out,
        entropy=entropy_out,
        stop_entropy=stop_entropy_out,
        move_entropy=move_entropy_out,
        value=value_out_scalar,
        value_logits=value_logits_out,
    )


def forward_factorized_encode(
    params: dict,
    policy: object,
    batch: TurnBatch,
):
    """Run the planet encoder once for a fixed ``TurnBatch``."""

    from src.jax.policy import factorized_encode

    return factorized_encode(params, policy, batch)


def forward_factorized_critic(
    params: dict,
    policy: object,
    encoder_out,
    *,
    player_count: jax.Array,
):
    """Run the critic head on cached encoder output."""

    from src.jax.policy import factorized_critic

    return factorized_critic(params, policy, encoder_out, player_count=player_count)


def forward_factored_policy(
    params: dict,
    policy: object,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    player_count: jax.Array,
    source_sequence: jax.Array,
    target_slot_sequence: jax.Array,
    decoder_hidden: jax.Array | None = None,
    deterministic: bool = True,
    encoder_out=None,
) -> FactoredPolicyOutput:
    """Run one factorized decoder forward with explicit teacher prefix sequences."""

    if encoder_out is None:
        encoder_out = forward_factorized_encode(params, policy, batch)
    apply_kwargs = {
        "player_count": player_count,
        "source_sequence": source_sequence,
        "target_slot_sequence": target_slot_sequence,
        "deterministic": deterministic,
    }
    if decoder_carry_enabled(cfg) and decoder_hidden is not None:
        apply_kwargs["decoder_hidden"] = decoder_hidden
    from src.jax.policy import factorized_decode

    return factorized_decode(
        params,
        policy,
        encoder_out,
        include_value=False,
        **apply_kwargs,
    )


def forward_factored_replay_policy(
    params: dict,
    policy: object,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    player_count: jax.Array,
    sequence_k: int,
    decoder_hidden: jax.Array | None = None,
) -> FactoredPolicyOutput:
    """Match the zeroed teacher sequences used by :func:`replay_factored_sequence_logprob`."""

    zeros_source = jnp.zeros(
        (batch.planet_features.shape[0], sequence_k), dtype=jnp.int32
    )
    zeros_target = jnp.zeros_like(zeros_source)
    return forward_factored_policy(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_sequence=zeros_source,
        target_slot_sequence=zeros_target,
        decoder_hidden=decoder_hidden,
        deterministic=True,
    )


def build_shield_prefix_teacher_sequences(
    source_index: jax.Array,
    target_slot: jax.Array,
    step_idx: int,
) -> tuple[jax.Array, jax.Array]:
    """Teacher sequences matching shield scan ``policy.apply`` at ``step_idx``."""

    step_cols = jnp.arange(source_index.shape[1], dtype=jnp.int32)
    committed = step_cols < jnp.asarray(step_idx, dtype=jnp.int32)
    source_prefix = jnp.where(committed[None, :], source_index, 0)
    target_prefix = jnp.where(committed[None, :], target_slot, 0)
    return source_prefix, target_prefix


def remaining_ships_before_step(
    cfg: TrainConfig,
    *,
    initial_remaining_ships: jax.Array,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_fraction: jax.Array | None,
    step_idx: int,
) -> jax.Array:
    """Reconstruct garrisons before sub-step ``step_idx`` (replay scan semantics)."""

    from src.jax.ship_action import is_continuous_ship_mode

    remaining = initial_remaining_ships.astype(jnp.float32)
    env_count = source_index.shape[0]
    batch_idx = jnp.arange(env_count, dtype=jnp.int32)
    continuous = is_continuous_ship_mode(cfg)

    def body(carry, prior_step):
        remaining_ships = carry
        step_active = step_mask[:, prior_step] > 0.0
        stored_stop = stop_flag[:, prior_step]
        stored_source = source_index[:, prior_step]
        stored_bucket = ship_bucket[:, prior_step]
        stop_bool = stored_stop.astype(bool) & step_active
        src_rows = stored_source
        current_source_ships = remaining_ships[batch_idx, src_rows]
        if continuous and ship_fraction is not None:
            fraction_arg = ship_fraction[:, prior_step]
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
        return remaining_ships.at[batch_idx, src_rows].set(
            jnp.where(
                launch_valid,
                jnp.maximum(current_source_ships - launched, 0.0),
                current_source_ships,
            )
        ), None

    if step_idx <= 0:
        return remaining
    return jax.lax.fori_loop(0, step_idx, lambda i, r: body(r, i)[0], remaining)


def _hygiene_adjusted_step_mask(
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    ship_bucket_mask: jax.Array,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_fraction: jax.Array | None,
    step_idx: int,
) -> jax.Array:
    """Recompute prefix-derived hygiene on shield-only ``ship_bucket_mask``."""

    return compose_hygiene_with_shield_mask(
        batch,
        ship_bucket_mask[:, step_idx],
        source_sequence=source_index,
        target_slot_sequence=target_slot,
        stop_flag=stop_flag,
        step_mask=step_mask,
        ship_bucket=ship_bucket,
        ship_fraction=ship_fraction,
        cfg=cfg,
        step_idx=step_idx,
    )


def factored_step_logprob_at_index(
    step_output: FactoredPolicyOutput,
    cfg: TrainConfig,
    step_idx: int,
    *,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    ship_bucket_mask: jax.Array,
    remaining_ships: jax.Array,
    ship_fraction: jax.Array | None = None,
    batch: TurnBatch | None = None,
    step_mask: jax.Array | None = None,
    hygiene_bucket_mask: jax.Array | None = None,
) -> jax.Array:
    """Log-prob of stored actions at ``step_idx`` under ``step_output`` logits."""

    if hygiene_bucket_mask is not None:
        step_bucket_mask = hygiene_bucket_mask
    else:
        step_bucket_mask = ship_bucket_mask[:, step_idx]
        if batch is not None:
            step_bucket_mask = _hygiene_adjusted_step_mask(
                batch,
                cfg,
                ship_bucket_mask=ship_bucket_mask,
                source_index=source_index,
                target_slot=target_slot,
                ship_bucket=ship_bucket,
                stop_flag=stop_flag,
                step_mask=step_mask
                if step_mask is not None
                else jnp.ones_like(stop_flag),
                ship_fraction=ship_fraction,
                step_idx=step_idx,
            )
    source_mask = jax.vmap(source_mask_from_bucket_mask_and_ships, in_axes=(0, 0))(
        step_bucket_mask, remaining_ships
    )
    fraction_arg = None
    if ship_fraction is not None:
        fraction_arg = ship_fraction[:, step_idx]
    step_lp, _, _, _ = _factored_step_log_prob_entropy(
        step_output.source_logits[:, step_idx, :],
        step_output.target_logits[:, step_idx, :],
        step_output.stop_logits[:, step_idx],
        step_output.ship_logits[:, step_idx, :, :],
        source_mask,
        step_bucket_mask,
        source_index[:, step_idx],
        target_slot[:, step_idx],
        ship_bucket[:, step_idx],
        stop_flag[:, step_idx],
        ship_fraction=fraction_arg,
    )
    return step_lp


def prefix_replay_step_logprob_deltas(
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
) -> dict[str, jax.Array]:
    """Per-step |log p_prefix - log p_replay| for active sub-steps."""

    if initial_remaining_ships is not None:
        initial_ships = initial_remaining_ships.astype(jnp.float32)
    else:
        initial_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)

    sequence_k = source_index.shape[1]
    replay_out = forward_factored_replay_policy(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        sequence_k=sequence_k,
        decoder_hidden=decoder_hidden,
    )

    deltas = []
    active_flags = []
    for step_idx in range(sequence_k):
        active = step_mask[:, step_idx] > 0.0
        if not bool(jnp.any(active)):
            continue
        ships_before = remaining_ships_before_step(
            cfg,
            initial_remaining_ships=initial_ships,
            source_index=source_index,
            target_slot=target_slot,
            ship_bucket=ship_bucket,
            stop_flag=stop_flag,
            step_mask=step_mask,
            ship_fraction=ship_fraction,
            step_idx=step_idx,
        )
        source_prefix, target_prefix = build_shield_prefix_teacher_sequences(
            source_index, target_slot, step_idx
        )
        prefix_out = forward_factored_policy(
            params,
            policy,
            batch,
            cfg,
            player_count=player_count,
            source_sequence=source_prefix,
            target_slot_sequence=target_prefix,
            decoder_hidden=decoder_hidden,
            deterministic=True,
        )
        lp_prefix = factored_step_logprob_at_index(
            prefix_out,
            cfg,
            step_idx,
            source_index=source_index,
            target_slot=target_slot,
            ship_bucket=ship_bucket,
            stop_flag=stop_flag,
            ship_bucket_mask=ship_bucket_mask,
            remaining_ships=ships_before,
            ship_fraction=ship_fraction,
            batch=batch,
            step_mask=step_mask,
        )
        lp_replay = factored_step_logprob_at_index(
            replay_out,
            cfg,
            step_idx,
            source_index=source_index,
            target_slot=target_slot,
            ship_bucket=ship_bucket,
            stop_flag=stop_flag,
            ship_bucket_mask=ship_bucket_mask,
            remaining_ships=ships_before,
            ship_fraction=ship_fraction,
            batch=batch,
            step_mask=step_mask,
        )
        deltas.append(jnp.abs(lp_prefix - lp_replay))
        active_flags.append(active.astype(jnp.float32))

    if not deltas:
        zero = jnp.zeros((source_index.shape[0],), dtype=jnp.float32)
        return {
            "delta_abs": zero,
            "delta_abs_max": jnp.array(0.0, dtype=jnp.float32),
            "delta_abs_mean_active": jnp.array(0.0, dtype=jnp.float32),
        }

    stacked = jnp.stack(deltas, axis=1)
    mask = jnp.stack(active_flags, axis=1)
    masked = stacked * mask
    denom = jnp.maximum(mask.sum(), 1.0)
    return {
        "delta_abs": stacked,
        "delta_abs_max": jnp.max(masked),
        "delta_abs_mean_active": masked.sum() / denom,
    }


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
    encoder_out=None,
) -> FactoredSequenceLogProbResult:
    """Replay stored factorized actions with rollout-matching prefix decoding."""

    return _replay_logprobs_with_prefix_forwards(
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
        initial_remaining_ships=initial_remaining_ships,
        decoder_hidden=decoder_hidden,
        encoder_out=encoder_out,
    )


def rollout_replay_parity_summary(
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
) -> dict[str, jax.Array]:
    """Lightweight rollout↔replay parity check at fixed policy params."""

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
    return {
        "parity_logprob_delta_abs_mean": masked_mean(jnp.abs(delta), mask),
        "parity_logprob_delta_abs_max": jnp.max(
            jnp.where(mask > 0.0, jnp.abs(delta), 0.0)
        ),
        "parity_old_log_prob_finite": jnp.all(jnp.isfinite(old_log_prob)).astype(
            jnp.float32
        ),
        "parity_new_log_prob_finite": jnp.all(jnp.isfinite(new_log_prob)).astype(
            jnp.float32
        ),
    }


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
        adv_for_mask = advantages
        if advantages.ndim == 1:
            adv_for_mask = advantages[:, None]
        metrics.update(
            {
                "debug/adv_min": jnp.min(advantages),
                "debug/adv_max": jnp.max(advantages),
                "debug/adv_abs_mean": masked_mean(jnp.abs(adv_for_mask), mask),
                "debug/neg_adv_ratio20_objective_mean": masked_mean(
                    jnp.where(adv_for_mask < 0.0, ratio20 * adv_for_mask, 0.0),
                    mask,
                ),
            }
        )
    return metrics
