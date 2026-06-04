"""Learner shielded action sampling for rollout and submission."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.artifacts.checkpoint_compat import (
    is_factorized_pointer_decoder,
    is_planet_flow_pointer_decoder,
)
from src.config import TrainConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.action_codec import (
    PlanetFlowPolicyOutput,
    _factored_step_log_prob_entropy,
    sample_planet_flow_pressure_action,
    source_mask_from_bucket_mask_and_ships,
)
from src.jax.decoder_carry import decoder_carry_enabled
from src.jax.decoders.factorized_topk_pointer import FactorizedDecodeCarry
from src.jax.env import JaxAction
from src.jax.features import TurnBatch
from src.jax.planet_flow import (
    compile_planet_flow_action,
    planet_flow_sampling_target_mask,
)
from src.jax.rollout.types import ShieldedSequenceSample
from src.jax.shield import (
    ShieldDiagnostics,
    apply_configured_trajectory_shield_factorized_topk,
    apply_trajectory_shield_to_turn_batch_v2,
    selected_factored_launch_is_exact_safe_jax,
    trajectory_shield_final_validate_selected,
    trajectory_shield_mode,
)
from src.jax.ship_action import (
    continuous_fraction_log_prob_at_action,
    fraction_from_logit,
    is_continuous_ship_mode,
    ship_count_for_action,
)
from src.opponents.jax_actions.builders import (
    build_action_from_edge_batch,
    build_action_from_factored_batch,
    noop_edge_index,
    owned_planet_ships,
)


def _merge_factorized_decode_carry(
    carry: FactorizedDecodeCarry,
    proposed: FactorizedDecodeCarry,
    active: jax.Array,
) -> FactorizedDecodeCarry:
    """Keep decoder carry fixed for env rows that finished the sub-move sequence."""

    active = active.reshape(active.shape + (1,) * (carry.state.ndim - 1))
    return FactorizedDecodeCarry(
        state=jnp.where(active, proposed.state, carry.state),
        input_emb=jnp.where(active, proposed.input_emb, carry.input_emb),
    )


def _policy_variables(params: dict) -> dict:
    """Accept either Flax variables or a raw params payload."""

    return params if "params" in params else {"params": params}


def _noop_bucket_mask(
    row_count: int, candidate_count: int, bucket_count: int
) -> jax.Array:
    mask = jnp.zeros((row_count, candidate_count, bucket_count), dtype=bool)
    return mask.at[:, 0, 0].set(True)


def _shield_diagnostic_zeros(env_count: int) -> jax.Array:
    return jnp.zeros((env_count,), dtype=jnp.float32)


def _empty_shield_diagnostics(env_count: int) -> ShieldDiagnostics:
    zeros = _shield_diagnostic_zeros(env_count)
    return ShieldDiagnostics(
        blocked_count=zeros,
        blocked_sun_count=zeros,
        blocked_bounds_count=zeros,
        blocked_unintended_hit_count=zeros,
        blocked_horizon_count=zeros,
        fallback_noop_count=zeros,
        legal_non_noop_count=zeros,
        original_non_noop_count=zeros,
        legal_non_noop_rate=zeros,
    )


def _merge_shield_step_diagnostics(
    diagnostics: ShieldDiagnostics,
    step_diagnostics: ShieldDiagnostics,
    *,
    rate_placeholder: jax.Array,
) -> ShieldDiagnostics:
    return ShieldDiagnostics(
        blocked_count=diagnostics.blocked_count + step_diagnostics.blocked_count,
        blocked_sun_count=diagnostics.blocked_sun_count
        + step_diagnostics.blocked_sun_count,
        blocked_bounds_count=diagnostics.blocked_bounds_count
        + step_diagnostics.blocked_bounds_count,
        blocked_unintended_hit_count=diagnostics.blocked_unintended_hit_count
        + step_diagnostics.blocked_unintended_hit_count,
        blocked_horizon_count=diagnostics.blocked_horizon_count
        + step_diagnostics.blocked_horizon_count,
        fallback_noop_count=diagnostics.fallback_noop_count
        + step_diagnostics.fallback_noop_count,
        legal_non_noop_count=diagnostics.legal_non_noop_count
        + step_diagnostics.legal_non_noop_count,
        original_non_noop_count=diagnostics.original_non_noop_count
        + step_diagnostics.original_non_noop_count,
        legal_non_noop_rate=rate_placeholder,
    )


def _finalize_shield_diagnostics(diagnostics: ShieldDiagnostics) -> ShieldDiagnostics:
    return ShieldDiagnostics(
        blocked_count=diagnostics.blocked_count,
        blocked_sun_count=diagnostics.blocked_sun_count,
        blocked_bounds_count=diagnostics.blocked_bounds_count,
        blocked_unintended_hit_count=diagnostics.blocked_unintended_hit_count,
        blocked_horizon_count=diagnostics.blocked_horizon_count,
        fallback_noop_count=diagnostics.fallback_noop_count,
        legal_non_noop_count=diagnostics.legal_non_noop_count,
        original_non_noop_count=diagnostics.original_non_noop_count,
        legal_non_noop_rate=jnp.where(
            diagnostics.original_non_noop_count > 0.0,
            diagnostics.legal_non_noop_count / diagnostics.original_non_noop_count,
            0.0,
        ),
    )


def _ensure_bucket_mask_has_choice(
    ship_bucket_mask: jax.Array,
    flat_decision: jax.Array,
) -> jax.Array:
    noop_mask = _noop_bucket_mask(
        ship_bucket_mask.shape[0], ship_bucket_mask.shape[1], ship_bucket_mask.shape[2]
    )
    has_choice = ship_bucket_mask.any(axis=(1, 2))
    use_original = flat_decision.astype(bool) & has_choice
    return jnp.where(use_original[:, None, None], ship_bucket_mask, noop_mask)


def _sample_step_from_logits(
    *,
    key: jax.Array,
    target_logits: jax.Array,
    ship_logits: jax.Array,
    ship_bucket_mask: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key_target, key_ship = jax.random.split(key)
    illegal_logit = jnp.finfo(jnp.float32).min
    target_mask = ship_bucket_mask.any(axis=-1)
    target_logits = jnp.where(target_mask, target_logits, illegal_logit)
    target = jnp.where(
        deterministic,
        jnp.argmax(target_logits, axis=-1),
        jax.random.categorical(key_target, target_logits, axis=-1),
    )
    selected_bucket_mask = jnp.take_along_axis(
        ship_bucket_mask,
        target[:, None, None].repeat(ship_bucket_mask.shape[-1], axis=-1),
        axis=1,
    ).squeeze(axis=1)
    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target[:, None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=1,
    ).squeeze(axis=1)
    selected_ship_logits = jnp.where(
        selected_bucket_mask, selected_ship_logits, illegal_logit
    )
    bucket = jnp.zeros((target_logits.shape[0],), dtype=jnp.int32)
    continuous_ship = selected_ship_logits.shape[-1] == 1
    if continuous_ship:
        ship_logit = selected_ship_logits[:, 0]
        if deterministic:
            ship_fraction = fraction_from_logit(ship_logit)
        else:
            ship_logit = ship_logit + jax.random.logistic(key_ship, ship_logit.shape)
            ship_fraction = fraction_from_logit(ship_logit)
        policy_loc = selected_ship_logits[:, 0]
        ship_lp = continuous_fraction_log_prob_at_action(policy_loc, ship_fraction)
        ship_entropy = jnp.zeros_like(ship_lp)
        bucket = jnp.where(
            ship_fraction > 0.0, jnp.ones_like(bucket), jnp.zeros_like(bucket)
        )
    else:
        bucket = jnp.where(
            deterministic,
            jnp.argmax(selected_ship_logits, axis=-1),
            jax.random.categorical(key_ship, selected_ship_logits, axis=-1),
        )
        ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
        ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
        ship_lp = jnp.take_along_axis(ship_log_probs, bucket[:, None], axis=-1).squeeze(
            -1
        )
        ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)
        ship_fraction = jnp.zeros_like(ship_lp)

    target_log_probs = jax.nn.log_softmax(target_logits, axis=-1)
    target_probs = jax.nn.softmax(target_logits, axis=-1)
    target_lp = jnp.take_along_axis(target_log_probs, target[:, None], axis=-1).squeeze(
        -1
    )
    entropy = -(target_probs * target_log_probs).sum(axis=-1) - ship_entropy
    return target, bucket, target_lp + ship_lp, entropy, ship_fraction


def _sample_factored_step_from_logits(
    key: jax.Array,
    source_logits: jax.Array,
    target_logits: jax.Array,
    stop_logits: jax.Array,
    ship_logits: jax.Array,
    source_mask: jax.Array,
    ship_bucket_mask: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key_stop, key_source, key_target, key_ship = jax.random.split(key, 4)

    stop_logit = stop_logits
    stop_prob = jax.nn.sigmoid(stop_logit)
    stop = jnp.where(
        deterministic,
        (stop_prob >= 0.5).astype(jnp.int32),
        jax.random.bernoulli(key_stop, stop_prob).astype(jnp.int32),
    )
    continuous_heads = ship_logits.shape[-1] == 1
    has_real_bucket = ship_bucket_mask[..., 1:].any()
    can_launch = source_mask.any() & (has_real_bucket | continuous_heads)
    stop = jnp.where(can_launch, stop, jnp.ones_like(stop))
    launch_active = (1.0 - stop.astype(jnp.float32)) > 0.0

    masked_source_logits = jnp.where(
        source_mask, source_logits, jnp.full_like(source_logits, -1.0e9)
    )
    source = jnp.where(
        launch_active,
        jnp.where(
            deterministic,
            jnp.argmax(masked_source_logits, axis=-1),
            jax.random.categorical(key_source, masked_source_logits, axis=-1),
        ),
        jnp.zeros_like(stop),
    )

    row_bucket_mask = ship_bucket_mask[source]
    target_mask = row_bucket_mask.any(axis=-1)
    has_target = target_mask.any()
    stop = jnp.where(launch_active & (~has_target), jnp.ones_like(stop), stop)
    launch_active = (1.0 - stop.astype(jnp.float32)) > 0.0
    masked_target_logits = jnp.where(
        target_mask, target_logits, jnp.full_like(target_logits, -1.0e9)
    )
    target_slot = jnp.where(
        launch_active,
        jnp.where(
            deterministic,
            jnp.argmax(masked_target_logits, axis=-1),
            jax.random.categorical(key_target, masked_target_logits, axis=-1),
        ),
        jnp.zeros_like(source),
    )

    selected_bucket_mask = row_bucket_mask[target_slot]
    selected_ship_logits = ship_logits[target_slot]
    continuous_ship = selected_ship_logits.shape[-1] == 1
    ship_fraction = jnp.zeros_like(source, dtype=jnp.float32)
    if continuous_ship:
        ship_logit = selected_ship_logits[..., 0]
        selected_target_legal = target_mask[target_slot]
        ship_logit = jnp.where(selected_target_legal, ship_logit, -1.0e9)
        if deterministic:
            ship_fraction = fraction_from_logit(ship_logit)
        else:
            ship_logit = ship_logit + jax.random.logistic(key_ship, ship_logit.shape)
            ship_fraction = fraction_from_logit(ship_logit)
        bucket = jnp.where(
            launch_active & (ship_fraction > 0.0),
            jnp.ones_like(target_slot, dtype=jnp.int32),
            jnp.zeros_like(target_slot, dtype=jnp.int32),
        )
    else:
        selected_ship_logits = jnp.where(
            selected_bucket_mask, selected_ship_logits, -1.0e9
        )
        bucket = jnp.where(
            launch_active,
            jnp.where(
                deterministic,
                jnp.argmax(selected_ship_logits, axis=-1),
                jax.random.categorical(key_ship, selected_ship_logits, axis=-1),
            ),
            jnp.zeros_like(target_slot, dtype=jnp.int32),
        )

    log_prob, entropy, _, _ = _factored_step_log_prob_entropy(
        source_logits[None, :],
        target_logits[None, :],
        stop_logit[None],
        ship_logits[None, :, :],
        source_mask[None, :],
        ship_bucket_mask[None, ...],
        source[None],
        target_slot[None],
        bucket[None],
        stop.astype(jnp.float32)[None],
        ship_fraction=ship_fraction[None] if continuous_ship else None,
    )
    log_prob = log_prob[0]
    entropy = entropy[0]
    source = jnp.where(stop.astype(bool), jnp.zeros_like(source), source)
    target_slot = jnp.where(stop.astype(bool), jnp.zeros_like(target_slot), target_slot)
    bucket = jnp.where(stop.astype(bool), jnp.zeros_like(bucket), bucket)
    ship_fraction = jnp.where(
        stop.astype(bool), jnp.zeros_like(ship_fraction), ship_fraction
    )
    return source, target_slot, bucket, stop, log_prob, entropy, ship_fraction


def _sample_shielded_factored_sequence_with_params(
    key: jax.Array,
    game,
    batch: TurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
    decoder_hidden_in: jax.Array | None = None,
) -> ShieldedSequenceSample:
    from src.jax.factored_sequence_scan import (
        forward_factorized_critic,
        forward_factorized_encode,
        replay_factored_sequence_logprob,
    )
    from src.jax.launch_hygiene import (
        apply_cumulative_forbidden_to_shield,
        apply_launch_to_cumulative_forbidden,
        build_hygiene_lookups,
        empty_forbidden_grid,
    )
    from src.jax.policy import (
        factorized_decode,
        factorized_decode_advance_carry,
        factorized_decode_init_carry,
        factorized_decode_step,
    )

    env_count = batch.planet_features.shape[0]
    player_count = jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32)
    carry_enabled = decoder_carry_enabled(cfg)
    continuous = is_continuous_ship_mode(cfg)
    encoder_out = forward_factorized_encode(params, policy, batch)
    value_out = forward_factorized_critic(
        params, policy, encoder_out, player_count=player_count
    )
    sequence_k = cfg.model.max_moves_k
    k = edge_k(cfg.task)
    source_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    slot_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    bucket_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    stop_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    step_mask_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    log_prob_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    entropy_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    ship_fraction_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    decoder_hidden_carry = decoder_hidden_in if carry_enabled else None
    decode_carry = factorized_decode_init_carry(
        params,
        policy,
        encoder_out,
        decoder_hidden=decoder_hidden_carry,
    )
    remaining_ships = owned_planet_ships(game)
    hygiene_lookups = build_hygiene_lookups(batch)
    cumulative_forbidden = empty_forbidden_grid(
        env_count,
        num_planets=MAX_PLANETS,
        max_k=k,
        buckets=cfg.task.ship_bucket_count,
    )
    diagnostic_zero = _shield_diagnostic_zeros(env_count)
    diagnostics = _empty_shield_diagnostics(env_count)
    bucket_mask_stack = jnp.zeros(
        (env_count, sequence_k, MAX_PLANETS, k, cfg.task.ship_bucket_count),
        dtype=jnp.bool_,
    )
    sequence_active = jnp.ones((env_count,), dtype=bool)

    def sequence_scan_body(carry, step_idx):
        (
            source_sequence,
            slot_sequence,
            bucket_sequence,
            stop_sequence,
            step_mask_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
            sequence_active,
            decode_carry,
            cumulative_forbidden,
        ) = carry
        step_logits, proposed_decode_carry = factorized_decode_step(
            params,
            policy,
            encoder_out,
            decode_carry,
            teacher_source=source_sequence[:, step_idx],
            teacher_target_slot=slot_sequence[:, step_idx],
            rng=jax.random.fold_in(key, step_idx),
            deterministic=deterministic,
        )
        decode_carry = _merge_factorized_decode_carry(
            decode_carry, proposed_decode_carry, sequence_active
        )
        shielded = jax.vmap(
            lambda game_row, batch_row, ships: (
                apply_configured_trajectory_shield_factorized_topk(
                    game_row, batch_row, cfg.task, remaining_planet_ships=ships
                )
            )
        )(game, batch, remaining_ships)
        diagnostics = _merge_shield_step_diagnostics(
            diagnostics,
            shielded.diagnostics,
            rate_placeholder=diagnostic_zero,
        )
        shield_step_mask = shielded.ship_bucket_mask
        step_bucket_mask = apply_cumulative_forbidden_to_shield(
            shield_step_mask,
            cumulative_forbidden,
        )
        source_mask = jax.vmap(source_mask_from_bucket_mask_and_ships, in_axes=(0, 0))(
            step_bucket_mask, remaining_ships
        )
        source, target_slot, bucket, stop, log_prob, entropy, ship_fraction = jax.vmap(
            _sample_factored_step_from_logits,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None),
        )(
            jax.random.split(jax.random.fold_in(key, 10_000 + step_idx), env_count),
            step_logits.source_logits,
            step_logits.target_logits,
            step_logits.stop_logits,
            step_logits.ship_logits,
            source_mask,
            step_bucket_mask,
            deterministic,
        )
        step_active = sequence_active.astype(jnp.float32)
        stop = jnp.where(sequence_active, stop, jnp.zeros_like(stop))
        log_prob = jnp.where(sequence_active, log_prob, jnp.zeros_like(log_prob))
        entropy = jnp.where(sequence_active, entropy, jnp.zeros_like(entropy))
        ship_fraction = jnp.where(
            sequence_active, ship_fraction, jnp.zeros_like(ship_fraction)
        )
        src_rows = source
        current_source_ships = remaining_ships[jnp.arange(env_count), src_rows]
        launched = ship_count_for_action(
            current_source_ships,
            bucket,
            ship_fraction if continuous else None,
            cfg,
        )
        launch_valid = (
            sequence_active
            & jnp.logical_not(stop.astype(bool))
            & (launched > 0.0)
            & jnp.where(continuous, ship_fraction > 0.0, bucket > 0)
        )

        # Tiered mode: cheap mask for sampling, exact check only the sampled launch.
        # Rejected launches are converted into stop/no-op so replay log-prob is
        # recomputed against the final stored sequence below.
        if trajectory_shield_mode(
            cfg.task
        ) == "tiered" and trajectory_shield_final_validate_selected(cfg.task):
            exact_safe = jax.vmap(
                lambda game_row, batch_row, src, slot, ships, stop_value, active: (
                    selected_factored_launch_is_exact_safe_jax(
                        game_row,
                        batch_row,
                        cfg.task,
                        src,
                        slot,
                        ships,
                        stop_value,
                        active,
                    )
                )
            )(
                game,
                batch,
                source,
                target_slot,
                launched,
                stop,
                sequence_active,
            )
            reject_launch = launch_valid & (~exact_safe)

            stop = jnp.where(reject_launch, jnp.ones_like(stop), stop)
            bucket = jnp.where(reject_launch, jnp.zeros_like(bucket), bucket)
            ship_fraction = jnp.where(
                reject_launch,
                jnp.zeros_like(ship_fraction),
                ship_fraction,
            )
            launched = jnp.where(reject_launch, 0.0, launched)
            launch_valid = launch_valid & exact_safe

        # Tiered exact reject runs before hygiene carry update: rejected launches
        # must not mark (source, slot) forbidden (stored stop=1 / bucket=0).
        cumulative_forbidden = apply_launch_to_cumulative_forbidden(
            cumulative_forbidden,
            batch=batch,
            lookups=hygiene_lookups,
            src_row=source,
            slot=target_slot,
            active=launch_valid,
        )
        remaining_ships = remaining_ships.at[jnp.arange(env_count), src_rows].set(
            jnp.where(
                launch_valid,
                jnp.maximum(current_source_ships - launched, 0.0),
                current_source_ships,
            )
        )
        source_sequence = source_sequence.at[:, step_idx].set(source)
        slot_sequence = slot_sequence.at[:, step_idx].set(target_slot)
        bucket_sequence = bucket_sequence.at[:, step_idx].set(bucket)
        stop_sequence = stop_sequence.at[:, step_idx].set(stop)
        step_mask_sequence = step_mask_sequence.at[:, step_idx].set(step_active)
        log_prob_sequence = log_prob_sequence.at[:, step_idx].set(log_prob)
        entropy_sequence = entropy_sequence.at[:, step_idx].set(entropy)
        ship_fraction_sequence = ship_fraction_sequence.at[:, step_idx].set(
            ship_fraction
        )
        bucket_mask_stack = bucket_mask_stack.at[:, step_idx].set(shield_step_mask)
        sequence_active = sequence_active & jnp.logical_not(stop.astype(bool))
        proposed_decode_carry = factorized_decode_advance_carry(
            params,
            policy,
            encoder_out,
            decode_carry,
            source=source,
            target_slot=target_slot,
        )
        decode_carry = _merge_factorized_decode_carry(
            decode_carry, proposed_decode_carry, sequence_active
        )
        return (
            source_sequence,
            slot_sequence,
            bucket_sequence,
            stop_sequence,
            step_mask_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
            sequence_active,
            decode_carry,
            cumulative_forbidden,
        ), None

    (
        (
            source_sequence,
            slot_sequence,
            bucket_sequence,
            stop_sequence,
            step_mask_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            _remaining_ships,
            diagnostics,
            bucket_mask_stack,
            _sequence_active,
            decode_carry,
            _cumulative_forbidden,
        ),
        _,
    ) = jax.lax.scan(
        sequence_scan_body,
        (
            source_sequence,
            slot_sequence,
            bucket_sequence,
            stop_sequence,
            step_mask_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
            sequence_active,
            decode_carry,
            cumulative_forbidden,
        ),
        jnp.arange(sequence_k, dtype=jnp.int32),
    )
    diagnostics = _finalize_shield_diagnostics(diagnostics)
    noop_idx = noop_edge_index(cfg.task)
    target_sequence = source_sequence * k + slot_sequence
    target_sequence = jnp.where(
        stop_sequence.astype(bool) | jnp.logical_not(step_mask_sequence.astype(bool)),
        noop_idx,
        target_sequence,
    )
    if carry_enabled:
        final_output = factorized_decode(
            params,
            policy,
            encoder_out,
            player_count=player_count,
            source_sequence=source_sequence,
            target_slot_sequence=slot_sequence,
            decoder_hidden=decoder_hidden_in,
            deterministic=deterministic,
            include_value=False,
        )
        decoder_hidden_out = final_output.decoder_hidden
    else:
        decoder_hidden_out = None
    tiered_revalidate = trajectory_shield_mode(
        cfg.task
    ) == "tiered" and trajectory_shield_final_validate_selected(cfg.task)
    if tiered_revalidate:
        replay = replay_factored_sequence_logprob(
            params,
            policy,
            batch,
            cfg,
            player_count=player_count,
            source_index=source_sequence,
            target_slot=slot_sequence,
            ship_bucket=bucket_sequence,
            stop_flag=stop_sequence.astype(jnp.float32),
            step_mask=step_mask_sequence,
            ship_bucket_mask=bucket_mask_stack,
            ship_fraction=ship_fraction_sequence if continuous else None,
            decoder_hidden=decoder_hidden_in if carry_enabled else None,
            initial_remaining_ships=owned_planet_ships(game),
            encoder_out=encoder_out,
        )
        log_prob_sequence = replay.log_prob
        entropy_sequence = replay.entropy
    return ShieldedSequenceSample(
        target_index=target_sequence,
        ship_bucket=bucket_sequence,
        log_prob=log_prob_sequence,
        entropy=entropy_sequence,
        value=value_out.value,
        ship_bucket_mask=bucket_mask_stack,
        diagnostics=diagnostics,
        source_index=source_sequence,
        target_slot=slot_sequence,
        stop_flag=stop_sequence,
        step_mask=step_mask_sequence,
        decoder_hidden_out=decoder_hidden_out if carry_enabled else None,
        ship_fraction=ship_fraction_sequence if continuous else None,
    )


def _sample_shielded_sequence_with_params(
    key: jax.Array,
    game,
    batch: TurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
    decoder_hidden_in: jax.Array | None = None,
) -> ShieldedSequenceSample:
    if is_factorized_pointer_decoder(cfg.model):
        return _sample_shielded_factored_sequence_with_params(
            key,
            game,
            batch,
            params,
            policy,
            cfg,
            deterministic=deterministic,
            decoder_hidden_in=decoder_hidden_in,
        )

    env_count = batch.planet_features.shape[0]
    player_count = jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32)
    carry_enabled = decoder_carry_enabled(cfg)
    continuous = is_continuous_ship_mode(cfg)
    probe_kwargs = {
        "player_count": player_count,
        "rng": key,
        "deterministic": deterministic,
    }
    if carry_enabled:
        probe_kwargs["decoder_hidden"] = decoder_hidden_in
    probe_output = policy.apply(params, batch, **probe_kwargs)
    sequence_k = probe_output.target_logits.shape[1]
    edge_count = probe_output.target_logits.shape[2]
    noop_idx = noop_edge_index(cfg.task)
    target_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    bucket_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    log_prob_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    entropy_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    ship_fraction_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    decoder_hidden_carry = decoder_hidden_in if carry_enabled else None
    remaining_ships = owned_planet_ships(game)
    diagnostic_zero = _shield_diagnostic_zeros(env_count)
    diagnostics = _empty_shield_diagnostics(env_count)
    bucket_mask_stack = jnp.zeros(
        (env_count, sequence_k, edge_count, cfg.task.ship_bucket_count),
        dtype=jnp.bool_,
    )

    def sequence_scan_body(carry, step_idx):
        (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
            decoder_hidden_carry,
        ) = carry
        step_kwargs = {
            "player_count": player_count,
            "target_sequence": target_sequence,
            "rng": jax.random.fold_in(key, step_idx),
            "deterministic": deterministic,
        }
        if carry_enabled:
            step_kwargs["decoder_hidden"] = decoder_hidden_carry
        step_output = policy.apply(params, batch, **step_kwargs)
        shielded = jax.vmap(
            lambda game_row, batch_row, ships: apply_trajectory_shield_to_turn_batch_v2(
                game_row, batch_row, cfg.task, remaining_planet_ships=ships
            )
        )(game, batch, remaining_ships)
        diagnostics = _merge_shield_step_diagnostics(
            diagnostics,
            shielded.diagnostics,
            rate_placeholder=diagnostic_zero,
        )
        edge_action_mask = jnp.concatenate(
            [
                shielded.batch.edge_mask.reshape(
                    env_count, MAX_PLANETS * edge_k(cfg.task)
                ),
                jnp.ones((env_count, 1), dtype=bool),
            ],
            axis=1,
        )
        step_bucket_mask = shielded.ship_bucket_mask.reshape(
            env_count, edge_count, cfg.task.ship_bucket_count
        )
        env_active = jnp.ones((env_count,), dtype=bool)
        step_bucket_mask = _ensure_bucket_mask_has_choice(
            step_bucket_mask.reshape(-1, edge_count, cfg.task.ship_bucket_count),
            env_active,
        )
        step_bucket_mask = step_bucket_mask.reshape(
            env_count, edge_count, cfg.task.ship_bucket_count
        )
        target, bucket, log_prob, entropy, ship_fraction = _sample_step_from_logits(
            key=jax.random.fold_in(key, 10_000 + step_idx),
            target_logits=step_output.target_logits[:, step_idx, :],
            ship_logits=step_output.ship_logits[:, step_idx, :, :],
            ship_bucket_mask=step_bucket_mask,
            deterministic=deterministic,
        )
        src_rows = target // edge_k(cfg.task)
        current_source_ships = remaining_ships[jnp.arange(env_count), src_rows]
        launched = ship_count_for_action(
            current_source_ships,
            bucket,
            ship_fraction if continuous else None,
            cfg,
        )
        launch_valid = (
            (target < noop_idx)
            & (launched > 0.0)
            & jnp.where(continuous, ship_fraction > 0.0, bucket > 0)
        )
        remaining_ships = remaining_ships.at[jnp.arange(env_count), src_rows].set(
            jnp.where(
                launch_valid,
                jnp.maximum(current_source_ships - launched, 0.0),
                current_source_ships,
            )
        )
        target_sequence = target_sequence.at[:, step_idx].set(target)
        bucket_sequence = bucket_sequence.at[:, step_idx].set(bucket)
        log_prob_sequence = log_prob_sequence.at[:, step_idx].set(log_prob)
        entropy_sequence = entropy_sequence.at[:, step_idx].set(entropy)
        ship_fraction_sequence = ship_fraction_sequence.at[:, step_idx].set(
            ship_fraction
        )
        bucket_mask_stack = bucket_mask_stack.at[:, step_idx].set(step_bucket_mask)
        return (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
            decoder_hidden_carry,
        ), None

    (
        (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            _remaining_ships,
            diagnostics,
            bucket_mask_stack,
            decoder_hidden_out,
        ),
        _,
    ) = jax.lax.scan(
        sequence_scan_body,
        (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            ship_fraction_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
            decoder_hidden_carry,
        ),
        jnp.arange(sequence_k, dtype=jnp.int32),
    )
    diagnostics = _finalize_shield_diagnostics(diagnostics)
    k = edge_k(cfg.task)
    if carry_enabled:
        final_output = policy.apply(
            params,
            batch,
            player_count=player_count,
            target_sequence=target_sequence,
            decoder_hidden=decoder_hidden_in,
            deterministic=deterministic,
        )
        decoder_hidden_out = final_output.decoder_hidden
    return ShieldedSequenceSample(
        target_index=target_sequence,
        ship_bucket=bucket_sequence,
        log_prob=log_prob_sequence,
        entropy=entropy_sequence,
        value=probe_output.value,
        ship_bucket_mask=bucket_mask_stack,
        diagnostics=diagnostics,
        source_index=target_sequence // k,
        target_slot=target_sequence % k,
        stop_flag=jnp.zeros_like(target_sequence),
        step_mask=jnp.ones_like(log_prob_sequence),
        decoder_hidden_out=decoder_hidden_out if carry_enabled else None,
        ship_fraction=ship_fraction_sequence if continuous else None,
    )


def _sample_policy_action_with_params(
    key: jax.Array,
    game,
    batch: TurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
    decoder_hidden_in: jax.Array | None = None,
) -> tuple[JaxAction, jax.Array | None]:
    if is_planet_flow_pointer_decoder(cfg.model):
        player_count = jnp.full(
            (batch.planet_mask.shape[0],), cfg.task.player_count, dtype=jnp.int32
        )
        output = policy.apply(
            _policy_variables(params),
            batch,
            player_count=player_count,
        )
        if not isinstance(output, PlanetFlowPolicyOutput):
            raise TypeError(
                "planet_flow_target_heatmap policy must return PlanetFlowPolicyOutput."
            )
        pressure_action = sample_planet_flow_pressure_action(
            key,
            output,
            jnp.asarray(
                cfg.model.planet_flow.pressure_bucket_values, dtype=jnp.float32
            ),
            planet_flow_sampling_target_mask(game, batch),
            deterministic=deterministic,
        )
        compile_result = compile_planet_flow_action(
            game,
            batch,
            pressure_action.target_pressure,
            cfg,
        )
        return compile_result.action, decoder_hidden_in

    sample = _sample_shielded_sequence_with_params(
        key,
        game,
        batch,
        params,
        policy,
        cfg,
        deterministic=deterministic,
        decoder_hidden_in=decoder_hidden_in,
    )
    if is_factorized_pointer_decoder(cfg.model):
        action = build_action_from_factored_batch(
            game,
            batch,
            sample.source_index,
            sample.target_slot,
            sample.ship_bucket,
            sample.stop_flag,
            sample.step_mask,
            cfg,
            ship_fraction=sample.ship_fraction,
        )
    else:
        action = build_action_from_edge_batch(
            game, batch, sample.target_index, sample.ship_bucket, cfg
        )
    return action, sample.decoder_hidden_out


def _sample_policy_action(
    key: jax.Array,
    game,
    batch: TurnBatch,
    train_state,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> tuple[JaxAction, jax.Array | None]:
    return _sample_policy_action_with_params(
        key,
        game,
        batch,
        train_state.params,
        policy,
        cfg,
        deterministic=deterministic,
    )
