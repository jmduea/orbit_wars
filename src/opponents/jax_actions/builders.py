from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.features.registry import candidate_feature_schema
from src.game.trajectory_shield import (
    ShieldDiagnostics,
    apply_trajectory_shield_to_turn_batch,
    default_ship_bucket_mask,
)
from src.jax.env import JaxAction
from src.jax.features import JaxTurnBatch
from src.jax.ppo_update import flatten_batch
from src.jax.rollout.types import JaxTrainState, ShieldedSequenceSample


def ship_count_for_bucket_jax(
    available_ships: jax.Array, bucket: jax.Array, bucket_count: int
) -> jax.Array:
    """Convert discrete ship buckets into concrete launched ship counts."""

    fraction = jnp.where(
        bucket <= 0, 0.0, bucket.astype(jnp.float32) / float(max(bucket_count - 1, 1))
    )
    ships = jnp.ceil(available_ships * fraction)
    ships = jnp.minimum(available_ships, jnp.maximum(1.0, ships))
    return jnp.where((available_ships <= 0.0) | (fraction <= 0.0), 0.0, ships)


def build_action_from_batch(
    batch: JaxTurnBatch,
    target_index: jax.Array,
    ship_bucket: jax.Array,
    cfg: TrainConfig,
) -> JaxAction:
    """Build fixed-size JAX action buffers from per-source policy choices.

    Only valid source rows with non-no-op targets and positive ship buckets are
    emitted. If ``max_fleets`` is smaller than ``max_planets``, extra source rows
    are clipped so the returned arrays always match the configured fleet buffer.
    """

    env_count = batch.self_features.shape[0]
    planet_count = batch.self_features.shape[1]
    target_index = target_index.reshape(env_count, planet_count, -1)
    ship_bucket = ship_bucket.reshape(env_count, planet_count, -1)
    launch_steps = target_index.shape[-1]
    chosen_mask = jnp.take_along_axis(
        batch.candidate_mask[..., None, :], target_index[..., None], axis=-1
    ).squeeze(-1)
    chosen_angle = jnp.take_along_axis(
        batch.target_angles[..., None, :], target_index[..., None], axis=-1
    ).squeeze(-1)
    step_valid = (
        batch.decision_mask[..., None]
        & chosen_mask
        & (target_index > 0)
        & (ship_bucket > 0)
    )

    def allocate_step(remaining_ships, step_inputs):
        step_bucket, step_is_valid = step_inputs
        requested = ship_count_for_bucket_jax(
            remaining_ships, step_bucket, cfg.task.ship_bucket_count
        )
        launched = jnp.where(
            step_is_valid, jnp.minimum(remaining_ships, requested), 0.0
        )
        return jnp.maximum(remaining_ships - launched, 0.0), launched

    _remaining_ships, ships_by_step = jax.lax.scan(
        allocate_step,
        batch.source_ships,
        (jnp.moveaxis(ship_bucket, -1, 0), jnp.moveaxis(step_valid, -1, 0)),
    )
    ships = jnp.moveaxis(ships_by_step, 0, -1)
    valid = step_valid & (ships > 0.0)
    fleet_slots = cfg.task.max_fleets
    action_width = min(planet_count * launch_steps, fleet_slots)
    pad = fleet_slots - action_width
    source_ids = jnp.broadcast_to(
        batch.source_ids[..., None], (env_count, planet_count, launch_steps)
    )
    source_id = jnp.pad(
        source_ids.reshape(env_count, planet_count * launch_steps)[:, :action_width],
        ((0, 0), (0, pad)),
        constant_values=-1,
    )
    angle = jnp.pad(
        chosen_angle.reshape(env_count, planet_count * launch_steps)[:, :action_width],
        ((0, 0), (0, pad)),
        constant_values=0.0,
    )
    ships = jnp.pad(
        ships.reshape(env_count, planet_count * launch_steps)[:, :action_width],
        ((0, 0), (0, pad)),
        constant_values=0.0,
    )
    valid = jnp.pad(
        valid.reshape(env_count, planet_count * launch_steps)[:, :action_width],
        ((0, 0), (0, pad)),
        constant_values=False,
    )
    return JaxAction(source_id=source_id, angle=angle, ships=ships, valid=valid)


def build_random_action_from_batch(
    key: jax.Array,
    batch: JaxTurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    """Sample a JAX-native random opponent action for each environment."""

    env_count = batch.self_features.shape[0]
    planet_count = batch.self_features.shape[1]
    key_target, key_bucket = jax.random.split(key)
    flat_mask = batch.candidate_mask.reshape(-1, cfg.task.candidate_count)
    flat_bucket_mask = (
        default_ship_bucket_mask(flat_mask, cfg.task.ship_bucket_count)
        if ship_bucket_mask is None
        else ship_bucket_mask.reshape(
            -1, cfg.task.candidate_count, cfg.task.ship_bucket_count
        )
    )
    real_bucket_mask = flat_bucket_mask & (
        jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)[None, None, :] > 0
    )
    real_candidate = (
        flat_mask
        & real_bucket_mask.any(axis=-1)
        & (jnp.arange(cfg.task.candidate_count, dtype=jnp.int32)[None, :] > 0)
    )
    has_target = real_candidate.any(axis=-1)
    target_logits = jnp.where(real_candidate, 0.0, jnp.finfo(jnp.float32).min)
    target = jnp.where(
        has_target,
        jax.random.categorical(key_target, target_logits, axis=-1),
        jnp.zeros((env_count * planet_count,), dtype=jnp.int32),
    )
    selected_bucket_mask = jnp.take_along_axis(
        flat_bucket_mask,
        target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_logits = jnp.where(selected_bucket_mask, 0.0, jnp.finfo(jnp.float32).min)
    bucket = jax.random.categorical(key_bucket, bucket_logits, axis=-1)
    bucket = jnp.where(has_target, bucket, jnp.zeros_like(bucket))
    return build_action_from_batch(batch, target, bucket, cfg)


def build_sniper_action_from_batch(
    batch: JaxTurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    """JAX-compatible scripted sniper: use nearest candidate slot aggressively."""

    bucket_mask = (
        default_ship_bucket_mask(batch.candidate_mask, cfg.task.ship_bucket_count)
        if ship_bucket_mask is None
        else ship_bucket_mask
    )
    nonzero_bucket_mask = bucket_mask[..., 1:].any(axis=-1)
    real_candidate_mask = (
        batch.candidate_mask
        & nonzero_bucket_mask
        & (
            jnp.arange(batch.candidate_mask.shape[-1], dtype=jnp.int32)[None, None, :]
            > 0
        )
    )
    nearest_slot = jnp.argmax(real_candidate_mask.astype(jnp.int32), axis=-1)
    has_target = real_candidate_mask.any(axis=-1)
    target = jnp.where(has_target, nearest_slot, 0).reshape(-1)
    selected_bucket_mask = jnp.take_along_axis(
        bucket_mask.reshape(-1, cfg.task.candidate_count, cfg.task.ship_bucket_count),
        target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_ids = jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)
    bucket = jnp.max(jnp.where(selected_bucket_mask, bucket_ids[None, :], 0), axis=-1)
    bucket = jnp.where(has_target.reshape(-1), bucket, 0)
    return build_action_from_batch(batch, target, bucket, cfg)


def build_turtle_action_from_batch(
    batch: JaxTurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    """Conservative scripted policy: small neutral expansion, otherwise no-op."""

    bucket_mask = (
        default_ship_bucket_mask(batch.candidate_mask, cfg.task.ship_bucket_count)
        if ship_bucket_mask is None
        else ship_bucket_mask
    )
    ownership = batch.candidate_features[
        ..., candidate_feature_schema(cfg.task).slice("target_ownership_flags")
    ]
    neutral = ownership[..., 0] > 0.5
    real_bucket = bucket_mask[..., 1:].any(axis=-1)
    valid_neutral = (
        batch.candidate_mask
        & neutral
        & real_bucket
        & (
            jnp.arange(batch.candidate_mask.shape[-1], dtype=jnp.int32)[None, None, :]
            > 0
        )
    )
    target = jnp.argmax(valid_neutral.astype(jnp.int32), axis=-1)
    has_target = valid_neutral.any(axis=-1)
    target = jnp.where(has_target, target, 0).reshape(-1)
    selected_bucket_mask = jnp.take_along_axis(
        bucket_mask.reshape(-1, cfg.task.candidate_count, cfg.task.ship_bucket_count),
        target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_ids = jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)
    nonzero_bucket_mask = selected_bucket_mask & (bucket_ids[None, :] > 0)
    bucket = jnp.argmax(nonzero_bucket_mask.astype(jnp.int32), axis=-1)
    bucket = jnp.where(
        has_target.reshape(-1) & nonzero_bucket_mask.any(axis=-1), bucket, 0
    )
    return build_action_from_batch(batch, target, bucket, cfg)


def build_opportunistic_action_from_batch(
    batch: JaxTurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    """Scripted policy that prioritizes immediately attackable enemy candidates."""

    bucket_mask = (
        default_ship_bucket_mask(batch.candidate_mask, cfg.task.ship_bucket_count)
        if ship_bucket_mask is None
        else ship_bucket_mask
    )
    ownership = batch.candidate_features[
        ..., candidate_feature_schema(cfg.task).slice("target_ownership_flags")
    ]
    enemy = ownership[..., 2] > 0.5
    real_bucket = bucket_mask[..., 1:].any(axis=-1)
    valid_enemy = (
        batch.candidate_mask
        & enemy
        & real_bucket
        & (
            jnp.arange(batch.candidate_mask.shape[-1], dtype=jnp.int32)[None, None, :]
            > 0
        )
    )
    target = jnp.argmax(valid_enemy.astype(jnp.int32), axis=-1)
    has_target = valid_enemy.any(axis=-1)
    target = jnp.where(has_target, target, 0).reshape(-1)
    selected_bucket_mask = jnp.take_along_axis(
        bucket_mask.reshape(-1, cfg.task.candidate_count, cfg.task.ship_bucket_count),
        target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_ids = jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)
    bucket = jnp.max(jnp.where(selected_bucket_mask, bucket_ids[None, :], 0), axis=-1)
    bucket = jnp.where(has_target.reshape(-1), bucket, 0)
    return build_action_from_batch(batch, target, bucket, cfg)


def build_noop_action_from_batch(batch: JaxTurnBatch, cfg: TrainConfig) -> JaxAction:
    """Build a pass/no-op action that launches no fleets."""

    env_count = batch.self_features.shape[0]
    planet_count = batch.self_features.shape[1]
    zeros = jnp.zeros((env_count * planet_count,), dtype=jnp.int32)
    return build_action_from_batch(batch, zeros, zeros, cfg)


def _noop_bucket_mask(
    row_count: int, candidate_count: int, bucket_count: int
) -> jax.Array:
    mask = jnp.zeros((row_count, candidate_count, bucket_count), dtype=bool)
    return mask.at[:, 0, 0].set(True)


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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
    bucket = jnp.where(
        deterministic,
        jnp.argmax(selected_ship_logits, axis=-1),
        jax.random.categorical(key_ship, selected_ship_logits, axis=-1),
    )

    target_log_probs = jax.nn.log_softmax(target_logits, axis=-1)
    target_probs = jax.nn.softmax(target_logits, axis=-1)
    target_lp = jnp.take_along_axis(target_log_probs, target[:, None], axis=-1).squeeze(
        -1
    )
    ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
    ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
    ship_lp = jnp.take_along_axis(ship_log_probs, bucket[:, None], axis=-1).squeeze(-1)
    entropy = -(target_probs * target_log_probs).sum(axis=-1) - (
        ship_probs * ship_log_probs
    ).sum(axis=-1)
    return target, bucket, target_lp + ship_lp, entropy


def _sample_shielded_sequence_with_params(
    key: jax.Array,
    game,
    batch: JaxTurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> ShieldedSequenceSample:
    flat_self, flat_candidate, flat_global, flat_mask, flat_decision = flatten_batch(
        batch
    )
    flat_player_count = jnp.full(
        (flat_mask.shape[0],), cfg.task.player_count, dtype=jnp.int32
    )
    probe_output = policy.apply(
        params,
        flat_self,
        flat_candidate,
        flat_global,
        flat_mask,
        player_count=flat_player_count,
        rng=key,
        deterministic=deterministic,
    )
    sequence_k = probe_output.target_logits.shape[1]
    row_count = flat_mask.shape[0]
    target_sequence = jnp.zeros((row_count, sequence_k), dtype=jnp.int32)
    bucket_sequence = jnp.zeros((row_count, sequence_k), dtype=jnp.int32)
    log_prob_sequence = jnp.zeros((row_count, sequence_k), dtype=jnp.float32)
    entropy_sequence = jnp.zeros((row_count, sequence_k), dtype=jnp.float32)
    remaining_ships = batch.source_ships.reshape(-1)
    env_count = batch.source_ships.shape[0]
    diagnostic_zero = jnp.zeros((env_count,), dtype=jnp.float32)
    diagnostics = ShieldDiagnostics(
        blocked_count=diagnostic_zero,
        blocked_sun_count=diagnostic_zero,
        blocked_bounds_count=diagnostic_zero,
        blocked_unintended_hit_count=diagnostic_zero,
        blocked_horizon_count=diagnostic_zero,
        fallback_noop_count=diagnostic_zero,
        legal_non_noop_count=diagnostic_zero,
        original_non_noop_count=diagnostic_zero,
        legal_non_noop_rate=diagnostic_zero,
    )

    bucket_mask_stack = jnp.zeros(
        (row_count, sequence_k, cfg.task.candidate_count, cfg.task.ship_bucket_count),
        dtype=jnp.bool_,
    )

    def sequence_scan_body(carry, step_idx):
        (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
        ) = carry
        step_output = policy.apply(
            params,
            flat_self,
            flat_candidate,
            flat_global,
            flat_mask,
            player_count=flat_player_count,
            target_sequence=target_sequence,
            rng=jax.random.fold_in(key, step_idx),
            deterministic=deterministic,
        )
        remaining_by_source = remaining_ships.reshape(batch.source_ships.shape)
        shielded_step = jax.vmap(
            lambda game_row, turn_row, source_ships: (
                apply_trajectory_shield_to_turn_batch(
                    game_row, turn_row, cfg.task, source_ships_override=source_ships
                )
            )
        )(game, batch, remaining_by_source)
        step_diagnostics = shielded_step.diagnostics
        diagnostics = ShieldDiagnostics(
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
            legal_non_noop_rate=diagnostic_zero,
        )
        step_bucket_mask = shielded_step.ship_bucket_mask.reshape(
            -1, cfg.task.candidate_count, cfg.task.ship_bucket_count
        )
        step_bucket_mask = _ensure_bucket_mask_has_choice(
            step_bucket_mask, flat_decision
        )
        target, bucket, log_prob, entropy = _sample_step_from_logits(
            key=jax.random.fold_in(key, 10_000 + step_idx),
            target_logits=step_output.target_logits[:, step_idx, :],
            ship_logits=step_output.ship_logits[:, step_idx, :, :],
            ship_bucket_mask=step_bucket_mask,
            deterministic=deterministic,
        )
        target = jnp.where(flat_decision, target, 0)
        bucket = jnp.where(flat_decision, bucket, 0)
        launched = ship_count_for_bucket_jax(
            remaining_ships, bucket, cfg.task.ship_bucket_count
        )
        launch_valid = flat_decision & (target > 0) & (bucket > 0) & (launched > 0.0)
        remaining_ships = jnp.where(
            launch_valid, jnp.maximum(remaining_ships - launched, 0.0), remaining_ships
        )
        target_sequence = target_sequence.at[:, step_idx].set(target)
        bucket_sequence = bucket_sequence.at[:, step_idx].set(bucket)
        log_prob_sequence = log_prob_sequence.at[:, step_idx].set(log_prob)
        entropy_sequence = entropy_sequence.at[:, step_idx].set(entropy)
        bucket_mask_stack = bucket_mask_stack.at[:, step_idx].set(step_bucket_mask)
        return (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
        ), None

    (
        (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
        ),
        _,
    ) = jax.lax.scan(
        sequence_scan_body,
        (
            target_sequence,
            bucket_sequence,
            log_prob_sequence,
            entropy_sequence,
            remaining_ships,
            diagnostics,
            bucket_mask_stack,
        ),
        jnp.arange(sequence_k, dtype=jnp.int32),
    )

    diagnostics = ShieldDiagnostics(
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
    return ShieldedSequenceSample(
        target_index=target_sequence,
        ship_bucket=bucket_sequence,
        log_prob=log_prob_sequence,
        entropy=entropy_sequence,
        value=probe_output.value,
        ship_bucket_mask=bucket_mask_stack,
        diagnostics=diagnostics,
    )


def _sample_policy_action_with_params(
    key: jax.Array,
    game,
    batch: JaxTurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    """Sample a fixed-size action buffer from a JAX policy parameter set."""

    sample = _sample_shielded_sequence_with_params(
        key,
        game,
        batch,
        params,
        policy,
        cfg,
        deterministic=deterministic,
    )
    return build_action_from_batch(batch, sample.target_index, sample.ship_bucket, cfg)


def _sample_policy_action(
    key: jax.Array,
    game,
    batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    """Sample a fixed-size action buffer from the trainable JAX policy."""

    return _sample_policy_action_with_params(
        key,
        game,
        batch,
        train_state.params,
        policy,
        cfg,
        deterministic=deterministic,
    )
