from __future__ import annotations

from typing import NamedTuple

import flax
import flax.struct
import jax.numpy as jnp
import optax

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import (
    candidate_feature_dim,
    candidate_feature_schema,
    global_feature_dim,
    global_feature_schema,
    self_feature_dim,
    self_feature_schema,
)
from src.game.constants import MAX_FLEET_SPEED, MAX_PLANETS, MAX_PRODUCTION
from src.game.trajectory_shield import (
    ShieldDiagnostics,
    apply_trajectory_shield_to_turn_batch,
    default_ship_bucket_mask,
    mask_policy_output_for_shield,
)
from src.opponents.pool import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_NEAREST_SNIPER,
    OPPONENT_NOOP,
    OPPONENT_OPPORTUNISTIC,
    OPPONENT_RANDOM,
    OPPONENT_TURTLE,
    sample_opponent_type_ids_jax,
)
from src.training.curriculum import StageView, default_stage_view

from .env import (
    JaxAction,
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
    batched_step_multi_player,
    fleet_speed,
)
from .features import JaxTurnBatch, encode_turn
from .policy import action_log_prob_and_entropy


def validate_policy_param_shapes(params: dict, env_cfg: TaskConfig) -> None:
    """Validate encoder input dimensions in Flax params against env features.

    Checks the first Dense kernel for the self/candidate/global encoder MLPs.
    Raises ValueError with expected/actual dimensions and remediation guidance
    when params are incompatible with the active environment configuration.
    """

    if not isinstance(params, dict):
        raise ValueError(
            "Policy params must be a Flax parameter dict. Received "
            f"{type(params).__name__}."
        )
    root = params.get("params", params)
    if not isinstance(root, dict):
        raise ValueError(
            "Policy params payload is malformed: expected a 'params' mapping."
        )

    expected_dims = {
        "self_encoder": int(self_feature_dim(env_cfg)),
        "candidate_encoder": int(candidate_feature_dim(env_cfg)),
        "global_encoder": int(global_feature_dim(env_cfg)),
    }

    mismatches: list[str] = []
    for encoder_name, expected_dim in expected_dims.items():
        dense_name = f"{encoder_name}_0"
        module_payload = root.get(dense_name)
        if not isinstance(module_payload, dict):
            mismatches.append(
                f"{encoder_name}: missing module '{dense_name}' in checkpoint params"
            )
            continue
        kernel = module_payload.get("kernel")
        if kernel is None or getattr(kernel, "ndim", 0) < 1:
            mismatches.append(
                f"{encoder_name}: missing/invalid kernel at '{dense_name}.kernel'"
            )
            continue
        actual_dim = int(kernel.shape[0])
        if actual_dim != expected_dim:
            mismatches.append(
                f"{encoder_name}: expected input dim {expected_dim}, got {actual_dim}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Loaded policy params are incompatible with the configured environment "
            f"feature dimensions ({mismatch_text}). "
            "Use a checkpoint trained with matching env/model settings or retrain."
        )


class JaxTransitionBatch(NamedTuple):
    """Rollout data consumed by the JAX PPO update.

    Arrays keep rollout, environment, and source-planet dimensions until the
    update step flattens them. ``decision_mask`` identifies valid learner-owned
    source rows that should contribute to PPO losses.
    """

    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    player_count: jax.Array
    ship_bucket_mask: jax.Array
    decision_mask: jax.Array
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    returns: jax.Array
    advantages: jax.Array


@flax.struct.dataclass
class JaxTrainState:
    """Minimal immutable train state for Flax parameters and Optax state."""

    params: dict
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation = flax.struct.field(pytree_node=False)


def init_train_state(key: jax.Array, policy: object, cfg: TrainConfig) -> JaxTrainState:
    """Initialize policy parameters and optimizer state for JAX PPO."""

    dummy_self = jnp.zeros((1, self_feature_dim(cfg.task)), dtype=jnp.float32)
    dummy_candidate = jnp.zeros(
        (1, cfg.task.candidate_count, candidate_feature_dim(cfg.task)),
        dtype=jnp.float32,
    )
    dummy_global = jnp.zeros((1, global_feature_dim(cfg.task)), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, cfg.task.candidate_count), dtype=bool)
    dummy_player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)
    params = policy.init(
        key,
        dummy_self,
        dummy_candidate,
        dummy_global,
        dummy_mask,
        player_count=dummy_player_count,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optax.adam(cfg.training.lr),
    )
    return JaxTrainState(
        params=params, opt_state=optimizer.init(params), optimizer=optimizer
    )


def flatten_batch(
    batch: JaxTurnBatch,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Flatten environment/source dimensions into policy decision rows."""

    return (
        batch.self_features.reshape(-1, batch.self_features.shape[-1]),
        batch.candidate_features.reshape(
            -1, batch.candidate_features.shape[-2], batch.candidate_features.shape[-1]
        ),
        batch.global_features.reshape(-1, batch.global_features.shape[-1]),
        batch.candidate_mask.reshape(-1, batch.candidate_mask.shape[-1]),
        batch.decision_mask.reshape(-1),
    )


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


class ShieldedSequenceSample(NamedTuple):
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    entropy: jax.Array
    value: jax.Array
    ship_bucket_mask: jax.Array
    diagnostics: ShieldDiagnostics


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


def _select_env_action(
    condition: jax.Array,
    true_action: JaxAction,
    false_action: JaxAction,
) -> JaxAction:
    """Select between two batched actions independently for each environment."""

    return jax.tree.map(
        lambda true, false: jnp.where(
            condition.reshape((condition.shape[0],) + (1,) * (true.ndim - 1)),
            true,
            false,
        ),
        true_action,
        false_action,
    )


def _slice_env_axis(tree: object, env_index: jax.Array) -> object:
    """Slice leading env axis to a single row (keeps batch dim size 1)."""

    def slice_leaf(value):
        if isinstance(value, jax.Array) and value.ndim > 0:
            return jax.lax.dynamic_index_in_dim(value, env_index, axis=0, keepdims=True)
        return value

    return jax.tree.map(slice_leaf, tree)


def _stack_player_actions(player_actions: tuple[JaxAction, ...]) -> JaxAction:
    """Stack per-player batched actions into batched_step_multi_player layout."""

    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *player_actions)


def _gather_action_by_env(pool_action: JaxAction, indices: jax.Array) -> JaxAction:
    env_indices = jnp.arange(indices.shape[0], dtype=jnp.int32)
    return jax.tree.map(lambda field: field[indices, env_indices], pool_action)


def _sample_historical_action(
    key: jax.Array,
    game,
    batch: JaxTurnBatch,
    historical_params_pool: dict | None,
    stage_view: StageView,
    current_action: JaxAction,
    policy: object,
    cfg: TrainConfig,
) -> tuple[JaxAction, jax.Array]:
    env_count = batch.self_features.shape[0]
    has_snapshot = jnp.any(stage_view.snapshot_valid_mask)
    if historical_params_pool is None:
        return current_action, jnp.zeros((env_count,), dtype=jnp.int32)
    logits = jnp.where(
        stage_view.snapshot_valid_mask,
        jnp.log(jnp.maximum(stage_view.historical_selection_probs, 1e-12)),
        jnp.asarray(-1e9, dtype=jnp.float32),
    )
    selected = jax.random.categorical(key, logits, shape=(env_count,))
    pool_size = stage_view.snapshot_valid_mask.shape[0]
    pool_actions = jax.vmap(
        lambda idx, params: _sample_policy_action_with_params(
            jax.random.fold_in(key, idx),
            game,
            batch,
            params,
            policy,
            cfg,
            deterministic=cfg.opponents.snapshot.deterministic,
        )
    )(jnp.arange(pool_size, dtype=jnp.int32), historical_params_pool)
    historical_action = _gather_action_by_env(pool_actions, selected)
    fallback = jnp.logical_not(has_snapshot)
    action = jax.tree.map(
        lambda hist, cur: jnp.where(fallback, cur, hist),
        historical_action,
        current_action,
    )
    fallback_count = jnp.where(
        fallback,
        jnp.ones((env_count,), dtype=jnp.int32),
        jnp.zeros((env_count,), dtype=jnp.int32),
    )
    return action, fallback_count


def _opponent_count_metrics(
    effective_type_ids: jax.Array,
    learner_player: jax.Array,
) -> dict[str, jax.Array]:
    player_ids = jnp.arange(effective_type_ids.shape[1], dtype=jnp.int32)
    slot_mask = player_ids[None, :] != learner_player[:, None]
    slot_values = slot_mask.astype(jnp.float32)
    return {
        "opponent_slots_total": slot_values.sum(),
        "opponent_slots_latest": ((effective_type_ids == OPPONENT_LATEST) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_historical": (
            (effective_type_ids == OPPONENT_HISTORICAL) & slot_mask
        )
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_random": ((effective_type_ids == OPPONENT_RANDOM) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_noop": ((effective_type_ids == OPPONENT_NOOP) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_nearest_sniper": (
            (effective_type_ids == OPPONENT_NEAREST_SNIPER) & slot_mask
        )
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_turtle": ((effective_type_ids == OPPONENT_TURTLE) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_opportunistic": (
            (effective_type_ids == OPPONENT_OPPORTUNISTIC) & slot_mask
        )
        .astype(jnp.float32)
        .sum(),
    }


def _single_stage_family_id(stage_view: StageView) -> jax.Array:
    """Return the sole configured opponent family id, or -1 for true mixtures."""

    active = stage_view.family_probs > 0.0
    single = active.astype(jnp.int32).sum() == 1
    family_index = jnp.argmax(active.astype(jnp.int32))
    family_id = stage_view.family_ids[family_index]
    return jnp.where(single, family_id, jnp.asarray(-1, dtype=jnp.int32))


def _maybe_effective_single_family_id(
    family_id: jax.Array, stage_view: StageView
) -> jax.Array:
    has_historical = jnp.any(stage_view.snapshot_valid_mask)
    return jnp.where(
        (family_id == OPPONENT_HISTORICAL) & jnp.logical_not(has_historical),
        stage_view.fallback_family_id,
        family_id,
    )


def _sample_single_family_2p_action(
    key: jax.Array,
    family_id: jax.Array,
    game,
    batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Build a 2p opponent action without constructing unused families."""

    def latest_branch(_: None) -> JaxAction:
        return _sample_policy_action(
            key,
            game,
            batch,
            train_state,
            policy,
            cfg,
            deterministic=cfg.opponents.self_play.deterministic,
        )

    def historical_branch(_: None) -> JaxAction:
        current_action = latest_branch(None)
        historical_action, _fallback = _sample_historical_action(
            jax.random.fold_in(key, 71),
            game,
            batch,
            historical_params_pool,
            stage_view,
            current_action,
            policy,
            cfg,
        )
        return historical_action

    def random_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_random_action_from_batch(
            key, shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def nearest_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_sniper_action_from_batch(
            shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def turtle_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_turtle_action_from_batch(
            shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def opportunistic_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_opportunistic_action_from_batch(
            shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def noop_branch(_: None) -> JaxAction:
        return build_noop_action_from_batch(batch, cfg)

    return jax.lax.switch(
        jnp.clip(family_id, 0, OPPONENT_NOOP),
        (
            latest_branch,
            historical_branch,
            nearest_branch,
            turtle_branch,
            opportunistic_branch,
            random_branch,
            noop_branch,
        ),
        None,
    )


def _opponent_params_for_player(
    player_id: jax.Array,
    train_state: JaxTrainState,
    opponent_params_by_player: tuple[dict, ...] | None,
    *,
    player_count: int,
) -> dict:
    if opponent_params_by_player is None:
        return train_state.params
    return jax.lax.switch(
        jnp.asarray(player_id, dtype=jnp.int32),
        tuple(opponent_params_by_player[index] for index in range(player_count)),
    )


def _sample_single_family_4p_action(
    key: jax.Array,
    family_id: jax.Array,
    player_id: jax.Array,
    game,
    batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Build one 4p player action without constructing unused families."""

    opponent_params = _opponent_params_for_player(
        player_id,
        train_state,
        opponent_params_by_player,
        player_count=int(cfg.task.player_count),
    )

    def latest_branch(_: None) -> JaxAction:
        return _sample_policy_action_with_params(
            key,
            game,
            batch,
            opponent_params,
            policy,
            cfg,
            deterministic=cfg.opponents.self_play.deterministic,
        )

    def historical_branch(_: None) -> JaxAction:
        current_action = latest_branch(None)
        historical_action, _fallback = _sample_historical_action(
            jax.random.fold_in(key, 71),
            game,
            batch,
            historical_params_pool,
            stage_view,
            current_action,
            policy,
            cfg,
        )
        return historical_action

    def random_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_random_action_from_batch(
            jax.random.fold_in(key, cfg.task.player_count),
            shielded.batch,
            cfg,
            shielded.ship_bucket_mask,
        )

    def nearest_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_sniper_action_from_batch(
            shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def turtle_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_turtle_action_from_batch(
            shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def opportunistic_branch(_: None) -> JaxAction:
        shielded = jax.vmap(
            lambda game_row, turn_row: apply_trajectory_shield_to_turn_batch(
                game_row, turn_row, cfg.task
            )
        )(game, batch)
        return build_opportunistic_action_from_batch(
            shielded.batch, cfg, shielded.ship_bucket_mask
        )

    def noop_branch(_: None) -> JaxAction:
        return build_noop_action_from_batch(batch, cfg)

    return jax.lax.switch(
        jnp.clip(family_id, 0, OPPONENT_NOOP),
        (
            latest_branch,
            historical_branch,
            nearest_branch,
            turtle_branch,
            opportunistic_branch,
            random_branch,
            noop_branch,
        ),
        None,
    )


def _sample_mixed_opponent_2p_action(
    opp_key: jax.Array,
    opp_game,
    opp_batch_cache: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    slot_type: jax.Array,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Sample per-env opponent actions with lax.switch (no eager multi-family compute)."""

    env_count = int(slot_type.shape[0])
    env_indices = jnp.arange(env_count, dtype=jnp.int32)
    per_env = jax.vmap(
        lambda env_index: _sample_single_family_2p_action(
            jax.random.fold_in(opp_key, env_index),
            jnp.asarray(slot_type[env_index], dtype=jnp.int32),
            _slice_env_axis(opp_game, env_index),
            _slice_env_axis(opp_batch_cache, env_index),
            train_state,
            policy,
            cfg,
            stage_view,
            historical_params_pool,
        )
    )(env_indices)
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=1), per_env)


def _four_player_step_action(
    player_id: jax.Array,
    *,
    opp_key: jax.Array,
    player_games,
    player_batches,
    effective_type_ids: jax.Array,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    learner_action: JaxAction,
    learner_player: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
    active_stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Build one player's env-batch action row (vmapped over player_id)."""

    player_batch = jax.tree.map(
        lambda x: jnp.take(x, player_id, axis=0), player_batches
    )
    player_game = jax.tree.map(lambda x: jnp.take(x, player_id, axis=0), player_games)
    player_key = jax.random.fold_in(opp_key, player_id)
    slot_type = effective_type_ids[:, player_id]
    if cfg.opponents.mode.opponent == "self":

        def single_player_branch(_: None) -> JaxAction:
            return _sample_single_family_4p_action(
                player_key,
                effective_single_family_id,
                player_id,
                player_game,
                player_batch,
                train_state,
                policy,
                cfg,
                opponent_params_by_player,
                active_stage_view,
                historical_params_pool,
            )

        def mixed_player_branch(_: None) -> JaxAction:
            return _sample_mixed_player_4p_action(
                player_key,
                player_id,
                player_game,
                player_batch,
                slot_type,
                train_state,
                policy,
                cfg,
                opponent_params_by_player,
                active_stage_view,
                historical_params_pool,
            )

        opponent_action = jax.lax.cond(
            single_family,
            single_player_branch,
            mixed_player_branch,
            None,
        )
    elif cfg.opponents.mode.opponent == "random":
        player_shielded = jax.vmap(
            lambda game, turn: apply_trajectory_shield_to_turn_batch(
                game, turn, cfg.task
            )
        )(player_game, player_batch)
        opponent_action = build_random_action_from_batch(
            player_key,
            player_shielded.batch,
            cfg,
            player_shielded.ship_bucket_mask,
        )
    else:
        raise ValueError(
            "JAX training supports opponent='self' or opponent='random', "
            f"got {cfg.opponents.mode.opponent!r}."
        )
    is_learner_player = learner_player == player_id
    return _select_env_action(is_learner_player, learner_action, opponent_action)


def _sample_mixed_player_4p_action(
    player_key: jax.Array,
    player_id: jax.Array,
    player_game,
    player_batch: JaxTurnBatch,
    slot_type: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Sample one 4p player's batched actions with per-env lax.switch."""

    env_count = int(slot_type.shape[0])
    env_indices = jnp.arange(env_count, dtype=jnp.int32)
    per_env = jax.vmap(
        lambda env_index: _sample_single_family_4p_action(
            jax.random.fold_in(player_key, env_index),
            jnp.asarray(slot_type[env_index], dtype=jnp.int32),
            player_id,
            _slice_env_axis(player_game, env_index),
            _slice_env_axis(player_batch, env_index),
            train_state,
            policy,
            cfg,
            opponent_params_by_player,
            stage_view,
            historical_params_pool,
        )
    )(env_indices)
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=1), per_env)


def collect_rollout_jax(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None = None,
    stage_view: StageView | None = None,
    historical_params_pool: dict | None = None,
    update: int = 0,
    env_index_offset: int | jax.Array = 0,
) -> tuple[
    jax.Array, JaxEnvState, JaxTurnBatch, JaxTransitionBatch, dict[str, jax.Array]
]:
    """Collect one fixed-length rollout entirely in JAX.

    The function is designed to be wrapped in ``jax.jit`` by the training loop.
    It samples learner actions, generates the configured opponent actions,
    advances the vectorized JAX environment, resets completed episodes, and
    returns PPO transitions plus rollout metrics.
    """

    env_indices = jnp.arange(turn_batch.self_features.shape[0], dtype=jnp.int32) + jnp.asarray(
        env_index_offset, dtype=jnp.int32
    )
    active_stage_view = default_stage_view(cfg) if stage_view is None else stage_view

    def scan_step(carry, _):
        key, state, batch, opp_batch_cache = carry
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
        sample = _sample_shielded_sequence_with_params(
            learner_key,
            state.game,
            batch,
            train_state.params,
            policy,
            cfg,
            deterministic=False,
        )
        target = sample.target_index
        bucket = sample.ship_bucket
        log_prob = sample.log_prob
        learner_action = build_action_from_batch(batch, target, bucket, cfg)

        env_count = state.game.step.shape[0]
        single_family_id = _single_stage_family_id(active_stage_view)
        effective_single_family_id = _maybe_effective_single_family_id(
            single_family_id, active_stage_view
        )
        single_family = single_family_id >= 0
        opponent_type_ids = sample_opponent_type_ids_jax(
            jax.random.fold_in(opp_key, 9973),
            env_count,
            cfg.task.player_count,
            ids=active_stage_view.family_ids,
            probs=active_stage_view.family_probs,
        )
        opponent_type_ids = jnp.where(
            single_family,
            jnp.full(
                (env_count, cfg.task.player_count),
                single_family_id,
                dtype=jnp.int32,
            ),
            opponent_type_ids,
        )
        has_historical = jnp.any(active_stage_view.snapshot_valid_mask)
        effective_type_ids = jnp.where(
            (opponent_type_ids == OPPONENT_HISTORICAL)
            & jnp.logical_not(has_historical),
            active_stage_view.fallback_family_id,
            opponent_type_ids,
        )
        family_counts = _opponent_count_metrics(
            effective_type_ids, state.learner_player
        )
        historical_fallback_slots = (
            (
                (
                    (opponent_type_ids == OPPONENT_HISTORICAL)
                    & (effective_type_ids == OPPONENT_LATEST)
                )
                & (
                    jnp.arange(cfg.task.player_count, dtype=jnp.int32)[None, :]
                    != state.learner_player[:, None]
                )
            )
            .astype(jnp.float32)
            .sum()
        )

        if cfg.task.player_count == 2:
            opp_game = state.game._replace(
                player=(1 - state.learner_player).astype(jnp.int32)
            )
            slot_type = jnp.take_along_axis(
                effective_type_ids,
                (1 - state.learner_player).astype(jnp.int32)[:, None],
                axis=1,
            ).squeeze(axis=1)
            if cfg.opponents.mode.opponent == "self":

                def single_opponent_branch(_: None) -> JaxAction:
                    return _sample_single_family_2p_action(
                        opp_key,
                        effective_single_family_id,
                        opp_game,
                        opp_batch_cache,
                        train_state,
                        policy,
                        cfg,
                        active_stage_view,
                        historical_params_pool,
                    )

                def mixed_opponent_branch(_: None) -> JaxAction:
                    return _sample_mixed_opponent_2p_action(
                        opp_key,
                        opp_game,
                        opp_batch_cache,
                        train_state,
                        policy,
                        cfg,
                        slot_type,
                        active_stage_view,
                        historical_params_pool,
                    )

                opponent_action = jax.lax.cond(
                    single_family,
                    single_opponent_branch,
                    mixed_opponent_branch,
                    None,
                )
            elif cfg.opponents.mode.opponent == "random":
                opp_shielded = jax.vmap(
                    lambda game, turn: apply_trajectory_shield_to_turn_batch(
                        game, turn, cfg.task
                    )
                )(opp_game, opp_batch_cache)
                opponent_action = build_random_action_from_batch(
                    opp_key, opp_shielded.batch, cfg, opp_shielded.ship_bucket_mask
                )
            else:
                raise ValueError(
                    "JAX training supports opponent='self' or opponent='random', "
                    f"got {cfg.opponents.mode.opponent!r}."
                )

            next_state, result = batched_step(
                state, learner_action, opponent_action, cfg.task, cfg.reward
            )
        elif cfg.task.player_count == 4:
            player_actions = []
            player_ids = jnp.arange(cfg.task.player_count, dtype=jnp.int32)
            player_games = jax.vmap(
                lambda player_id: state.game._replace(
                    player=jnp.full_like(state.game.step, player_id, dtype=jnp.int32)
                )
            )(player_ids)
            flat_player_games = jax.tree.map(
                lambda x: x.reshape((cfg.task.player_count * env_count,) + x.shape[2:]),
                player_games,
            )
            flat_player_batch = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                flat_player_games
            )
            player_batches = jax.tree.map(
                lambda x: x.reshape((cfg.task.player_count, env_count) + x.shape[1:]),
                flat_player_batch,
            )
            per_player_action = jax.vmap(
                lambda player_id: _four_player_step_action(
                    player_id,
                    opp_key=opp_key,
                    player_games=player_games,
                    player_batches=player_batches,
                    effective_type_ids=effective_type_ids,
                    single_family=single_family,
                    effective_single_family_id=effective_single_family_id,
                    learner_action=learner_action,
                    learner_player=state.learner_player,
                    train_state=train_state,
                    policy=policy,
                    cfg=cfg,
                    opponent_params_by_player=opponent_params_by_player,
                    active_stage_view=active_stage_view,
                    historical_params_pool=historical_params_pool,
                )
            )(player_ids)
            multi_player_action = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, 1), per_player_action
            )
            next_state, result = batched_step_multi_player(
                state, multi_player_action, cfg.task, cfg.reward
            )
        else:
            raise ValueError(
                "JAX PPO rollout supports env.player_count of 2 or 4, "
                f"got {cfg.task.player_count}."
            )

        def maybe_reset(new, old):
            cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
            return jnp.where(cond, new, old)

        def reset_branch(_):
            reset_keys = jax.random.split(reset_key, batch.self_features.shape[0])
            reset_states, reset_batches = batched_reset(reset_keys, cfg.task)
            reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
            reset_states, reset_batches = assign_learner_players(
                reset_states,
                env_indices,
                reset_episode_counts,
                cfg.task,
                cfg.opponents.mode.alternate_player_sides,
            )
            merged_state = jax.tree.map(maybe_reset, reset_states, next_state)
            merged_batch = jax.tree.map(maybe_reset, reset_batches, result.batch)
            return merged_state, merged_batch

        def no_reset_branch(_):
            return next_state, result.batch

        next_state, next_batch = jax.lax.cond(
            jnp.any(result.done), reset_branch, no_reset_branch, operand=None
        )
        if cfg.task.player_count == 2:
            next_opp_game = next_state.game._replace(
                player=(1 - next_state.learner_player).astype(jnp.int32)
            )
            next_opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                next_opp_game
            )
        else:
            next_opp_batch_cache = opp_batch_cache

        transition = {
            "self_features": batch.self_features,
            "candidate_features": batch.candidate_features,
            "global_features": batch.global_features,
            "candidate_mask": batch.candidate_mask,
            "player_count": jnp.full(
                batch.decision_mask.shape, cfg.task.player_count, dtype=jnp.int32
            ),
            "ship_bucket_mask": sample.ship_bucket_mask.reshape(
                batch.decision_mask.shape
                + (
                    target.shape[-1],
                    cfg.task.candidate_count,
                    cfg.task.ship_bucket_count,
                )
            ),
            "decision_mask": jnp.broadcast_to(
                batch.decision_mask[..., None],
                batch.decision_mask.shape + (target.shape[-1],),
            ),
            "target_index": target.reshape(
                batch.decision_mask.shape + (target.shape[-1],)
            ),
            "ship_bucket": bucket.reshape(
                batch.decision_mask.shape + (bucket.shape[-1],)
            ),
            "log_prob": log_prob.reshape(
                batch.decision_mask.shape + (log_prob.shape[-1],)
            ),
            "value": sample.value.reshape(batch.decision_mask.shape),
            "reward": result.reward,
            "done": result.done,
            "terminal_is_first": result.terminal_is_first,
            "terminal_placement": result.terminal_placement,
            "terminal_score_share": result.terminal_score_share,
            "terminal_survival_time": result.terminal_survival_time,
        }
        if not cfg.training.lean_rollout_metrics:
            transition.update(
                {
                    "trajectory_shield_blocked_count": sample.diagnostics.blocked_count,
                    "trajectory_shield_blocked_sun_count": sample.diagnostics.blocked_sun_count,
                    "trajectory_shield_blocked_bounds_count": sample.diagnostics.blocked_bounds_count,
                    "trajectory_shield_blocked_unintended_hit_count": sample.diagnostics.blocked_unintended_hit_count,
                    "trajectory_shield_blocked_horizon_count": sample.diagnostics.blocked_horizon_count,
                    "trajectory_shield_fallback_noop_count": sample.diagnostics.fallback_noop_count,
                    "trajectory_shield_legal_non_noop_count": sample.diagnostics.legal_non_noop_count,
                    "trajectory_shield_original_non_noop_count": sample.diagnostics.original_non_noop_count,
                    "trajectory_shield_legal_non_noop_rate": sample.diagnostics.legal_non_noop_rate,
                    "opponent_slots_total": family_counts["opponent_slots_total"],
                    "opponent_slots_latest": family_counts["opponent_slots_latest"],
                    "opponent_slots_historical": family_counts[
                        "opponent_slots_historical"
                    ],
                    "opponent_slots_random": family_counts["opponent_slots_random"],
                    "opponent_slots_noop": family_counts["opponent_slots_noop"],
                    "opponent_slots_nearest_sniper": family_counts[
                        "opponent_slots_nearest_sniper"
                    ],
                    "opponent_slots_turtle": family_counts["opponent_slots_turtle"],
                    "opponent_slots_opportunistic": family_counts[
                        "opponent_slots_opportunistic"
                    ],
                    "opponent_historical_fallback_latest_slots": historical_fallback_slots,
                }
            )
        return (key, next_state, next_batch, next_opp_batch_cache), transition

    if cfg.task.player_count == 2:
        initial_opp_game = env_state.game._replace(
            player=(1 - env_state.learner_player).astype(jnp.int32)
        )
        initial_opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
            initial_opp_game
        )
    else:
        initial_opp_batch_cache = turn_batch

    (key, env_state, turn_batch, _), data = jax.lax.scan(
        scan_step,
        (key, env_state, turn_batch, initial_opp_batch_cache),
        None,
        length=cfg.training.rollout_steps,
    )
    returns_step = discounted_returns(data["reward"], data["done"], cfg.training.gamma)
    returns = jnp.broadcast_to(
        returns_step[..., None, None], data["target_index"].shape
    )
    advantages = returns - data["value"][..., None]
    transitions = JaxTransitionBatch(
        self_features=data["self_features"],
        candidate_features=data["candidate_features"],
        global_features=data["global_features"],
        candidate_mask=data["candidate_mask"],
        player_count=data["player_count"],
        ship_bucket_mask=data["ship_bucket_mask"],
        decision_mask=data["decision_mask"],
        target_index=data["target_index"],
        ship_bucket=data["ship_bucket"],
        log_prob=data["log_prob"],
        returns=returns,
        advantages=advantages,
    )
    opponent_slots = jnp.array(
        cfg.training.rollout_steps
        * turn_batch.self_features.shape[0]
        * max(cfg.task.player_count - 1, 0),
        dtype=jnp.float32,
    )
    mode = (
        cfg.opponents.mode.multi_opponent_mode.strip().lower()
        if cfg.opponents.self_play.enabled
        else "shared_current"
    )
    snapshot_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if (
            cfg.opponents.mode.opponent == "self"
            and mode == "sampled_pool"
            and opponent_params_by_player is not None
        )
        else jnp.array(0.0, dtype=jnp.float32)
    )
    current_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if (
            cfg.opponents.mode.opponent == "self"
            and (
                mode == "shared_current"
                or (mode == "sampled_pool" and opponent_params_by_player is None)
            )
        )
        else (
            jnp.array(
                min(max(cfg.opponents.mix.weights.get("latest", 0.0), 0.0), 1.0),
                dtype=jnp.float32,
            )
            if cfg.opponents.mode.opponent == "self" and mode == "mixed"
            else jnp.array(0.0, dtype=jnp.float32)
        )
    )
    random_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if cfg.opponents.mode.opponent == "random"
        else (
            (1.0 - current_share)
            if cfg.opponents.mode.opponent == "self" and mode == "mixed"
            else jnp.array(0.0, dtype=jnp.float32)
        )
    )
    metrics = (
        _rollout_diagnostics_lean(
            data=data,
            transitions=transitions,
            turn_batch=turn_batch,
            cfg=cfg,
        )
        if cfg.training.lean_rollout_metrics
        else _rollout_diagnostics(
            data=data,
            transitions=transitions,
            turn_batch=turn_batch,
            cfg=cfg,
            opponent_slots=opponent_slots,
            snapshot_share=snapshot_share,
            current_share=current_share,
            random_share=random_share,
        )
    )
    return key, env_state, turn_batch, transitions, metrics


def _rollout_diagnostics(
    *,
    data: dict[str, jax.Array],
    transitions: JaxTransitionBatch,
    turn_batch: JaxTurnBatch,
    cfg: TrainConfig,
    opponent_slots: jax.Array,
    snapshot_share: jax.Array,
    current_share: jax.Array,
    random_share: jax.Array,
) -> dict[str, jax.Array]:
    self_schema = self_feature_schema(cfg.task)
    candidate_schema = candidate_feature_schema(cfg.task)
    global_schema = global_feature_schema(cfg.task)

    valid_non_noop_targets = (
        data["candidate_mask"][..., 1:].astype(jnp.float32).sum(axis=-1)
    )
    row_mask = transitions.decision_mask.astype(jnp.float32)
    valid_non_noop_targets_sum = (valid_non_noop_targets[..., None] * row_mask).sum()
    valid_non_noop_target_rows = row_mask.sum()
    only_noop_rows = (
        (valid_non_noop_targets[..., None] <= 0.0).astype(jnp.float32) * row_mask
    ).sum()
    only_noop_fraction = jnp.where(
        valid_non_noop_target_rows > 0.0,
        only_noop_rows / valid_non_noop_target_rows,
        0.0,
    )
    done_float = data["done"].astype(jnp.float32)
    reward_mean = data["reward"].mean()
    episode_done = done_float.sum()
    episode_reward_sum = (data["reward"] * done_float).sum()
    episodes_2p = jnp.where(cfg.task.player_count == 2, episode_done, 0.0)
    episodes_4p = jnp.where(cfg.task.player_count == 4, episode_done, 0.0)
    first_place_sum = (data["terminal_is_first"] * done_float).sum()
    placement_4p_sum = jnp.where(
        cfg.task.player_count == 4, (data["terminal_placement"] * done_float).sum(), 0.0
    )
    survival_time_sum = (data["terminal_survival_time"] * done_float).sum()
    score_share_sum = (data["terminal_score_share"] * done_float).sum()
    selected_target = data["target_index"]
    decision_count = row_mask.sum()
    noop_count = ((selected_target == 0).astype(jnp.float32) * row_mask).sum()
    non_noop_count = (((selected_target != 0).astype(jnp.float32)) * row_mask).sum()
    source_ships = (
        data["self_features"][..., self_schema.slice("source_ships")].squeeze(-1)
        * cfg.task.max_ships
    )[..., None]
    launched_ships = ship_count_for_bucket_jax(
        source_ships, data["ship_bucket"], cfg.task.ship_bucket_count
    )
    launched_ship_mask = (selected_target != 0).astype(jnp.float32) * row_mask
    launched_ship_count = launched_ship_mask.sum()
    launched_ship_total = (launched_ships * launched_ship_mask).sum()
    launched_ship_speed_total = (
        launched_ship_mask * fleet_speed(launched_ships, MAX_FLEET_SPEED)
    ).sum()

    terminal_row_mask = row_mask * done_float[..., None, None]
    win_row_mask = terminal_row_mask * data["terminal_is_first"][..., None, None]
    loss_row_mask = terminal_row_mask * (
        1.0 - data["terminal_is_first"][..., None, None]
    )
    win_episode_rows = win_row_mask.sum()
    loss_episode_rows = loss_row_mask.sum()

    planet_fractions_slice = global_schema.slice("planet_fractions")
    ship_fractions_slice = global_schema.slice("ship_fractions")
    planet_delta_slots_slice = global_schema.slice("planet_delta_slots")
    owner_production_slice = global_schema.slice("owner_relative_production")

    planet_fractions = data["global_features"][..., planet_fractions_slice]
    ship_fractions = data["global_features"][..., ship_fractions_slice]
    planet_delta_slots = data["global_features"][..., planet_delta_slots_slice]
    owner_production = data["global_features"][..., owner_production_slice]

    my_planets = planet_fractions[..., 0] * MAX_PLANETS
    my_garrison_ships = ship_fractions[..., 0] * (MAX_PLANETS * cfg.task.max_ships)
    planet_delta = planet_delta_slots[..., 0] * MAX_PLANETS
    production_diff = owner_production[..., 0] * MAX_PRODUCTION
    planet_diff = planet_delta
    planets_taken_step = jnp.maximum(planet_delta, 0.0)
    planets_lost_step = jnp.maximum(-planet_delta, 0.0)
    selected_candidate_features = jnp.take_along_axis(
        data["candidate_features"][..., None, :, :],
        selected_target[..., None, None].repeat(
            data["candidate_features"].shape[-1], axis=-1
        ),
        axis=4,
    ).squeeze(axis=4)

    target_ownership_slice = candidate_schema.slice("target_ownership_flags")
    target_ownership = selected_candidate_features[..., target_ownership_slice]

    neutral_target_count = (target_ownership[..., 0] * row_mask).sum()
    friendly_target_count = (target_ownership[..., 1] * row_mask).sum()
    enemy_target_count = (target_ownership[..., 2] * row_mask).sum()
    garrisoned_ships_per_planet = my_garrison_ships / jnp.maximum(my_planets, 1.0)
    won_planets_owned_total = (my_planets[..., None] * win_row_mask).sum()
    lost_planets_owned_total = (my_planets[..., None] * loss_row_mask).sum()
    won_planets_lost_total = (planets_lost_step[..., None] * win_row_mask).sum()
    lost_planets_lost_total = (planets_lost_step[..., None] * loss_row_mask).sum()
    won_planets_taken_total = (planets_taken_step[..., None] * win_row_mask).sum()
    lost_planets_taken_total = (planets_taken_step[..., None] * loss_row_mask).sum()
    won_garrisoned_ships_per_planet_total = (
        garrisoned_ships_per_planet[..., None] * win_row_mask
    ).sum()
    lost_garrisoned_ships_per_planet_total = (
        garrisoned_ships_per_planet[..., None] * loss_row_mask
    ).sum()
    won_planet_diff_total = (planet_diff[..., None] * win_row_mask).sum()
    lost_planet_diff_total = (planet_diff[..., None] * loss_row_mask).sum()
    won_production_diff_total = (production_diff[..., None] * win_row_mask).sum()
    lost_production_diff_total = (production_diff[..., None] * loss_row_mask).sum()

    metrics = {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * turn_batch.self_features.shape[0],
            dtype=jnp.float32,
        ),
        "samples": transitions.decision_mask.astype(jnp.float32).sum(),
        "valid_non_noop_targets_sum": valid_non_noop_targets_sum,
        "valid_non_noop_target_rows": valid_non_noop_target_rows,
        "only_noop_rows": only_noop_rows,
        "valid_non_noop_targets_per_row": jnp.where(
            valid_non_noop_target_rows > 0.0,
            valid_non_noop_targets_sum / valid_non_noop_target_rows,
            0.0,
        ),
        "only_noop_fraction": only_noop_fraction,
        "trajectory_shield_blocked_count": data[
            "trajectory_shield_blocked_count"
        ].sum(),
        "trajectory_shield_blocked_sun_count": data[
            "trajectory_shield_blocked_sun_count"
        ].sum(),
        "trajectory_shield_blocked_bounds_count": data[
            "trajectory_shield_blocked_bounds_count"
        ].sum(),
        "trajectory_shield_blocked_unintended_hit_count": data[
            "trajectory_shield_blocked_unintended_hit_count"
        ].sum(),
        "trajectory_shield_blocked_horizon_count": data[
            "trajectory_shield_blocked_horizon_count"
        ].sum(),
        "trajectory_shield_fallback_noop_count": data[
            "trajectory_shield_fallback_noop_count"
        ].sum(),
        "trajectory_shield_legal_non_noop_count": data[
            "trajectory_shield_legal_non_noop_count"
        ].sum(),
        "trajectory_shield_original_non_noop_count": data[
            "trajectory_shield_original_non_noop_count"
        ].sum(),
        "trajectory_shield_legal_non_noop_rate": jnp.where(
            data["trajectory_shield_original_non_noop_count"].sum() > 0.0,
            data["trajectory_shield_legal_non_noop_count"].sum()
            / data["trajectory_shield_original_non_noop_count"].sum(),
            0.0,
        ),
        "episode_done": episode_done,
        "win_episode_rows": win_episode_rows,
        "loss_episode_rows": loss_episode_rows,
        "non_noop_count": non_noop_count,
        "launched_ship_count": launched_ship_count,
        "launched_ship_total": launched_ship_total,
        "launched_ship_speed_total": launched_ship_speed_total,
        "won_planets_owned_total": won_planets_owned_total,
        "lost_planets_owned_total": lost_planets_owned_total,
        "won_planets_lost_total": won_planets_lost_total,
        "lost_planets_lost_total": lost_planets_lost_total,
        "won_planets_taken_total": won_planets_taken_total,
        "lost_planets_taken_total": lost_planets_taken_total,
        "won_garrisoned_ships_per_planet_total": won_garrisoned_ships_per_planet_total,
        "lost_garrisoned_ships_per_planet_total": lost_garrisoned_ships_per_planet_total,
        "won_planet_diff_total": won_planet_diff_total,
        "lost_planet_diff_total": lost_planet_diff_total,
        "won_production_diff_total": won_production_diff_total,
        "lost_production_diff_total": lost_production_diff_total,
        "average_reward": reward_mean,
        "episode_reward_mean": jnp.where(
            episode_done > 0.0, episode_reward_sum / episode_done, 0.0
        ),
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "wins_2p": jnp.where(cfg.task.player_count == 2, first_place_sum, 0.0),
        "first_places_4p": jnp.where(cfg.task.player_count == 4, first_place_sum, 0.0),
        "placement_4p_sum": placement_4p_sum,
        "survival_time_sum": survival_time_sum,
        "score_share_sum": score_share_sum,
        "decision_count": decision_count,
        "noop_count": noop_count,
        "friendly_target_count": friendly_target_count,
        "enemy_target_count": enemy_target_count,
        "neutral_target_count": neutral_target_count,
        "win_rate_2p": jnp.where(episodes_2p > 0.0, first_place_sum / episodes_2p, 0.0),
        "first_place_rate_4p": jnp.where(
            episodes_4p > 0.0, first_place_sum / episodes_4p, 0.0
        ),
        "average_placement_4p": jnp.where(
            episodes_4p > 0.0, placement_4p_sum / episodes_4p, 0.0
        ),
        "survival_time": jnp.where(
            episode_done > 0.0, survival_time_sum / episode_done, 0.0
        ),
        "score_share": jnp.where(
            episode_done > 0.0, score_share_sum / episode_done, 0.0
        ),
        "noop_percent": jnp.where(
            decision_count > 0.0, (noop_count / decision_count) * 100.0, 0.0
        ),
        "friendly_target_percent": jnp.where(
            decision_count > 0.0, (friendly_target_count / decision_count) * 100.0, 0.0
        ),
        "enemy_target_percent": jnp.where(
            decision_count > 0.0, (enemy_target_count / decision_count) * 100.0, 0.0
        ),
        "neutral_target_percent": jnp.where(
            decision_count > 0.0, (neutral_target_count / decision_count) * 100.0, 0.0
        ),
        "overall_win_rate": jnp.where(
            episode_done > 0.0, first_place_sum / episode_done, 0.0
        ),
        "opponent_slots_total": data["opponent_slots_total"].sum(),
        "opponent_slots_latest": data["opponent_slots_latest"].sum(),
        "opponent_slots_historical": data["opponent_slots_historical"].sum(),
        "opponent_slots_random": data["opponent_slots_random"].sum(),
        "opponent_slots_noop": data["opponent_slots_noop"].sum(),
        "opponent_slots_nearest_sniper": data["opponent_slots_nearest_sniper"].sum(),
        "opponent_slots_turtle": data["opponent_slots_turtle"].sum(),
        "opponent_slots_opportunistic": data["opponent_slots_opportunistic"].sum(),
        "opponent_historical_fallback_latest_slots": data[
            "opponent_historical_fallback_latest_slots"
        ].sum(),
        "opponent_current_slots": data["opponent_slots_latest"].sum(),
        "opponent_random_slots": data["opponent_slots_random"].sum(),
        "opponent_snapshot_slots": data["opponent_slots_historical"].sum(),
        "won_non_noop_actions_per_step": jnp.where(
            win_episode_rows > 0.0,
            (non_noop_count * done_float.sum())
            / jnp.maximum(win_episode_rows * done_float.sum(), 1.0),
            0.0,
        ),
        "lost_non_noop_actions_per_step": jnp.where(
            loss_episode_rows > 0.0,
            (non_noop_count * done_float.sum())
            / jnp.maximum(loss_episode_rows * done_float.sum(), 1.0),
            0.0,
        ),
        "won_avg_fleet_launch_size": jnp.where(
            win_episode_rows > 0.0,
            launched_ship_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
        "lost_avg_fleet_launch_size": jnp.where(
            loss_episode_rows > 0.0,
            launched_ship_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
        "won_avg_planets_owned": jnp.where(
            win_episode_rows > 0.0,
            won_planets_owned_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planets_owned": jnp.where(
            loss_episode_rows > 0.0,
            lost_planets_owned_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_planets_lost": jnp.where(
            win_episode_rows > 0.0,
            won_planets_lost_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planets_lost": jnp.where(
            loss_episode_rows > 0.0,
            lost_planets_lost_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_planets_taken": jnp.where(
            win_episode_rows > 0.0,
            won_planets_taken_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planets_taken": jnp.where(
            loss_episode_rows > 0.0,
            lost_planets_taken_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_garrisoned_ships_per_planet": jnp.where(
            win_episode_rows > 0.0,
            won_garrisoned_ships_per_planet_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_garrisoned_ships_per_planet": jnp.where(
            loss_episode_rows > 0.0,
            lost_garrisoned_ships_per_planet_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_planet_diff": jnp.where(
            win_episode_rows > 0.0,
            won_planet_diff_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planet_diff": jnp.where(
            loss_episode_rows > 0.0,
            lost_planet_diff_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_production_diff": jnp.where(
            win_episode_rows > 0.0,
            won_production_diff_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_production_diff": jnp.where(
            loss_episode_rows > 0.0,
            lost_production_diff_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_launch_fleet_speed": jnp.where(
            win_episode_rows > 0.0,
            launched_ship_speed_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
        "lost_avg_launch_fleet_speed": jnp.where(
            loss_episode_rows > 0.0,
            launched_ship_speed_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
    }
    return metrics


def _rollout_diagnostics_lean(
    *,
    data: dict[str, jax.Array],
    transitions: JaxTransitionBatch,
    turn_batch: JaxTurnBatch,
    cfg: TrainConfig,
) -> dict[str, jax.Array]:
    """Compute rollout metrics without per-step shield/opponent scan payloads."""

    row_mask = transitions.decision_mask.astype(jnp.float32)
    done_float = data["done"].astype(jnp.float32)
    reward_mean = data["reward"].mean()
    episode_done = done_float.sum()
    episode_reward_sum = (data["reward"] * done_float).sum()
    episodes_2p = jnp.where(cfg.task.player_count == 2, episode_done, 0.0)
    episodes_4p = jnp.where(cfg.task.player_count == 4, episode_done, 0.0)
    first_place_sum = (data["terminal_is_first"] * done_float).sum()
    placement_4p_sum = jnp.where(
        cfg.task.player_count == 4, (data["terminal_placement"] * done_float).sum(), 0.0
    )
    survival_time_sum = (data["terminal_survival_time"] * done_float).sum()
    score_share_sum = (data["terminal_score_share"] * done_float).sum()
    selected_target = data["target_index"]
    decision_count = row_mask.sum()
    noop_count = ((selected_target == 0).astype(jnp.float32) * row_mask).sum()
    zero = jnp.array(0.0, dtype=jnp.float32)
    return {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * turn_batch.self_features.shape[0],
            dtype=jnp.float32,
        ),
        "samples": transitions.decision_mask.astype(jnp.float32).sum(),
        "valid_non_noop_targets_sum": zero,
        "valid_non_noop_target_rows": row_mask.sum(),
        "only_noop_rows": zero,
        "valid_non_noop_targets_per_row": zero,
        "only_noop_fraction": zero,
        "trajectory_shield_blocked_count": zero,
        "trajectory_shield_blocked_sun_count": zero,
        "trajectory_shield_blocked_bounds_count": zero,
        "trajectory_shield_blocked_unintended_hit_count": zero,
        "trajectory_shield_blocked_horizon_count": zero,
        "trajectory_shield_fallback_noop_count": zero,
        "trajectory_shield_legal_non_noop_count": zero,
        "trajectory_shield_original_non_noop_count": zero,
        "trajectory_shield_legal_non_noop_rate": zero,
        "episode_done": episode_done,
        "win_episode_rows": zero,
        "loss_episode_rows": zero,
        "non_noop_count": zero,
        "launched_ship_count": zero,
        "launched_ship_total": zero,
        "launched_ship_speed_total": zero,
        "won_planets_owned_total": zero,
        "lost_planets_owned_total": zero,
        "won_planets_lost_total": zero,
        "lost_planets_lost_total": zero,
        "won_planets_taken_total": zero,
        "lost_planets_taken_total": zero,
        "won_garrisoned_ships_per_planet_total": zero,
        "lost_garrisoned_ships_per_planet_total": zero,
        "won_planet_diff_total": zero,
        "lost_planet_diff_total": zero,
        "won_production_diff_total": zero,
        "lost_production_diff_total": zero,
        "average_reward": reward_mean,
        "episode_reward_mean": jnp.where(
            episode_done > 0.0, episode_reward_sum / episode_done, 0.0
        ),
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "wins_2p": jnp.where(cfg.task.player_count == 2, first_place_sum, 0.0),
        "first_places_4p": jnp.where(cfg.task.player_count == 4, first_place_sum, 0.0),
        "placement_4p_sum": placement_4p_sum,
        "survival_time_sum": survival_time_sum,
        "score_share_sum": score_share_sum,
        "decision_count": decision_count,
        "noop_count": noop_count,
        "friendly_target_count": zero,
        "enemy_target_count": zero,
        "neutral_target_count": zero,
        "win_rate_2p": jnp.where(episodes_2p > 0.0, first_place_sum / episodes_2p, 0.0),
        "first_place_rate_4p": jnp.where(
            episodes_4p > 0.0, first_place_sum / episodes_4p, 0.0
        ),
        "average_placement_4p": jnp.where(
            episodes_4p > 0.0, placement_4p_sum / episodes_4p, 0.0
        ),
        "survival_time": jnp.where(
            episode_done > 0.0, survival_time_sum / episode_done, 0.0
        ),
        "score_share": jnp.where(
            episode_done > 0.0, score_share_sum / episode_done, 0.0
        ),
        "noop_percent": jnp.where(
            decision_count > 0.0, (noop_count / decision_count) * 100.0, 0.0
        ),
        "friendly_target_percent": zero,
        "enemy_target_percent": zero,
        "neutral_target_percent": zero,
        "overall_win_rate": jnp.where(
            episode_done > 0.0, first_place_sum / episode_done, 0.0
        ),
        "opponent_slots_total": zero,
        "opponent_slots_latest": zero,
        "opponent_slots_historical": zero,
        "opponent_slots_random": zero,
        "opponent_slots_noop": zero,
        "opponent_slots_nearest_sniper": zero,
        "opponent_slots_turtle": zero,
        "opponent_slots_opportunistic": zero,
        "opponent_historical_fallback_latest_slots": zero,
        "opponent_current_slots": zero,
        "opponent_random_slots": zero,
        "opponent_snapshot_slots": zero,
        "won_non_noop_actions_per_step": zero,
        "lost_non_noop_actions_per_step": zero,
        "won_avg_fleet_launch_size": zero,
        "lost_avg_fleet_launch_size": zero,
        "won_avg_planets_owned": zero,
        "lost_avg_planets_owned": zero,
        "won_avg_planets_lost": zero,
        "lost_avg_planets_lost": zero,
        "won_avg_planets_taken": zero,
        "lost_avg_planets_taken": zero,
        "won_avg_garrisoned_ships_per_planet": zero,
        "lost_avg_garrisoned_ships_per_planet": zero,
        "won_avg_planet_diff": zero,
        "lost_avg_planet_diff": zero,
        "won_avg_production_diff": zero,
        "lost_avg_production_diff": zero,
        "won_avg_launch_fleet_speed": zero,
        "lost_avg_launch_fleet_speed": zero,
    }


def concatenate_transition_batches(
    batches: tuple[JaxTransitionBatch, ...] | list[JaxTransitionBatch],
) -> JaxTransitionBatch:
    """Concatenate compatible rollout batches along the environment axis.

    Mixed-format JAX training uses one compiled collector per static player
    count. The resulting transition tensors share rollout and feature shapes,
    so PPO can consume a single larger batch by joining their independent
    environment axes.
    """

    if not batches:
        raise ValueError("At least one transition batch is required.")
    if len(batches) == 1:
        return batches[0]
    reference_shape = batches[0].self_features.shape
    for batch in batches[1:]:
        if (
            batch.self_features.shape[0] != reference_shape[0]
            or batch.self_features.shape[2:] != reference_shape[2:]
        ):
            raise ValueError(
                "Transition batches must share rollout and feature dimensions "
                "to concatenate along the environment axis."
            )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *batches)


def discounted_returns(rewards: jax.Array, done: jax.Array, gamma: float) -> jax.Array:
    """Compute discounted returns over rollout time with terminal resets."""

    def step(carry, item):
        reward, terminal = item
        carry = reward + gamma * carry * (1.0 - terminal.astype(jnp.float32))
        return carry, carry

    _, out = jax.lax.scan(
        step, jnp.zeros_like(rewards[-1]), (rewards, done), reverse=True
    )
    return out


def ppo_update_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    """Apply one PPO epoch using memory-bounded minibatches.

    Rollouts can be large when benchmarking long attention-policy runs. Running
    the policy over every rollout row in a single XLA program forces the GPU to
    materialize attention intermediates for the entire rollout at once. Instead,
    flatten once, pad to static memory chunks, and scan sequential optimizer
    steps over those chunks. The chunk size honors large configured minibatches
    and the ``training.update_chunk_rows_min``/``training.update_chunk_rows_max``
    limits so rollouts can trade memory pressure for throughput.
    """

    sequence_k = batch.target_index.shape[-1]
    mask = batch.decision_mask.reshape(-1, sequence_k).astype(jnp.float32)
    self_features = batch.self_features.reshape(-1, self_feature_dim(cfg.task))
    candidate_features = batch.candidate_features.reshape(
        -1, cfg.task.candidate_count, candidate_feature_dim(cfg.task)
    )
    global_features = batch.global_features.reshape(-1, global_feature_dim(cfg.task))
    candidate_mask = batch.candidate_mask.reshape(-1, cfg.task.candidate_count)
    player_count = batch.player_count.reshape(-1)
    ship_bucket_mask = batch.ship_bucket_mask.reshape(
        -1, sequence_k, cfg.task.candidate_count, cfg.task.ship_bucket_count
    )
    target = batch.target_index.reshape(-1, sequence_k)
    bucket = batch.ship_bucket.reshape(-1, sequence_k)
    old_log_prob = batch.log_prob.reshape(-1, sequence_k)
    returns = batch.returns.reshape(-1, sequence_k)
    advantages = batch.advantages.reshape(-1, sequence_k)
    advantage_mean = masked_mean(advantages, mask)
    advantages = (advantages - advantage_mean) / jnp.sqrt(
        masked_mean((advantages - advantage_mean) ** 2, mask) + 1e-8
    )

    total_rows = mask.shape[0]
    min_chunk_rows = int(cfg.training.update_chunk_rows_min)
    max_chunk_rows = (
        int(cfg.training.update_chunk_rows_max)
        if cfg.training.update_chunk_rows_max is not None
        else total_rows
    )
    chunk_target = max(int(cfg.training.minibatch_size), min_chunk_rows)
    minibatch_size = min(max(chunk_target, 1), max_chunk_rows, total_rows)
    minibatch_count = (total_rows + minibatch_size - 1) // minibatch_size
    minibatches = {
        "mask": _reshape_minibatches(mask, minibatch_count, minibatch_size, 0.0),
        "self_features": _reshape_minibatches(
            self_features, minibatch_count, minibatch_size, 0.0
        ),
        "candidate_features": _reshape_minibatches(
            candidate_features, minibatch_count, minibatch_size, 0.0
        ),
        "global_features": _reshape_minibatches(
            global_features, minibatch_count, minibatch_size, 0.0
        ),
        "candidate_mask": _reshape_minibatches(
            candidate_mask, minibatch_count, minibatch_size, False
        ),
        "player_count": _reshape_minibatches(
            player_count, minibatch_count, minibatch_size, 0
        ),
        "ship_bucket_mask": _reshape_minibatches(
            ship_bucket_mask, minibatch_count, minibatch_size, False
        ),
        "target": _reshape_minibatches(target, minibatch_count, minibatch_size, 0),
        "bucket": _reshape_minibatches(bucket, minibatch_count, minibatch_size, 0),
        "old_log_prob": _reshape_minibatches(
            old_log_prob, minibatch_count, minibatch_size, 0.0
        ),
        "returns": _reshape_minibatches(returns, minibatch_count, minibatch_size, 0.0),
        "advantages": _reshape_minibatches(
            advantages, minibatch_count, minibatch_size, 0.0
        ),
    }

    def update_minibatch(carry, minibatch):
        params, opt_state = carry

        def loss_fn(params):
            output = policy.apply(
                params,
                minibatch["self_features"],
                minibatch["candidate_features"],
                minibatch["global_features"],
                minibatch["candidate_mask"],
                player_count=minibatch["player_count"],
                target_sequence=minibatch["target"],
            )
            output = mask_policy_output_for_shield(
                output,
                minibatch["candidate_mask"],
                cfg.task.ship_bucket_count,
                minibatch["ship_bucket_mask"],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                output, minibatch["target"], minibatch["bucket"]
            )
            approx_kl = masked_mean(
                minibatch["old_log_prob"] - new_log_prob,
                minibatch["mask"],
            )
            ratio = jnp.exp(new_log_prob - minibatch["old_log_prob"])
            clipped_ratio = jnp.clip(
                ratio, 1.0 - cfg.training.clip_coef, 1.0 + cfg.training.clip_coef
            )
            policy_objective = jnp.minimum(
                minibatch["advantages"] * ratio,
                minibatch["advantages"] * clipped_ratio,
            )
            value_error = (minibatch["returns"] - output.value[:, None]) ** 2
            policy_loss = -masked_mean(policy_objective, minibatch["mask"])
            value_loss = 0.5 * masked_mean(value_error, minibatch["mask"])
            entropy_loss = masked_mean(entropy, minibatch["mask"])
            loss = (
                policy_loss
                + cfg.training.vf_coef * value_loss
                - cfg.training.ent_coef * entropy_loss
            )
            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_loss,
                "approx_kl": approx_kl,
                "loss": loss,
                "sample_count": minibatch["mask"].sum(),
            }
            for player_count in (2, 4):
                suffix = f"{player_count}p"
                format_mask = minibatch["mask"] * (
                    minibatch["player_count"][:, None] == player_count
                ).astype(jnp.float32)
                format_policy_loss = -masked_mean(policy_objective, format_mask)
                format_value_loss = 0.5 * masked_mean(value_error, format_mask)
                format_entropy = masked_mean(entropy, format_mask)
                format_approx_kl = masked_mean(
                    minibatch["old_log_prob"] - new_log_prob,
                    format_mask,
                )
                format_total_loss = (
                    format_policy_loss
                    + cfg.training.vf_coef * format_value_loss
                    - cfg.training.ent_coef * format_entropy
                )
                metrics[f"policy_loss_{suffix}"] = format_policy_loss
                metrics[f"value_loss_{suffix}"] = format_value_loss
                metrics[f"entropy_{suffix}"] = format_entropy
                metrics[f"approx_kl_{suffix}"] = format_approx_kl
                metrics[f"total_loss_{suffix}"] = format_total_loss
                metrics[f"loss_sample_count_{suffix}"] = format_mask.sum()
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = loss
        return (params, opt_state), metrics

    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    format_metric_names = frozenset(
        f"{metric_name}_{suffix}"
        for suffix in ("2p", "4p")
        for metric_name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "total_loss",
        )
    )
    format_sample_names = frozenset(
        f"loss_sample_count_{suffix}" for suffix in ("2p", "4p")
    )
    metric_weights = jnp.where(metrics_by_minibatch["sample_count"] > 0.0, 1.0, 0.0)
    metric_denominator = jnp.maximum(metric_weights.sum(), 1.0)
    metrics = {
        name: (values * metric_weights).sum() / metric_denominator
        for name, values in metrics_by_minibatch.items()
        if name not in {"sample_count", *format_metric_names, *format_sample_names}
    }
    for suffix in ("2p", "4p"):
        sample_name = f"loss_sample_count_{suffix}"
        sample_counts = metrics_by_minibatch[sample_name]
        sample_denominator = jnp.maximum(sample_counts.sum(), 1.0)
        metrics[sample_name] = sample_counts.sum()
        for metric_name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "total_loss",
        ):
            name = f"{metric_name}_{suffix}"
            metrics[name] = (
                metrics_by_minibatch[name] * sample_counts
            ).sum() / sample_denominator
    metrics["minibatches"] = jnp.array(minibatch_count, dtype=jnp.float32)
    return (
        JaxTrainState(
            params=params, opt_state=opt_state, optimizer=train_state.optimizer
        ),
        metrics,
    )


def _reshape_minibatches(
    value: jax.Array,
    minibatch_count: int,
    minibatch_size: int,
    padding_value: float | int | bool,
) -> jax.Array:
    """Pad and reshape a flat leading axis into static minibatches."""

    padded_rows = minibatch_count * minibatch_size
    pad_rows = padded_rows - value.shape[0]
    pad_width = [(0, pad_rows)] + [(0, 0)] * (value.ndim - 1)
    padded = jnp.pad(value, pad_width, constant_values=padding_value)
    return padded.reshape((minibatch_count, minibatch_size) + value.shape[1:])


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Return the mean of ``x`` over entries where ``mask`` is non-zero."""

    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)
