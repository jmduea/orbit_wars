from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.game.trajectory_shield import (
    ShieldDiagnostics,
    apply_trajectory_shield_to_turn_batch_v2,
    default_edge_action_bucket_mask,
)
from src.jax.env import JaxAction
from src.jax.features import TurnBatch
from src.jax.policy import edge_action_count
from src.jax.rollout.types import ShieldedSequenceSample

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




def noop_edge_index(task_cfg) -> int:
    return MAX_PLANETS * edge_k(task_cfg)


def owned_planet_ships(game) -> jax.Array:
    player = game.player
    if player.ndim > 0:
        player = player[:, None]
    owned = game.planets.active & (game.planets.owner == player)
    return jnp.where(owned, game.planets.ships, 0.0)


def _launch_angle_for_edge(game, batch: TurnBatch, src_row, slot):
    src_x = game.planets.x[src_row]
    src_y = game.planets.y[src_row]
    tgt_id = batch.edge_tgt_ids[src_row, slot]
    match = game.planets.id == tgt_id
    tgt_x = jnp.sum(jnp.where(match, game.planets.x, 0.0))
    tgt_y = jnp.sum(jnp.where(match, game.planets.y, 0.0))
    return jnp.arctan2(tgt_y - src_y, tgt_x - src_x)


def build_action_from_edge_batch(
    game,
    batch: TurnBatch,
    target_index: jax.Array,
    ship_bucket: jax.Array,
    cfg: TrainConfig,
) -> JaxAction:
    env_count = batch.planet_features.shape[0]
    k = edge_k(cfg.task)
    noop_idx = noop_edge_index(cfg.task)
    target_index = target_index.reshape(env_count, -1)
    ship_bucket = ship_bucket.reshape(env_count, -1)
    launch_steps = target_index.shape[-1]
    fleet_slots = cfg.task.max_fleets

    def build_env_action(game_row, batch_row, targets, buckets):
        def step_fn(remaining, step_inputs):
            flat_idx, bucket = step_inputs
            src_row = flat_idx // k
            valid = (flat_idx < noop_idx) & (bucket > 0) & (remaining[src_row] > 0.0)
            requested = ship_count_for_bucket_jax(
                remaining[src_row], bucket, cfg.task.ship_bucket_count
            )
            launched = jnp.where(valid, jnp.minimum(remaining[src_row], requested), 0.0)
            remaining = remaining.at[src_row].set(
                jnp.maximum(remaining[src_row] - launched, 0.0)
            )
            src_id = batch_row.edge_src_ids[src_row]
            angle = _launch_angle_for_edge(game_row, batch_row, src_row, flat_idx % k)
            return remaining, (src_id, angle, launched, valid)

        remaining = owned_planet_ships(game_row)
        _, steps = jax.lax.scan(
            step_fn,
            remaining,
            (jnp.moveaxis(targets, -1, 0), jnp.moveaxis(buckets, -1, 0)),
        )
        source_id, angle, ships, valid = steps
        source_id = jnp.moveaxis(source_id, 0, -1)
        angle = jnp.moveaxis(angle, 0, -1)
        ships = jnp.moveaxis(ships, 0, -1)
        valid = jnp.moveaxis(valid, 0, -1)
        flat_source = source_id.reshape(launch_steps)
        flat_angle = angle.reshape(launch_steps)
        flat_ships = ships.reshape(launch_steps)
        flat_valid = valid.reshape(launch_steps)
        action_width = min(launch_steps, fleet_slots)
        pad = fleet_slots - action_width
        return JaxAction(
            source_id=jnp.pad(flat_source[:action_width], (0, pad), constant_values=-1),
            angle=jnp.pad(flat_angle[:action_width], (0, pad), constant_values=0.0),
            ships=jnp.pad(flat_ships[:action_width], (0, pad), constant_values=0.0),
            valid=jnp.pad(flat_valid[:action_width], (0, pad), constant_values=False),
        )

    return jax.vmap(build_env_action)(game, batch, target_index, ship_bucket)


def build_random_action_from_edge_batch(
    key: jax.Array,
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    env_count = batch.planet_features.shape[0]
    k = edge_k(cfg.task)
    edge_count = edge_action_count(cfg.task)
    key_target, key_bucket = jax.random.split(key)
    flat_mask = jnp.concatenate(
        [batch.edge_mask.reshape(env_count, MAX_PLANETS * k), jnp.ones((env_count, 1), dtype=bool)],
        axis=1,
    )
    if ship_bucket_mask is None:
        flat_bucket_mask = default_edge_action_bucket_mask(flat_mask, cfg.task.ship_bucket_count)
    else:
        flat_bucket_mask = ship_bucket_mask
    real_bucket_mask = flat_bucket_mask & (
        jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)[None, None, :] > 0
    )
    real_edge = (
        flat_mask
        & real_bucket_mask.any(axis=-1)
        & (jnp.arange(edge_count, dtype=jnp.int32)[None, :] < noop_edge_index(cfg.task))
    )
    has_target = real_edge.any(axis=-1)
    target_logits = jnp.where(real_edge, 0.0, jnp.finfo(jnp.float32).min)
    target = jnp.where(
        has_target,
        jax.random.categorical(key_target, target_logits, axis=-1),
        jnp.full((env_count,), noop_edge_index(cfg.task), dtype=jnp.int32),
    )
    selected_bucket_mask = jnp.take_along_axis(
        flat_bucket_mask,
        target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_logits = jnp.where(selected_bucket_mask, 0.0, jnp.finfo(jnp.float32).min)
    bucket = jax.random.categorical(key_bucket, bucket_logits, axis=-1)
    bucket = jnp.where(has_target, bucket, jnp.zeros_like(bucket))
    return build_action_from_edge_batch(
        game, batch, target[:, None], bucket[:, None], cfg
    )



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
    batch: TurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> ShieldedSequenceSample:
    env_count = batch.planet_features.shape[0]
    player_count = jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32)
    probe_output = policy.apply(
        params,
        batch,
        player_count=player_count,
        rng=key,
        deterministic=deterministic,
    )
    sequence_k = probe_output.target_logits.shape[1]
    edge_count = probe_output.target_logits.shape[2]
    noop_idx = noop_edge_index(cfg.task)
    target_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    bucket_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.int32)
    log_prob_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    entropy_sequence = jnp.zeros((env_count, sequence_k), dtype=jnp.float32)
    remaining_ships = owned_planet_ships(game)
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
        (env_count, sequence_k, edge_count, cfg.task.ship_bucket_count),
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
            batch,
            player_count=player_count,
            target_sequence=target_sequence,
            rng=jax.random.fold_in(key, step_idx),
            deterministic=deterministic,
        )
        shielded = jax.vmap(
            lambda game_row, batch_row, ships: apply_trajectory_shield_to_turn_batch_v2(
                game_row, batch_row, cfg.task, remaining_planet_ships=ships
            )
        )(game, batch, remaining_ships)
        step_diagnostics = shielded.diagnostics
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
        edge_action_mask = jnp.concatenate(
            [
                shielded.batch.edge_mask.reshape(env_count, MAX_PLANETS * edge_k(cfg.task)),
                jnp.ones((env_count, 1), dtype=bool),
            ],
            axis=1,
        )
        step_bucket_mask = shielded.ship_bucket_mask.reshape(
            env_count, edge_count, cfg.task.ship_bucket_count
        )
        env_active = jnp.ones((env_count,), dtype=bool)
        step_bucket_mask = _ensure_bucket_mask_has_choice(step_bucket_mask.reshape(-1, edge_count, cfg.task.ship_bucket_count), env_active)
        step_bucket_mask = step_bucket_mask.reshape(
            env_count, edge_count, cfg.task.ship_bucket_count
        )
        target, bucket, log_prob, entropy = _sample_step_from_logits(
            key=jax.random.fold_in(key, 10_000 + step_idx),
            target_logits=step_output.target_logits[:, step_idx, :],
            ship_logits=step_output.ship_logits[:, step_idx, :, :],
            ship_bucket_mask=step_bucket_mask,
            deterministic=deterministic,
        )
        src_rows = target // edge_k(cfg.task)
        current_source_ships = remaining_ships[jnp.arange(env_count), src_rows]
        launched = ship_count_for_bucket_jax(
            current_source_ships,
            bucket,
            cfg.task.ship_bucket_count,
        )
        launch_valid = (target < noop_idx) & (bucket > 0) & (launched > 0.0)
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
            _remaining_ships,
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


def _edge_scripted_context(
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None,
):
    from src.features.registry import EDGE_FEATURE_SCHEMA, edge_k

    env_count = batch.planet_features.shape[0]
    k = edge_k(cfg.task)
    flat_count = MAX_PLANETS * k
    flat_mask = batch.edge_mask.reshape(env_count, flat_count)
    owner_slice = EDGE_FEATURE_SCHEMA.slice("target_owner_slot")
    owner_slots = batch.edge_features[..., owner_slice].reshape(env_count, flat_count, 4)
    distance = batch.edge_features[..., EDGE_FEATURE_SCHEMA.slice("distance")].reshape(
        env_count, flat_count
    )
    if ship_bucket_mask is None:
        full_mask = jnp.concatenate(
            [flat_mask, jnp.ones((env_count, 1), dtype=bool)], axis=1
        )
        bucket_mask = default_edge_action_bucket_mask(
            full_mask, cfg.task.ship_bucket_count
        )
    else:
        bucket_mask = ship_bucket_mask
    real_bucket = bucket_mask[:, :flat_count, :][..., 1:].any(axis=-1)
    valid = flat_mask & real_bucket
    owner_sum = owner_slots.sum(axis=-1)
    neutral = valid & (owner_sum < 0.5)
    enemy = valid & (owner_sum > 0.5) & (owner_slots[..., 0] < 0.5)
    return {
        "env_count": env_count,
        "flat_count": flat_count,
        "valid": valid,
        "neutral": neutral,
        "enemy": enemy,
        "distance": distance,
        "bucket_mask": bucket_mask,
    }


def _bucket_for_flat_target(
    flat_target: jax.Array,
    bucket_mask: jax.Array,
    has_target: jax.Array,
    cfg: TrainConfig,
    *,
    conservative: bool,
) -> jax.Array:
    selected_bucket_mask = jnp.take_along_axis(
        bucket_mask,
        flat_target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_ids = jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)
    if conservative:
        nonzero = selected_bucket_mask & (bucket_ids[None, :] > 0)
        bucket = jnp.argmax(nonzero.astype(jnp.int32), axis=-1)
        bucket = jnp.where(has_target & nonzero.any(axis=-1), bucket, 0)
    else:
        bucket = jnp.max(jnp.where(selected_bucket_mask, bucket_ids[None, :], 0), axis=-1)
        bucket = jnp.where(has_target, bucket, 0)
    return bucket


def _build_scripted_edge_action(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    pick_mask: jax.Array,
    distance: jax.Array,
    bucket_mask: jax.Array,
    use_nearest: bool,
    conservative_bucket: bool,
) -> JaxAction:
    noop_idx = noop_edge_index(cfg.task)
    if use_nearest:
        masked_distance = jnp.where(pick_mask, distance, jnp.inf)
        flat_target = jnp.argmin(masked_distance, axis=-1)
    else:
        flat_target = jnp.argmax(pick_mask.astype(jnp.int32), axis=-1)
    has_target = pick_mask.any(axis=-1)
    bucket = _bucket_for_flat_target(
        flat_target,
        bucket_mask,
        has_target,
        cfg,
        conservative=conservative_bucket,
    )
    target = jnp.where(has_target, flat_target, noop_idx)
    bucket = jnp.where(has_target, bucket, 0)
    return build_action_from_edge_batch(
        game, batch, target[:, None], bucket[:, None], cfg
    )


def build_sniper_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    ctx = _edge_scripted_context(batch, cfg, ship_bucket_mask)
    return _build_scripted_edge_action(
        game,
        batch,
        cfg,
        pick_mask=ctx["valid"],
        distance=ctx["distance"],
        bucket_mask=ctx["bucket_mask"],
        use_nearest=True,
        conservative_bucket=False,
    )


def build_turtle_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    ctx = _edge_scripted_context(batch, cfg, ship_bucket_mask)
    return _build_scripted_edge_action(
        game,
        batch,
        cfg,
        pick_mask=ctx["neutral"],
        distance=ctx["distance"],
        bucket_mask=ctx["bucket_mask"],
        use_nearest=False,
        conservative_bucket=True,
    )


def build_opportunistic_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    ctx = _edge_scripted_context(batch, cfg, ship_bucket_mask)
    return _build_scripted_edge_action(
        game,
        batch,
        cfg,
        pick_mask=ctx["enemy"],
        distance=ctx["distance"],
        bucket_mask=ctx["bucket_mask"],
        use_nearest=False,
        conservative_bucket=False,
    )

def build_noop_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
) -> JaxAction:
    """Build a pass/no-op action that launches no fleets."""

    env_count = batch.planet_features.shape[0]
    noop_idx = noop_edge_index(cfg.task)
    noop_target = jnp.full((env_count, 1), noop_idx, dtype=jnp.int32)
    noop_bucket = jnp.zeros((env_count, 1), dtype=jnp.int32)
    return build_action_from_edge_batch(game, batch, noop_target, noop_bucket, cfg)


def _sample_policy_action_with_params(
    key: jax.Array,
    game,
    batch: TurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    sample = _sample_shielded_sequence_with_params(
        key,
        game,
        batch,
        params,
        policy,
        cfg,
        deterministic=deterministic,
    )
    return build_action_from_edge_batch(
        game, batch, sample.target_index, sample.ship_bucket, cfg
    )


def _sample_policy_action(
    key: jax.Array,
    game,
    batch: TurnBatch,
    train_state,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    return _sample_policy_action_with_params(
        key,
        game,
        batch,
        train_state.params,
        policy,
        cfg,
        deterministic=deterministic,
    )

