"""Launch hygiene masks for factored K-step action sampling and PPO replay.

After each active launch in a turn, cumulatively mask:
- duplicate ``(source_row, target_slot)`` pairs (R1–R3), and
- friendly reverse edges ``B→A`` when ``A→B`` was already chosen (R4–R6).

Masks apply to ``ship_bucket_mask`` before ``source_mask_from_bucket_mask_and_ships``.

Scope (R10): applies to learner rollout, PPO replay, submission/eval inference, and
neural opponents using the factorized K-step decoder. Heuristic single-step
edge-batch opponents (random, turtle, sniper, etc.) are out of scope.
"""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry import PLANET_FEATURE_SCHEMA
from src.jax.features import TurnBatch
from src.jax.ship_action import is_continuous_ship_mode


def _launch_valid_at_step(
    *,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_bucket: jax.Array,
    ship_fraction: jax.Array | None,
    cfg: TrainConfig,
    step_idx: int,
) -> jax.Array:
    """Whether step ``step_idx`` is an active launch (matches action_sampling)."""
    if is_continuous_ship_mode(cfg):
        if ship_fraction is None:
            raise ValueError(
                "ship_fraction is required when ship_action_mode is continuous_fraction"
            )
        return (
            (stop_flag[:, step_idx] < 0.5)
            & (step_mask[:, step_idx] > 0.5)
            & (ship_fraction[:, step_idx] > 0.0)
        )
    return (
        (stop_flag[:, step_idx] < 0.5)
        & (step_mask[:, step_idx] > 0.5)
        & (ship_bucket[:, step_idx] > 0)
    )


def _planet_id_to_row(batch: TurnBatch, planet_id: jax.Array) -> jax.Array:
    """Map planet id to source row index; ``num_planets`` when not found."""

    num_planets = batch.planet_mask.shape[-1]
    matches = batch.edge_src_ids == planet_id[:, None]
    sentinel = jnp.full((1, num_planets), num_planets, dtype=jnp.int32)
    rows = jnp.arange(num_planets, dtype=jnp.int32)[None, :]
    return jnp.where(matches, rows, sentinel).min(axis=-1)


def _slots_matching_target_on_row(
    batch: TurnBatch,
    src_row: jax.Array,
    tgt_planet_id: jax.Array,
) -> jax.Array:
    """Per-env slot mask where ``src_row`` edges target ``tgt_planet_id``."""

    env_count = tgt_planet_id.shape[0]
    batch_idx = jnp.arange(env_count, dtype=jnp.int32)
    tgt_ids_at_row = batch.edge_tgt_ids[batch_idx, src_row, :]
    return tgt_ids_at_row == tgt_planet_id[:, None]


def _owner_is_learner_pov(batch: TurnBatch, planet_id: jax.Array) -> jax.Array:
    """True when ``planet_id`` is learner-owned in turn-start ``TurnBatch`` features."""

    env_count = planet_id.shape[0]
    num_planets = batch.planet_mask.shape[-1]
    batch_idx = jnp.arange(env_count, dtype=jnp.int32)
    row = _planet_id_to_row(batch, planet_id)
    valid_row = row < num_planets
    owner_slice = PLANET_FEATURE_SCHEMA.base_slice("owner_slot")
    owner_slot = batch.planet_features[batch_idx, row, owner_slice]
    is_self = owner_slot[..., 0] > 0.5
    return valid_row & is_self & batch.planet_mask[batch_idx, row]


def _apply_one_launch_hygiene(
    bucket_mask: jax.Array,
    *,
    batch: TurnBatch,
    src_row: jax.Array,
    slot: jax.Array,
    active: jax.Array,
) -> jax.Array:
    """Mask duplicate pair and friendly reverse for one prefix launch."""
    env_count = batch.planet_features.shape[0]
    num_planets = bucket_mask.shape[-3]
    max_k = bucket_mask.shape[-2]
    batch_idx = jnp.arange(env_count, dtype=jnp.int32)

    src_id = batch.edge_src_ids[batch_idx, src_row]
    tgt_id = batch.edge_tgt_ids[batch_idx, src_row, slot]

    dup_row_mask = (
        jnp.arange(num_planets, dtype=jnp.int32)[None, :, None]
        == src_row[:, None, None]
    )
    dup_slot_mask = (
        jnp.arange(max_k, dtype=jnp.int32)[None, None, :] == slot[:, None, None]
    )
    dup_mask = dup_row_mask & dup_slot_mask
    bucket_mask = jnp.where(
        active[:, None, None, None],
        bucket_mask & ~dup_mask[..., None],
        bucket_mask,
    )

    rev_src_row = _planet_id_to_row(batch, tgt_id)
    rev_slot_match = _slots_matching_target_on_row(batch, rev_src_row, src_id)
    rev_valid = (rev_src_row < num_planets) & rev_slot_match.any(axis=-1)

    friendly = _owner_is_learner_pov(batch, src_id) & _owner_is_learner_pov(
        batch, tgt_id
    )
    apply_rev = active & rev_valid & friendly

    rev_row_mask = (
        jnp.arange(num_planets, dtype=jnp.int32)[None, :, None]
        == rev_src_row[:, None, None]
    )
    rev_mask = rev_row_mask & rev_slot_match[:, None, :]
    return jnp.where(
        apply_rev[:, None, None, None],
        bucket_mask & ~rev_mask[..., None],
        bucket_mask,
    )


def hygiene_adjusted_bucket_mask_at_step(
    batch: TurnBatch,
    shield_bucket_mask: jax.Array,
    *,
    source_sequence: jax.Array,
    target_slot_sequence: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_bucket: jax.Array,
    ship_fraction: jax.Array | None,
    cfg: TrainConfig,
    step_idx: int,
) -> jax.Array:
    """Apply cumulative launch-hygiene to shield ``ship_bucket_mask`` at ``step_idx``."""

    squeeze_env = shield_bucket_mask.ndim == 3
    if squeeze_env:
        shield_bucket_mask = shield_bucket_mask[None, ...]

    def body_fn(i: int, mask: jax.Array) -> jax.Array:
        active = _launch_valid_at_step(
            stop_flag=stop_flag,
            step_mask=step_mask,
            ship_bucket=ship_bucket,
            ship_fraction=ship_fraction,
            cfg=cfg,
            step_idx=i,
        )
        return _apply_one_launch_hygiene(
            mask,
            batch=batch,
            src_row=source_sequence[:, i],
            slot=target_slot_sequence[:, i],
            active=active,
        )

    out = jax.lax.fori_loop(0, step_idx, body_fn, shield_bucket_mask)
    if squeeze_env:
        return out[0]
    return out


def compose_hygiene_with_shield_mask(
    batch: TurnBatch,
    shield_bucket_mask: jax.Array,
    *,
    source_sequence: jax.Array,
    target_slot_sequence: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_bucket: jax.Array,
    ship_fraction: jax.Array | None,
    cfg: TrainConfig,
    step_idx: int,
) -> jax.Array:
    """Shield mask + prefix-derived hygiene for one K-step index."""
    return hygiene_adjusted_bucket_mask_at_step(
        batch,
        shield_bucket_mask,
        source_sequence=source_sequence,
        target_slot_sequence=target_slot_sequence,
        stop_flag=stop_flag,
        step_mask=step_mask,
        ship_bucket=ship_bucket,
        ship_fraction=ship_fraction,
        cfg=cfg,
        step_idx=step_idx,
    )
