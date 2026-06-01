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

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry import PLANET_FEATURE_SCHEMA
from src.jax.features import TurnBatch
from src.jax.ship_action import is_continuous_ship_mode


class HygieneLookups(NamedTuple):
    """Turn-static tables for incremental hygiene updates."""

    planet_id_to_row: jax.Array
    learner_owned_at_row: jax.Array


def build_hygiene_lookups(batch: TurnBatch) -> HygieneLookups:
    """Precompute planet-id → row and learner-ownership flags for one turn."""

    env_count = batch.edge_src_ids.shape[0]
    num_planets = batch.planet_mask.shape[-1]
    sentinel = jnp.int32(num_planets)
    table = jnp.full((env_count, num_planets + 1), sentinel, dtype=jnp.int32)
    env_idx = jnp.arange(env_count, dtype=jnp.int32)[:, None]
    row_idx = jnp.arange(num_planets, dtype=jnp.int32)[None, :]
    table = table.at[env_idx, batch.edge_src_ids].set(row_idx)

    owner_slice = PLANET_FEATURE_SCHEMA.base_slice("owner_slot")
    is_self = batch.planet_features[..., owner_slice][..., 0] > 0.5
    learner_owned = is_self & batch.planet_mask
    return HygieneLookups(
        planet_id_to_row=table,
        learner_owned_at_row=learner_owned,
    )


def empty_cumulative_forbidden(
    env_count: int,
    *,
    num_planets: int,
    max_k: int,
    buckets: int,
) -> jax.Array:
    """Zero cumulative forbidden mask for scan carry initialization."""

    return jnp.zeros((env_count, num_planets, max_k, buckets), dtype=jnp.bool_)


def apply_cumulative_forbidden_to_shield(
    shield_bucket_mask: jax.Array,
    cumulative_forbidden: jax.Array,
) -> jax.Array:
    """Compose shield legality with prefix-derived forbidden cells."""

    return shield_bucket_mask & ~cumulative_forbidden


def launch_valid_at_step(
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


def _row_for_planet_id(lookups: HygieneLookups, planet_id: jax.Array) -> jax.Array:
    env_count = planet_id.shape[0]
    batch_idx = jnp.arange(env_count, dtype=jnp.int32)
    pid = jnp.clip(planet_id, 0, lookups.planet_id_to_row.shape[-1] - 1)
    return lookups.planet_id_to_row[batch_idx, pid]


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


def _forbidden_cells_for_launch(
    shape: tuple[int, ...],
    *,
    batch: TurnBatch,
    lookups: HygieneLookups,
    src_row: jax.Array,
    slot: jax.Array,
) -> jax.Array:
    """Bool mask of bucket cells newly forbidden by one launch."""

    env_count = shape[0]
    num_planets = shape[1]
    max_k = shape[2]
    batch_idx = jnp.arange(env_count, dtype=jnp.int32)

    src_id = batch.edge_src_ids[batch_idx, src_row]
    tgt_id = batch.edge_tgt_ids[batch_idx, src_row, slot]

    forbidden = jnp.zeros(shape, dtype=jnp.bool_)
    forbidden = forbidden.at[batch_idx, src_row, slot, :].set(True)

    rev_row = _row_for_planet_id(lookups, tgt_id)
    rev_valid = rev_row < num_planets
    rev_slot_match = _slots_matching_target_on_row(batch, rev_row, src_id)
    tgt_row = _row_for_planet_id(lookups, tgt_id)
    friendly = (
        lookups.learner_owned_at_row[batch_idx, src_row]
        & rev_valid
        & lookups.learner_owned_at_row[batch_idx, tgt_row]
    )
    apply_rev = rev_valid & rev_slot_match.any(axis=-1) & friendly

    def mark_rev_slot(i: int, mask: jax.Array) -> jax.Array:
        slot_active = apply_rev & rev_slot_match[:, i]
        return mask.at[batch_idx, rev_row, i, :].set(
            jnp.where(slot_active[:, None], True, mask[batch_idx, rev_row, i, :])
        )

    return jax.lax.fori_loop(0, max_k, mark_rev_slot, forbidden)


def apply_launch_to_cumulative_forbidden(
    cumulative_forbidden: jax.Array,
    *,
    batch: TurnBatch,
    lookups: HygieneLookups,
    src_row: jax.Array,
    slot: jax.Array,
    active: jax.Array,
) -> jax.Array:
    """Mark duplicate and friendly-reverse cells forbidden after one launch."""

    delta = _forbidden_cells_for_launch(
        cumulative_forbidden.shape,
        batch=batch,
        lookups=lookups,
        src_row=src_row,
        slot=slot,
    )
    return jnp.where(
        active[:, None, None, None], cumulative_forbidden | delta, cumulative_forbidden
    )


def _apply_one_launch_hygiene(
    bucket_mask: jax.Array,
    *,
    batch: TurnBatch,
    lookups: HygieneLookups,
    src_row: jax.Array,
    slot: jax.Array,
    active: jax.Array,
) -> jax.Array:
    """Mask duplicate pair and friendly reverse for one prefix launch."""

    delta = _forbidden_cells_for_launch(
        bucket_mask.shape,
        batch=batch,
        lookups=lookups,
        src_row=src_row,
        slot=slot,
    )
    return jnp.where(
        active[:, None, None, None],
        bucket_mask & ~delta,
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
    lookups: HygieneLookups | None = None,
) -> jax.Array:
    """Apply cumulative launch-hygiene to shield ``ship_bucket_mask`` at ``step_idx``."""

    if lookups is None:
        lookups = build_hygiene_lookups(batch)

    squeeze_env = shield_bucket_mask.ndim == 3
    if squeeze_env:
        shield_bucket_mask = shield_bucket_mask[None, ...]

    def body_fn(i: int, mask: jax.Array) -> jax.Array:
        active = launch_valid_at_step(
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
            lookups=lookups,
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
    lookups: HygieneLookups | None = None,
) -> jax.Array:
    """Shield mask + prefix-derived hygiene for one K-step index (oracle path)."""
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
        lookups=lookups,
    )


def cumulative_forbidden_matches_oracle(
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
    cumulative_forbidden: jax.Array,
    lookups: HygieneLookups | None = None,
) -> jax.Array:
    """True per-env when carry forbidden matches prefix oracle at ``step_idx``."""

    oracle = compose_hygiene_with_shield_mask(
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
        lookups=lookups,
    )
    carry = apply_cumulative_forbidden_to_shield(
        shield_bucket_mask, cumulative_forbidden
    )
    if oracle.ndim == 3:
        oracle = oracle[None, ...]
    if carry.ndim == 3:
        carry = carry[None, ...]
    return jnp.all(oracle == carry, axis=(-3, -2, -1))
