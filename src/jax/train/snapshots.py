from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.train.checkpoint import HistoricalSnapshotPool


def init_historical_snapshot_pool(
    params: dict, pool_size: int
) -> HistoricalSnapshotPool:
    capacity = max(int(pool_size), 1)
    stacked_params = jax.tree.map(
        lambda value: jnp.broadcast_to(
            jnp.asarray(value)[None, ...], (capacity,) + jnp.asarray(value).shape
        ),
        params,
    )
    return HistoricalSnapshotPool(
        params=stacked_params,
        snapshot_ids=jnp.zeros((capacity,), dtype=jnp.int32),
        snapshot_updates=jnp.zeros((capacity,), dtype=jnp.int32),
        valid_mask=jnp.zeros((capacity,), dtype=bool),
    )


def add_historical_snapshot(
    pool: HistoricalSnapshotPool, params: dict, *, update: int
) -> tuple[HistoricalSnapshotPool, dict[str, object]]:
    slot = int(pool.next_slot)
    snapshot_id = int(pool.next_id)
    new_params = jax.tree.map(
        lambda bank, value: bank.at[slot].set(value), pool.params, params
    )
    was_valid = bool(jax.device_get(pool.valid_mask[slot]))
    next_pool = HistoricalSnapshotPool(
        params=new_params,
        snapshot_ids=pool.snapshot_ids.at[slot].set(snapshot_id),
        snapshot_updates=pool.snapshot_updates.at[slot].set(int(update)),
        valid_mask=pool.valid_mask.at[slot].set(True),
        next_slot=(slot + 1) % int(pool.valid_mask.shape[0]),
        next_id=snapshot_id + 1,
    )
    event = {
        "event": "historical_snapshot_added",
        "update": int(update),
        "snapshot_id": snapshot_id,
        "snapshot_slot": slot,
        "historical_snapshot_evicted": was_valid,
    }
    return next_pool, event


def snapshot_due(cfg: TrainConfig, update: int) -> bool:
    if not cfg.curriculum.enabled:
        return False
    interval = int(cfg.opponents.snapshot.interval_updates)
    return interval > 0 and update % interval == 0
