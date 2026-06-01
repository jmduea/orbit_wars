from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.env import batched_reset
from src.jax.launch_hygiene import compose_hygiene_with_shield_mask


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(**kwargs) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.max_moves_k = 3
    cfg.task = _task_cfg(**kwargs.pop("task", {}))
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.jax
def test_empty_prefix_leaves_shield_mask_unchanged() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)
    source = jnp.zeros((1, 3), dtype=jnp.int32)
    slot = jnp.zeros((1, 3), dtype=jnp.int32)
    stop = jnp.zeros((1, 3), dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.ones((1, 3), dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=0,
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(shield[0]))


@pytest.mark.jax
def test_duplicate_source_slot_masked_after_prior_launch() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(1), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)

    src_row, tgt_slot = 1, 2
    source = jnp.array([[src_row, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[tgt_slot, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    assert not bool(np.asarray(out[src_row, tgt_slot, :]).any())


@pytest.mark.jax
def test_hygiene_can_empty_targets_triggering_stop_path() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(2), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.zeros((1, MAX_PLANETS, k, buckets), dtype=bool)
    shield = shield.at[0, 1, 2, 1].set(True)

    source = jnp.array([[1, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[2, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    assert not bool(np.asarray(out).any())
