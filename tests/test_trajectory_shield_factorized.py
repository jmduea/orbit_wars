from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax
from src.config import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.game.trajectory_shield import (
    apply_trajectory_shield_factorized_topk,
    factorized_source_mask_from_shield,
)
from src.jax.env import reset


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def test_factorized_shield_bucket_mask_shape() -> None:
    cfg = _task_cfg()
    k = edge_k(cfg)
    state, batch = reset(jax.random.PRNGKey(0), cfg)
    shielded = apply_trajectory_shield_factorized_topk(
        state.game, batch, cfg, remaining_planet_ships=state.game.planets.ships
    )

    assert shielded.ship_bucket_mask.shape == (MAX_PLANETS, k, cfg.ship_bucket_count)
    assert shielded.batch.edge_mask.shape == batch.edge_mask.shape


def test_factorized_shield_disabled_returns_all_legal() -> None:
    cfg = _task_cfg(trajectory_shield_enabled=False)
    k = edge_k(cfg)
    state, batch = reset(jax.random.PRNGKey(1), cfg)
    shielded = apply_trajectory_shield_factorized_topk(state.game, batch, cfg)

    assert bool(np.asarray(shielded.ship_bucket_mask[..., 0]).all())
    assert bool(np.asarray(shielded.ship_bucket_mask[..., 1:]).any())
    assert shielded.ship_bucket_mask.shape == (MAX_PLANETS, k, cfg.ship_bucket_count)


def test_factorized_source_mask_requires_ships_and_buckets() -> None:
    cfg = _task_cfg(trajectory_shield_enabled=False)
    state, batch = reset(jax.random.PRNGKey(2), cfg)
    shielded = apply_trajectory_shield_factorized_topk(state.game, batch, cfg)
    planet_ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    source_mask = factorized_source_mask_from_shield(
        shielded.batch.edge_mask,
        shielded.ship_bucket_mask,
        planet_ships,
    )
    np.testing.assert_array_equal(np.asarray(source_mask), False)

    source_mask_with_ships = factorized_source_mask_from_shield(
        shielded.batch.edge_mask,
        shielded.ship_bucket_mask,
        state.game.planets.ships,
    )
    assert bool(np.asarray(source_mask_with_ships).any())
