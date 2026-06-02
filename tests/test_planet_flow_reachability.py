from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import TrainConfig
from src.game.constants import MAX_PLANETS
from src.jax.env import reset
from src.jax.planet_flow import (
    catalog_target_reachability,
    planet_flow_sampling_target_mask,
)


def _cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.task.max_fleets = 8
    cfg.task.candidate_count = 4
    return cfg


def _batched_reset(cfg: TrainConfig):
    state, batch = reset(jax.random.PRNGKey(0), cfg.task)
    batched_game = jax.tree.map(lambda x: x[None, ...], state.game)
    batched_batch = jax.tree.map(lambda x: x[None, ...], batch)
    return state.game, batch, batched_game, batched_batch


def test_catalog_reachability_marks_off_catalog_enemy_unreachable() -> None:
    cfg = _cfg()
    _game, _batch, batched_game, batched_batch = _batched_reset(cfg)
    src_row = 0
    neutral_row = 1
    enemy_row = 2
    src_id = 16
    neutral_id = 24
    enemy_id = 19
    planet_ids = (
        batched_game.planets.id.at[0, src_row]
        .set(src_id)
        .at[0, neutral_row]
        .set(neutral_id)
        .at[0, enemy_row]
        .set(enemy_id)
    )
    active = jnp.zeros((1, MAX_PLANETS), dtype=bool)
    active = active.at[0, src_row].set(True)
    active = active.at[0, neutral_row].set(True)
    active = active.at[0, enemy_row].set(True)
    owned_planets = batched_game.planets._replace(
        id=planet_ids,
        active=active,
        owner=batched_game.planets.owner.at[0, src_row].set(
            int(np.asarray(batched_game.player[0]))
        ),
        ships=jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32).at[0, src_row].set(20.0),
    )
    batched_game = batched_game._replace(planets=owned_planets)
    edge_mask = jnp.zeros_like(batched_batch.edge_mask).at[0, src_row, 0].set(True)
    edge_src_ids = (
        batched_batch.edge_src_ids.at[0, src_row]
        .set(src_id)
        .at[0, neutral_row]
        .set(neutral_id)
        .at[0, enemy_row]
        .set(enemy_id)
    )
    edge_tgt_ids = jnp.full_like(batched_batch.edge_tgt_ids, -1)
    edge_tgt_ids = edge_tgt_ids.at[0, src_row, 0].set(neutral_id)
    batched_batch = batched_batch._replace(
        planet_mask=active,
        edge_mask=edge_mask,
        edge_src_ids=edge_src_ids,
        edge_tgt_ids=edge_tgt_ids,
    )

    reachability = catalog_target_reachability(batched_game, batched_batch)
    sampling_mask = planet_flow_sampling_target_mask(batched_game, batched_batch)

    assert not bool(reachability[0, src_row])
    assert bool(reachability[0, neutral_row])
    assert not bool(reachability[0, enemy_row])
    assert jnp.array_equal(sampling_mask, reachability)


def test_catalog_reachability_becomes_true_after_capture_adds_edge() -> None:
    cfg = _cfg()
    _game, _batch, batched_game, batched_batch = _batched_reset(cfg)
    src_row = 0
    enemy_row = 2
    src_id = 16
    enemy_id = 19
    planet_ids = (
        batched_game.planets.id.at[0, src_row]
        .set(src_id)
        .at[0, enemy_row]
        .set(enemy_id)
    )
    active = jnp.zeros((1, MAX_PLANETS), dtype=bool).at[0, src_row].set(True)
    active = active.at[0, enemy_row].set(True)
    owned_planets = batched_game.planets._replace(
        id=planet_ids,
        active=active,
        owner=batched_game.planets.owner.at[0, src_row].set(
            int(np.asarray(batched_game.player[0]))
        ),
        ships=jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32).at[0, src_row].set(20.0),
    )
    batched_game = batched_game._replace(planets=owned_planets)
    edge_src_ids = batched_batch.edge_src_ids.at[0, src_row].set(src_id).at[
        0, enemy_row
    ].set(enemy_id)
    batched_batch = batched_batch._replace(
        planet_mask=active,
        edge_mask=jnp.zeros_like(batched_batch.edge_mask),
        edge_src_ids=edge_src_ids,
        edge_tgt_ids=jnp.full_like(batched_batch.edge_tgt_ids, -1),
    )

    before = catalog_target_reachability(batched_game, batched_batch)
    edge_mask = jnp.zeros_like(batched_batch.edge_mask).at[0, src_row, 0].set(True)
    edge_tgt_ids = jnp.full_like(batched_batch.edge_tgt_ids, -1).at[
        0, src_row, 0
    ].set(enemy_id)
    after_batch = batched_batch._replace(edge_mask=edge_mask, edge_tgt_ids=edge_tgt_ids)
    after = catalog_target_reachability(batched_game, after_batch)

    assert not bool(before[0, enemy_row])
    assert bool(after[0, enemy_row])


def test_catalog_reachability_false_without_owned_ships() -> None:
    cfg = _cfg()
    _game, _batch, batched_game, batched_batch = _batched_reset(cfg)
    active = batched_batch.planet_mask.at[0, :3].set(True)
    batched_batch = batched_batch._replace(planet_mask=active)
    batched_game = batched_game._replace(
        planets=batched_game.planets._replace(
            ships=jnp.zeros_like(batched_game.planets.ships),
        )
    )

    reachability = catalog_target_reachability(batched_game, batched_batch)

    assert not bool(jnp.any(reachability[0]))
