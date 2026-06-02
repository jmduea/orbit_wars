from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from src.jax.planet_flow import (
    compile_planet_flow_action,
    compile_seeded_random_planet_flow_control,
    seeded_random_target_pressure,
)

import jax
from src.config import TrainConfig
from src.game.constants import MAX_PLANETS
from src.jax.env import reset


def _cfg(*, max_fleets: int = 32) -> TrainConfig:
    cfg = TrainConfig()
    cfg.task.max_fleets = max_fleets
    cfg.task.candidate_count = 4
    return cfg


def _batched_reset(cfg: TrainConfig):
    state, batch = reset(jax.random.PRNGKey(0), cfg.task)
    batched_game = jax.tree.map(lambda x: x[None, ...], state.game)
    batched_batch = jax.tree.map(lambda x: x[None, ...], batch)
    return state.game, batch, batched_game, batched_batch


def _first_owned_candidate(game, batch) -> tuple[int, int, int]:
    owner = int(np.asarray(game.player))
    owned = np.asarray((game.planets.active & (game.planets.owner == owner)))
    edge_mask = np.asarray(batch.edge_mask)
    for src_row in np.where(owned)[0]:
        slots = np.where(edge_mask[src_row])[0]
        if len(slots) == 0:
            continue
        slot = int(slots[0])
        target_id = int(np.asarray(batch.edge_tgt_ids[src_row, slot]))
        target_row = int(np.where(np.asarray(batch.edge_src_ids) == target_id)[0][0])
        return int(src_row), slot, target_row
    raise AssertionError("expected at least one owned source with a candidate edge")


def test_compile_planet_flow_action_emits_candidate_bounded_launch() -> None:
    cfg = _cfg()
    game, batch, batched_game, batched_batch = _batched_reset(cfg)
    src_row, _slot, target_row = _first_owned_candidate(game, batch)
    target_pressure = jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32)
    target_pressure = target_pressure.at[0, target_row].set(0.5)

    result = compile_planet_flow_action(
        batched_game,
        batched_batch,
        target_pressure,
        cfg,
    )

    assert bool(result.action.valid[0, src_row])
    assert int(result.action.source_id[0, src_row]) == int(
        np.asarray(batch.edge_src_ids[src_row])
    )
    assert float(result.action.ships[0, src_row]) > 0.0
    assert float(result.diagnostics.emitted_ship_mass[0]) > 0.0
    assert float(result.diagnostics.unreachable_demand_mass[0]) == 0.0


def test_compile_planet_flow_action_holds_when_no_target_demand() -> None:
    cfg = _cfg()
    _game, _batch, batched_game, batched_batch = _batched_reset(cfg)
    target_pressure = jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32)

    result = compile_planet_flow_action(
        batched_game,
        batched_batch,
        target_pressure,
        cfg,
    )

    assert not bool(jnp.any(result.action.valid[0]))
    assert float(result.diagnostics.emitted_ship_mass[0]) == 0.0
    assert float(result.diagnostics.held_demand_mass[0]) == 0.0


def test_compile_planet_flow_action_reports_unreachable_demand() -> None:
    cfg = _cfg()
    game, batch, batched_game, batched_batch = _batched_reset(cfg)
    _src_row, _slot, target_row = _first_owned_candidate(game, batch)
    target_pressure = jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32)
    target_pressure = target_pressure.at[0, target_row].set(0.75)
    blocked_batch = batched_batch._replace(edge_mask=jnp.zeros_like(batched_batch.edge_mask))

    result = compile_planet_flow_action(
        batched_game,
        blocked_batch,
        target_pressure,
        cfg,
    )

    assert not bool(jnp.any(result.action.valid[0]))
    assert float(result.diagnostics.unreachable_demand_mass[0]) == 0.75


def test_compile_planet_flow_action_reports_capacity_drops() -> None:
    full_cfg = _cfg(max_fleets=32)
    _game, _batch, batched_game, batched_batch = _batched_reset(full_cfg)
    planet_ids = jnp.arange(MAX_PLANETS, dtype=jnp.int32)[None, :]
    active = jnp.zeros_like(batched_game.planets.active).at[:, :3].set(True)
    owned_planets = batched_game.planets._replace(
        id=planet_ids,
        active=active,
        owner=jnp.where(active, batched_game.player[:, None], batched_game.planets.owner),
        ships=jnp.where(active, 10.0, batched_game.planets.ships),
    )
    batched_game = batched_game._replace(planets=owned_planets)
    edge_mask = jnp.zeros_like(batched_batch.edge_mask).at[:, :2, 0].set(True)
    edge_tgt_ids = jnp.zeros_like(batched_batch.edge_tgt_ids).at[:, :2, 0].set(2)
    batched_batch = batched_batch._replace(
        planet_mask=active,
        edge_mask=edge_mask,
        edge_src_ids=planet_ids,
        edge_tgt_ids=edge_tgt_ids,
    )
    target_pressure = jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32).at[:, 2].set(1.0)
    full_result = compile_planet_flow_action(
        batched_game,
        batched_batch,
        target_pressure,
        full_cfg,
    )
    full_launches = float(full_result.diagnostics.emitted_launch_count[0])
    assert full_launches > 1.0

    cfg = _cfg(max_fleets=1)

    result = compile_planet_flow_action(
        batched_game,
        batched_batch,
        target_pressure,
        cfg,
    )

    assert result.action.source_id.shape == (1, 1)
    assert float(result.diagnostics.emitted_launch_count[0]) == 1.0
    assert float(result.diagnostics.capacity_dropped_launches[0]) == full_launches - 1.0


def test_seeded_random_target_pressure_is_stable_and_masks_inactive() -> None:
    cfg = _cfg()
    _game, _batch, _batched_game, batched_batch = _batched_reset(cfg)
    target_mask = jnp.ones_like(batched_batch.planet_mask)
    target_mask = target_mask.at[:, 3:].set(False)
    masked_batch = batched_batch._replace(planet_mask=target_mask)
    bucket_values = jnp.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float32)

    first = seeded_random_target_pressure(
        jax.random.PRNGKey(123),
        masked_batch,
        bucket_values,
    )
    second = seeded_random_target_pressure(
        jax.random.PRNGKey(123),
        masked_batch,
        bucket_values,
    )

    assert jnp.array_equal(first, second)
    assert jnp.all(first[:, 3:] == 0.0)


def test_seeded_random_compiler_control_uses_planet_flow_compiler() -> None:
    cfg = _cfg()
    _game, _batch, batched_game, batched_batch = _batched_reset(cfg)

    result = compile_seeded_random_planet_flow_control(
        jax.random.PRNGKey(456),
        batched_game,
        batched_batch,
        cfg,
    )

    assert result.action.source_id.shape == (1, cfg.task.max_fleets)
    assert result.diagnostics.demanded_mass.shape == (1,)
