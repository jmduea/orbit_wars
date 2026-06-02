from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.game.constants import MAX_PLANETS
from src.jax.env import JaxAction
from src.jax.features import TurnBatch
from src.opponents.jax_actions.builders import owned_planet_ships


class PlanetFlowCompilerDiagnostics(NamedTuple):
    """Lean diagnostics for the P0 target-demand compiler."""

    demanded_mass: jax.Array
    unreachable_demand_mass: jax.Array
    held_demand_mass: jax.Array
    requested_ship_mass: jax.Array
    emitted_ship_mass: jax.Array
    capacity_dropped_launches: jax.Array
    emitted_launch_count: jax.Array
    small_launch_count: jax.Array
    duplicate_source_target_count: jax.Array


class PlanetFlowCompileResult(NamedTuple):
    action: JaxAction
    diagnostics: PlanetFlowCompilerDiagnostics


def seeded_random_target_pressure(
    key: jax.Array,
    batch: TurnBatch,
    pressure_bucket_values: jax.Array,
) -> jax.Array:
    """Sample a seeded random Planet Flow demand field for compiler controls."""

    bucket_values = jnp.asarray(pressure_bucket_values, dtype=jnp.float32)
    bucket_count = bucket_values.shape[0]
    sampled_bucket = jax.random.randint(
        key,
        batch.planet_mask.shape,
        minval=0,
        maxval=bucket_count,
        dtype=jnp.int32,
    )
    pressure = jnp.take(bucket_values, sampled_bucket, axis=0)
    return jnp.where(batch.planet_mask, pressure, 0.0)


def compile_seeded_random_planet_flow_control(
    key: jax.Array,
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
) -> PlanetFlowCompileResult:
    """Run the P0 compiler with seeded random demand for attribution baselines."""

    target_pressure = seeded_random_target_pressure(
        key,
        batch,
        jnp.asarray(cfg.model.planet_flow.pressure_bucket_values, dtype=jnp.float32),
    )
    return compile_planet_flow_action(game, batch, target_pressure, cfg)


def _edge_target_pressure(
    batch_row: TurnBatch,
    target_pressure: jax.Array,
) -> jax.Array:
    """Gather per-edge target demand without an all-pairs source-target tensor."""

    target_ids = batch_row.edge_tgt_ids
    safe_target_ids = jnp.clip(target_ids, 0, MAX_PLANETS - 1)
    return jnp.take(target_pressure, safe_target_ids, axis=0)


def _target_reachability(
    batch_row: TurnBatch,
    feasible_edge: jax.Array,
) -> jax.Array:
    planet_ids = batch_row.edge_src_ids
    target_ids = batch_row.edge_tgt_ids
    safe_target_ids = jnp.clip(target_ids, 0, MAX_PLANETS - 1)
    reachable_by_id = jnp.zeros((MAX_PLANETS,), dtype=jnp.int32)
    reachable_by_id = reachable_by_id.at[safe_target_ids].max(
        feasible_edge.astype(jnp.int32)
    )
    safe_planet_ids = jnp.clip(planet_ids, 0, MAX_PLANETS - 1)
    return jnp.take(reachable_by_id > 0, safe_planet_ids, axis=0)


def _compile_planet_flow_row(
    game_row,
    batch_row: TurnBatch,
    target_pressure: jax.Array,
    cfg: TrainConfig,
) -> PlanetFlowCompileResult:
    fleet_slots = int(cfg.task.max_fleets)
    slot_count = min(MAX_PLANETS, fleet_slots)
    source_ships = owned_planet_ships(game_row)
    source_active = (source_ships > 0.0) & batch_row.planet_mask
    edge_pressure = _edge_target_pressure(batch_row, target_pressure)
    feasible_edge = batch_row.edge_mask & source_active[:, None] & (edge_pressure > 0.0)
    feasible_pressure = jnp.where(feasible_edge, edge_pressure, -1.0)
    best_slot = jnp.argmax(feasible_pressure, axis=-1)
    best_pressure = jnp.take_along_axis(
        feasible_pressure, best_slot[:, None], axis=-1
    ).squeeze(-1)
    selected_target_id = jnp.take_along_axis(
        batch_row.edge_tgt_ids, best_slot[:, None], axis=-1
    ).squeeze(-1)
    valid_by_source = source_active & (best_pressure > 0.0)
    requested_ships = source_ships * jnp.clip(best_pressure, 0.0, 1.0)
    ships_by_source = jnp.where(valid_by_source, requested_ships, 0.0)
    source_ids_by_source = jnp.where(valid_by_source, batch_row.edge_src_ids, -1)
    safe_planet_ids = jnp.clip(game_row.planets.id, 0, MAX_PLANETS - 1)
    x_by_id = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32).at[safe_planet_ids].set(
        game_row.planets.x
    )
    y_by_id = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32).at[safe_planet_ids].set(
        game_row.planets.y
    )
    safe_selected_target_id = jnp.clip(selected_target_id, 0, MAX_PLANETS - 1)
    target_x = jnp.take(x_by_id, safe_selected_target_id, axis=0)
    target_y = jnp.take(y_by_id, safe_selected_target_id, axis=0)
    angles_by_source = jnp.arctan2(
        target_y - game_row.planets.y,
        target_x - game_row.planets.x,
    )
    angles_by_source = jnp.where(valid_by_source, angles_by_source, 0.0)

    out_source_id = jnp.full((fleet_slots,), -1, dtype=jnp.int32)
    out_angle = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    out_ships = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    out_valid = jnp.zeros((fleet_slots,), dtype=bool)
    out_source_id = out_source_id.at[:slot_count].set(source_ids_by_source[:slot_count])
    out_angle = out_angle.at[:slot_count].set(angles_by_source[:slot_count])
    out_ships = out_ships.at[:slot_count].set(ships_by_source[:slot_count])
    out_valid = out_valid.at[:slot_count].set(valid_by_source[:slot_count])

    target_reachable = _target_reachability(batch_row, feasible_edge)
    active_demand = jnp.where(batch_row.planet_mask, target_pressure, 0.0)
    demanded_mass = active_demand.sum()
    unreachable_demand_mass = jnp.where(
        batch_row.planet_mask & (active_demand > 0.0) & (~target_reachable),
        active_demand,
        0.0,
    ).sum()
    requested_ship_mass = ships_by_source.sum()
    emitted_ship_mass = out_ships.sum()
    emitted_pressure = jnp.where(valid_by_source, best_pressure, 0.0).sum()
    held_demand_mass = jnp.maximum(demanded_mass - emitted_pressure, 0.0)
    capacity_dropped_launches = valid_by_source[slot_count:].sum().astype(jnp.float32)
    emitted_launch_count = out_valid.astype(jnp.float32).sum()
    small_launch_count = (
        out_valid & (out_ships > 0.0) & (out_ships <= 1.0)
    ).sum().astype(jnp.float32)
    return PlanetFlowCompileResult(
        action=JaxAction(
            source_id=out_source_id,
            angle=out_angle,
            ships=out_ships,
            valid=out_valid,
        ),
        diagnostics=PlanetFlowCompilerDiagnostics(
            demanded_mass=demanded_mass,
            unreachable_demand_mass=unreachable_demand_mass,
            held_demand_mass=held_demand_mass,
            requested_ship_mass=requested_ship_mass,
            emitted_ship_mass=emitted_ship_mass,
            capacity_dropped_launches=capacity_dropped_launches,
            emitted_launch_count=emitted_launch_count,
            small_launch_count=small_launch_count,
            duplicate_source_target_count=jnp.array(0.0, dtype=jnp.float32),
        ),
    )


def compile_planet_flow_action(
    game,
    batch: TurnBatch,
    target_pressure: jax.Array,
    cfg: TrainConfig,
) -> PlanetFlowCompileResult:
    """Compile all-active target demand into ordinary env ``JaxAction`` launches."""

    return jax.vmap(_compile_planet_flow_row, in_axes=(0, 0, 0, None))(
        game, batch, target_pressure, cfg
    )
