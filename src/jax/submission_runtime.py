"""Kaggle submission helpers for JAX v2 encode / policy / action decode."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.game.constants import MAX_PLANETS
from src.game.types import parse_observation
from src.jax.env import JaxAction, JaxFleetState, JaxGameState, JaxPlanetState
from src.jax.features_v2 import JaxTurnBatchV2


def jax_game_from_observation(obs: Any, *, max_fleet_slots: int | None = None) -> JaxGameState:
    """Convert a Kaggle observation into a single-env ``JaxGameState``."""

    game = parse_observation(obs)
    fleet_slots = int(max_fleet_slots if max_fleet_slots is not None else max(256, len(game.fleets) * 4))

    planet_ids = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32)
    owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32)
    x = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    y = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    radius = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    production = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    active = jnp.zeros((MAX_PLANETS,), dtype=bool)

    init_planet_ids = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32)
    init_owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32)
    init_x = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    init_y = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    init_radius = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    init_ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    init_production = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    init_active = jnp.zeros((MAX_PLANETS,), dtype=bool)

    for planet in game.planets:
        slot = int(planet.id)
        if slot < 0 or slot >= MAX_PLANETS:
            continue
        planet_ids = planet_ids.at[slot].set(int(planet.id))
        owner = owner.at[slot].set(int(planet.owner))
        x = x.at[slot].set(float(planet.x))
        y = y.at[slot].set(float(planet.y))
        radius = radius.at[slot].set(float(planet.radius))
        ships = ships.at[slot].set(float(planet.ships))
        production = production.at[slot].set(float(planet.production))
        active = active.at[slot].set(True)

    initial_rows = game.initial_planets or game.planets
    for planet in initial_rows:
        slot = int(planet.id)
        if slot < 0 or slot >= MAX_PLANETS:
            continue
        init_planet_ids = init_planet_ids.at[slot].set(int(planet.id))
        init_owner = init_owner.at[slot].set(int(planet.owner))
        init_x = init_x.at[slot].set(float(planet.x))
        init_y = init_y.at[slot].set(float(planet.y))
        init_radius = init_radius.at[slot].set(float(planet.radius))
        init_ships = init_ships.at[slot].set(float(planet.ships))
        init_production = init_production.at[slot].set(float(planet.production))
        init_active = init_active.at[slot].set(True)

    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=radius,
        ships=ships,
        production=production,
        active=active,
    )
    initial_planets = JaxPlanetState(
        id=init_planet_ids,
        owner=init_owner,
        x=init_x,
        y=init_y,
        radius=init_radius,
        ships=init_ships,
        production=init_production,
        active=init_active,
    )

    fleet_id = jnp.full((fleet_slots,), -1, dtype=jnp.int32)
    fleet_owner = jnp.full((fleet_slots,), -1, dtype=jnp.int32)
    fleet_x = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    fleet_y = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    fleet_angle = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    fleet_from_planet = jnp.full((fleet_slots,), -1, dtype=jnp.int32)
    fleet_ships = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    fleet_active = jnp.zeros((fleet_slots,), dtype=bool)

    next_fleet_id = 0
    for slot_idx, fleet in enumerate(game.fleets):
        if slot_idx >= fleet_slots:
            break
        fleet_id = fleet_id.at[slot_idx].set(int(fleet.id))
        fleet_owner = fleet_owner.at[slot_idx].set(int(fleet.owner))
        fleet_x = fleet_x.at[slot_idx].set(float(fleet.x))
        fleet_y = fleet_y.at[slot_idx].set(float(fleet.y))
        fleet_angle = fleet_angle.at[slot_idx].set(float(fleet.angle))
        fleet_from_planet = fleet_from_planet.at[slot_idx].set(int(fleet.from_planet_id))
        fleet_ships = fleet_ships.at[slot_idx].set(float(fleet.ships))
        fleet_active = fleet_active.at[slot_idx].set(True)
        next_fleet_id = max(next_fleet_id, int(fleet.id) + 1)

    fleets = JaxFleetState(
        id=fleet_id,
        owner=fleet_owner,
        x=fleet_x,
        y=fleet_y,
        angle=fleet_angle,
        from_planet_id=fleet_from_planet,
        ships=fleet_ships,
        active=fleet_active,
    )

    return JaxGameState(
        step=jnp.asarray(int(game.step), dtype=jnp.int32),
        player=jnp.asarray(int(game.player), dtype=jnp.int32),
        angular_velocity=jnp.asarray(float(game.angular_velocity), dtype=jnp.float32),
        next_fleet_id=jnp.asarray(next_fleet_id, dtype=jnp.int32),
        planets=planets,
        initial_planets=initial_planets,
        fleets=fleets,
    )


def batch_game(game: JaxGameState) -> JaxGameState:
    """Add a leading batch dimension to a single-env game state."""

    return JaxGameState(
        step=game.step[None],
        player=game.player[None],
        angular_velocity=game.angular_velocity[None],
        next_fleet_id=game.next_fleet_id[None],
        planets=jax.tree_util.tree_map(lambda value: value[None, ...], game.planets),
        initial_planets=jax.tree_util.tree_map(
            lambda value: value[None, ...], game.initial_planets
        ),
        fleets=jax.tree_util.tree_map(lambda value: value[None, ...], game.fleets),
    )


def batch_turn(batch: JaxTurnBatchV2) -> JaxTurnBatchV2:
    """Add a leading batch dimension to a single-env v2 turn batch."""

    return JaxTurnBatchV2(
        planet_features=batch.planet_features[None, ...],
        planet_mask=batch.planet_mask[None, ...],
        edge_features=batch.edge_features[None, ...],
        edge_mask=batch.edge_mask[None, ...],
        edge_src_ids=batch.edge_src_ids[None, ...],
        edge_tgt_ids=batch.edge_tgt_ids[None, ...],
        global_features=batch.global_features[None, ...],
        theta_ref=batch.theta_ref[None],
    )


def select_runtime_shielded_policy_actions_v2(
    key: jax.Array,
    policy: object,
    variables: dict[str, object],
    game: JaxGameState,
    batch: JaxTurnBatchV2,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    """Sample a shielded v2 policy action sequence and decode it to ``JaxAction``."""

    from src.opponents.jax_actions.builders_v2 import _sample_policy_action_v2_with_params

    return _sample_policy_action_v2_with_params(
        key,
        game,
        batch,
        variables,
        policy,
        cfg,
        deterministic=deterministic,
    )


def moves_from_jax_action(action: JaxAction, *, env_index: int = 0) -> list[list[float | int]]:
    """Convert a ``JaxAction`` buffer into Kaggle move lists."""

    source_ids = jax.device_get(action.source_id[env_index])
    angles = jax.device_get(action.angle[env_index])
    ships = jax.device_get(action.ships[env_index])
    valid = jax.device_get(action.valid[env_index])

    moves: list[list[float | int]] = []
    for source_id, angle, ship_count, is_valid in zip(
        source_ids, angles, ships, valid, strict=False
    ):
        if not bool(is_valid):
            continue
        if int(source_id) < 0 or float(ship_count) <= 0.0:
            continue
        moves.append([int(source_id), float(angle), int(ship_count)])
    return moves
