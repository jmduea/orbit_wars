"""Kaggle submission helpers for JAX encode / policy / action decode."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.game.constants import MAX_PLANETS
from src.game.types import parse_observation
from src.jax.env import JaxAction, JaxFleetState, JaxGameState, JaxPlanetState
from src.jax.features import TurnBatch


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




def apply_feature_metadata_to_model_config(
    cfg: TrainConfig,
    feature_metadata: Mapping[str, object] | None,
) -> TrainConfig:
    """Align runtime model config with checkpoint feature metadata."""

    if not feature_metadata:
        return cfg
    stored_decoder = feature_metadata.get("pointer_decoder")
    if stored_decoder is not None:
        cfg.model.pointer_decoder = str(stored_decoder)
    return cfg

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


def batch_turn(batch: TurnBatch) -> TurnBatch:
    """Add a leading batch dimension to a single-env v2 turn batch."""

    return TurnBatch(
        planet_features=batch.planet_features[None, ...],
        planet_mask=batch.planet_mask[None, ...],
        edge_features=batch.edge_features[None, ...],
        edge_mask=batch.edge_mask[None, ...],
        edge_src_ids=batch.edge_src_ids[None, ...],
        edge_tgt_ids=batch.edge_tgt_ids[None, ...],
        global_features=batch.global_features[None, ...],
        theta_ref=batch.theta_ref[None],
    )


def select_runtime_shielded_policy_actions(
    key: jax.Array,
    policy: object,
    variables: dict[str, object],
    game: JaxGameState,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    deterministic: bool,
    deterministic_eval: bool = False,
) -> JaxAction:
    """Sample a shielded v2 policy action sequence and decode it to ``JaxAction``."""

    from src.opponents.jax_actions.builders import _sample_policy_action_with_params

    return _sample_policy_action_with_params(
        key,
        game,
        batch,
        variables,
        policy,
        cfg,
        deterministic=deterministic,
        deterministic_eval=deterministic_eval,
    )



def compile_shielded_policy_act(
    policy: object,
    variables: dict[str, object],
    cfg: TrainConfig,
    *,
    deterministic: bool = True,
    deterministic_eval: bool = True,
):
    """Return a JIT-compiled shielded policy act fn for submission inference."""

    from src.opponents.jax_actions.builders import _sample_policy_action_with_params

    def _compiled_act(
        game: JaxGameState,
        batch: TurnBatch,
        key: jax.Array,
    ) -> JaxAction:
        return _sample_policy_action_with_params(
            key,
            game,
            batch,
            variables,
            policy,
            cfg,
            deterministic=deterministic,
            deterministic_eval=deterministic_eval,
        )

    return jax.jit(_compiled_act)


def _jax_game_from_parsed(game, *, fleet_slots: int) -> JaxGameState:
    """Build a ``JaxGameState`` with NumPy bulk fills (submission hot path)."""

    import numpy as np

    planet_ids = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    owner = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    x = np.zeros((MAX_PLANETS,), dtype=np.float32)
    y = np.zeros((MAX_PLANETS,), dtype=np.float32)
    radius = np.zeros((MAX_PLANETS,), dtype=np.float32)
    ships = np.zeros((MAX_PLANETS,), dtype=np.float32)
    production = np.zeros((MAX_PLANETS,), dtype=np.float32)
    active = np.zeros((MAX_PLANETS,), dtype=bool)

    init_planet_ids = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    init_owner = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    init_x = np.zeros((MAX_PLANETS,), dtype=np.float32)
    init_y = np.zeros((MAX_PLANETS,), dtype=np.float32)
    init_radius = np.zeros((MAX_PLANETS,), dtype=np.float32)
    init_ships = np.zeros((MAX_PLANETS,), dtype=np.float32)
    init_production = np.zeros((MAX_PLANETS,), dtype=np.float32)
    init_active = np.zeros((MAX_PLANETS,), dtype=bool)

    for planet in game.planets:
        slot = int(planet.id)
        if slot < 0 or slot >= MAX_PLANETS:
            continue
        planet_ids[slot] = int(planet.id)
        owner[slot] = int(planet.owner)
        x[slot] = float(planet.x)
        y[slot] = float(planet.y)
        radius[slot] = float(planet.radius)
        ships[slot] = float(planet.ships)
        production[slot] = float(planet.production)
        active[slot] = True

    initial_rows = game.initial_planets or game.planets
    for planet in initial_rows:
        slot = int(planet.id)
        if slot < 0 or slot >= MAX_PLANETS:
            continue
        init_planet_ids[slot] = int(planet.id)
        init_owner[slot] = int(planet.owner)
        init_x[slot] = float(planet.x)
        init_y[slot] = float(planet.y)
        init_radius[slot] = float(planet.radius)
        init_ships[slot] = float(planet.ships)
        init_production[slot] = float(planet.production)
        init_active[slot] = True

    fleet_id = np.full((fleet_slots,), -1, dtype=np.int32)
    fleet_owner = np.full((fleet_slots,), -1, dtype=np.int32)
    fleet_x = np.zeros((fleet_slots,), dtype=np.float32)
    fleet_y = np.zeros((fleet_slots,), dtype=np.float32)
    fleet_angle = np.zeros((fleet_slots,), dtype=np.float32)
    fleet_from_planet = np.full((fleet_slots,), -1, dtype=np.int32)
    fleet_ships = np.zeros((fleet_slots,), dtype=np.float32)
    fleet_active = np.zeros((fleet_slots,), dtype=bool)

    next_fleet_id = 0
    for slot_idx, fleet in enumerate(game.fleets):
        if slot_idx >= fleet_slots:
            break
        fleet_id[slot_idx] = int(fleet.id)
        fleet_owner[slot_idx] = int(fleet.owner)
        fleet_x[slot_idx] = float(fleet.x)
        fleet_y[slot_idx] = float(fleet.y)
        fleet_angle[slot_idx] = float(fleet.angle)
        fleet_from_planet[slot_idx] = int(fleet.from_planet_id)
        fleet_ships[slot_idx] = float(fleet.ships)
        fleet_active[slot_idx] = True
        next_fleet_id = max(next_fleet_id, int(fleet.id) + 1)

    planets = JaxPlanetState(
        id=jnp.asarray(planet_ids),
        owner=jnp.asarray(owner),
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        radius=jnp.asarray(radius),
        ships=jnp.asarray(ships),
        production=jnp.asarray(production),
        active=jnp.asarray(active),
    )
    initial_planets = JaxPlanetState(
        id=jnp.asarray(init_planet_ids),
        owner=jnp.asarray(init_owner),
        x=jnp.asarray(init_x),
        y=jnp.asarray(init_y),
        radius=jnp.asarray(init_radius),
        ships=jnp.asarray(init_ships),
        production=jnp.asarray(init_production),
        active=jnp.asarray(init_active),
    )
    fleets = JaxFleetState(
        id=jnp.asarray(fleet_id),
        owner=jnp.asarray(fleet_owner),
        x=jnp.asarray(fleet_x),
        y=jnp.asarray(fleet_y),
        angle=jnp.asarray(fleet_angle),
        from_planet_id=jnp.asarray(fleet_from_planet),
        ships=jnp.asarray(fleet_ships),
        active=jnp.asarray(fleet_active),
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


def jax_game_from_observation_fast(
    obs: Any,
    *,
    max_fleet_slots: int | None = None,
) -> JaxGameState:
    """Parse a Kaggle observation and materialize a JAX game state efficiently."""

    game = parse_observation(obs)
    fleet_slots = int(
        max_fleet_slots if max_fleet_slots is not None else max(256, len(game.fleets) * 4)
    )
    return _jax_game_from_parsed(game, fleet_slots=fleet_slots)


def compile_batched_feature_encode(task_cfg: TaskConfig):
    """Return a JIT-compiled ``encode_turn`` plus batch-dim expansion."""

    from src.jax.features import FeatureHistory, encode_turn

    def _encode_batched(
        game: JaxGameState,
        history: FeatureHistory,
    ) -> tuple[JaxGameState, TurnBatch]:
        batch = encode_turn(game, task_cfg, history)
        return batch_game(game), batch_turn(batch)

    return jax.jit(_encode_batched)


def compile_feature_history_append(task_cfg: TaskConfig):
    """Return a JIT-compiled feature-history append for one env step."""

    from src.jax.features import FeatureHistory, append_feature_history

    def _append(history: FeatureHistory, game: JaxGameState) -> FeatureHistory:
        return append_feature_history(history, game, task_cfg)

    return jax.jit(_append)

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
