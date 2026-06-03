"""JAX-native Orbit Wars environment.

This module provides a pure, fixed-shape implementation of the core Orbit Wars
mechanics used by the training code.  It intentionally stores planets and fleets
as padded arrays so ``reset``/``step`` can be composed with ``jax.vmap`` and
``jax.jit``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import jax
from src.config import RewardConfig, TaskConfig
from src.game.constants import (
    BOARD_SIZE,
    COMET_OFF_BOARD,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMET_SPEED,
    COMETS_PER_GROUP,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_FLEET_SPEED,
    MAX_PLANETS,
    MAX_STEPS,
    PLANET_LAUNCH_RADIUS_OFFSET,
    SUN_RADIUS,
    TOTAL_COMETS,
)
from src.jax.rewards import apply_early_terminal_reward_shaping_jax

from .features import (
    FeatureHistory,
    TurnBatch,
    empty_feature_history,
    encode_learner_turn,
)

BOARD_CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_SHIP_SPEED = 6.0


class JaxPlanetState(NamedTuple):
    """Fixed-shape planet table used by the JAX environment.

    Every field has shape ``(max_planets,)``. Inactive padding rows are marked
    by ``active=False`` and should be ignored by mechanics and feature encoders.
    """

    id: jax.Array
    owner: jax.Array
    x: jax.Array
    y: jax.Array
    radius: jax.Array
    ships: jax.Array
    production: jax.Array
    active: jax.Array


class JaxFleetState(NamedTuple):
    """Fixed-shape fleet table used by the JAX environment.

    Every field has shape ``(max_fleets,)`` after compaction. Inactive padding
    rows are marked by ``active=False`` and are safe to carry through JIT code.
    """

    id: jax.Array
    owner: jax.Array
    x: jax.Array
    y: jax.Array
    angle: jax.Array
    from_planet_id: jax.Array
    ships: jax.Array
    active: jax.Array


class JaxCometState(NamedTuple):
    """Fixed-shape comet groups spawned at ``COMET_SPAWN_STEPS``."""

    group_count: jax.Array
    path_index: jax.Array
    planet_ids: jax.Array
    paths_x: jax.Array
    paths_y: jax.Array
    path_lengths: jax.Array
    group_active: jax.Array


class JaxGameState(NamedTuple):
    """Complete Orbit Wars game state represented as JAX arrays.

    The state is immutable from the caller's perspective: transition functions
    return updated copies so the structure can be used with ``jax.jit``,
    ``jax.vmap``, and ``jax.lax.scan``.
    """

    step: jax.Array
    player: jax.Array
    angular_velocity: jax.Array
    next_fleet_id: jax.Array
    episode_seed: jax.Array
    planets: JaxPlanetState
    initial_planets: JaxPlanetState
    fleets: JaxFleetState
    comets: JaxCometState


def empty_comet_state() -> JaxCometState:
    """Return an empty comet schedule for reset and observation replay."""

    return JaxCometState(
        group_count=jnp.array(0, dtype=jnp.int32),
        path_index=jnp.full((MAX_COMET_GROUPS,), -1, dtype=jnp.int32),
        planet_ids=jnp.full(
            (MAX_COMET_GROUPS, COMETS_PER_GROUP), -1, dtype=jnp.int32
        ),
        paths_x=jnp.zeros(
            (MAX_COMET_GROUPS, COMETS_PER_GROUP, MAX_COMET_PATH_LEN),
            dtype=jnp.float32,
        ),
        paths_y=jnp.zeros(
            (MAX_COMET_GROUPS, COMETS_PER_GROUP, MAX_COMET_PATH_LEN),
            dtype=jnp.float32,
        ),
        path_lengths=jnp.zeros(
            (MAX_COMET_GROUPS, COMETS_PER_GROUP), dtype=jnp.int32
        ),
        group_active=jnp.zeros((MAX_COMET_GROUPS,), dtype=bool),
    )


class JaxEnvState(NamedTuple):
    """Environment wrapper state containing game data and learner side.

    ``episode_count`` tracks how many completed episodes each vectorized
    environment has reset through so training can rotate learner sides
    deterministically across both environment slots and episodes.
    """

    game: JaxGameState
    learner_player: jax.Array
    episode_count: jax.Array
    feature_history: FeatureHistory | None = None
    decoder_hidden: jax.Array | None = None


class JaxAction(NamedTuple):
    """Fixed-size move buffer.

    Each slot mirrors Kaggle's ``[from_planet_id, angle, num_ships]`` action.
    ``valid`` marks populated slots; inactive slots are ignored.
    """

    source_id: jax.Array
    angle: jax.Array
    ships: jax.Array
    valid: jax.Array


class JaxStepResult(NamedTuple):
    """Result payload returned by :func:`step`.

    Rewards are split into terminal and shaping components so callers can log
    diagnostics without re-computing game-state deltas.
    """

    batch: TurnBatch
    reward: jax.Array
    done: jax.Array
    terminal_reward: jax.Array
    shaping_reward: jax.Array
    reward_capture_planet: jax.Array
    reward_ship_delta: jax.Array
    reward_production_delta: jax.Array
    terminal_rank: jax.Array
    terminal_placement: jax.Array
    terminal_is_first: jax.Array
    terminal_score_share: jax.Array
    terminal_ship_differential: jax.Array
    terminal_survival_time: jax.Array


def max_fleets(cfg: TaskConfig) -> int:
    """Return the configured fixed fleet-buffer length for JAX state arrays."""

    return int(getattr(cfg, "max_fleets", max(256, cfg.max_fleets * 4)))


def empty_action(cfg: TaskConfig) -> JaxAction:
    """Create an empty fixed-size action buffer for one environment."""

    fleet_count = max_fleets(cfg)
    return JaxAction(
        source_id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        angle=jnp.zeros((fleet_count,), dtype=jnp.float32),
        ships=jnp.zeros((fleet_count,), dtype=jnp.float32),
        valid=jnp.zeros((fleet_count,), dtype=bool),
    )


def _reference_planet_tables(
    seed: np.ndarray, player_count: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Build padded planet tables with the Kaggle reference generator."""

    import random

    from src.game.planet_generation import (
        assign_home_planets,
        generate_planets,
        planets_to_padded_rows,
    )

    rng = random.Random(int(np.asarray(seed).item()))
    planets = generate_planets(rng)
    num_groups = max(1, len(planets) // 4)
    home_group = rng.randint(0, num_groups - 1)
    assign_home_planets(
        planets,
        player_count=int(np.asarray(player_count).item()),
        home_group=home_group,
    )
    rows = planets_to_padded_rows(planets)
    return tuple(np.asarray(row) for row in rows)


def _planet_table_specs() -> tuple[jax.ShapeDtypeStruct, ...]:
    return (
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.int32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.int32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.float32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.float32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.float32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.float32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), jnp.float32),
        jax.ShapeDtypeStruct((MAX_PLANETS,), np.bool_),
    )


_COMET_SPAWN_STEPS = jnp.array(COMET_SPAWN_STEPS, dtype=jnp.int32)


def _active_comet_planet_ids(comets: JaxCometState) -> jax.Array:
    valid = comets.group_active[:, None] & (comets.planet_ids >= 0)
    return jnp.where(valid, comets.planet_ids, -1).reshape(
        (MAX_COMET_GROUPS * COMETS_PER_GROUP,)
    )


def _is_comet_planet(comets: JaxCometState, planet_ids: jax.Array) -> jax.Array:
    return jnp.isin(planet_ids, _active_comet_planet_ids(comets))


def _pack_initial_planets_arrays(
    planet_id: np.ndarray,
    owner: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    packed = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    for i in np.flatnonzero(np.asarray(active)):
        packed[i] = [
            float(planet_id[i]),
            float(owner[i]),
            float(x[i]),
            float(y[i]),
            float(radius[i]),
            float(ships[i]),
            float(production[i]),
        ]
    return packed


def _reference_comet_paths(
    episode_seed: np.ndarray,
    spawn_step: np.ndarray,
    angular_velocity: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_speed: np.ndarray,
    initial_packed: np.ndarray,
) -> tuple[np.ndarray, ...]:
    import random

    from src.game.comet_generation import generate_comet_paths

    rows = [
        [
            int(row[0]),
            int(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
        ]
        for row in np.asarray(initial_packed).reshape(MAX_PLANETS, 7)
        if int(row[0]) >= 0
    ]
    excluded = [int(x) for x in np.asarray(comet_planet_ids).reshape(-1) if int(x) >= 0]
    seed = int(np.asarray(episode_seed).item())
    step = int(np.asarray(spawn_step).item())
    rng = random.Random(f"orbit_wars-comet-{seed}-{step}")
    paths = generate_comet_paths(
        rows,
        float(np.asarray(angular_velocity).item()),
        step,
        excluded,
        float(np.asarray(comet_speed).item()),
        rng=rng,
    )
    paths_x = np.zeros((COMETS_PER_GROUP, MAX_COMET_PATH_LEN), dtype=np.float32)
    paths_y = np.zeros((COMETS_PER_GROUP, MAX_COMET_PATH_LEN), dtype=np.float32)
    path_lengths = np.zeros((COMETS_PER_GROUP,), dtype=np.int32)
    comet_ships = np.array(1.0, dtype=np.float32)
    ok = np.array(False, dtype=np.bool_)
    if paths:
        ok = np.array(True, dtype=np.bool_)
        comet_ships = float(
            min(
                rng.randint(1, 99),
                rng.randint(1, 99),
                rng.randint(1, 99),
                rng.randint(1, 99),
            )
        )
        for i, path in enumerate(paths[:COMETS_PER_GROUP]):
            length = min(len(path), MAX_COMET_PATH_LEN)
            path_lengths[i] = length
            for j in range(length):
                paths_x[i, j] = float(path[j][0])
                paths_y[i, j] = float(path[j][1])
    return paths_x, paths_y, path_lengths, np.array(comet_ships, dtype=np.float32), ok


def _comet_path_specs() -> tuple[jax.ShapeDtypeStruct, ...]:
    return (
        jax.ShapeDtypeStruct((COMETS_PER_GROUP, MAX_COMET_PATH_LEN), jnp.float32),
        jax.ShapeDtypeStruct((COMETS_PER_GROUP, MAX_COMET_PATH_LEN), jnp.float32),
        jax.ShapeDtypeStruct((COMETS_PER_GROUP,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((), np.bool_),
    )


def _deactivate_planets_by_id(
    planets: JaxPlanetState,
    remove_ids: jax.Array,
) -> JaxPlanetState:
    remove = jnp.isin(planets.id, remove_ids) & (remove_ids >= 0)
    return planets._replace(
        active=planets.active & (~remove),
        owner=jnp.where(remove, -1, planets.owner),
    )


def _expire_comets_pre_launch(
    planets: JaxPlanetState,
    initial_planets: JaxPlanetState,
    comets: JaxCometState,
) -> tuple[JaxPlanetState, JaxPlanetState, JaxCometState]:
    def group_body(g, carry):
        planets, initial, comets = carry
        active = comets.group_active[g]
        idx = comets.path_index[g]

        def comet_body(i, inner):
            planets, initial, comets = inner
            pid = comets.planet_ids[g, i]
            path_len = comets.path_lengths[g, i]
            expire = active & (pid >= 0) & (idx >= path_len)
            planets = _deactivate_planets_by_id(
                planets, jnp.where(expire, pid, jnp.array(-1, dtype=jnp.int32))
            )
            initial = _deactivate_planets_by_id(
                initial, jnp.where(expire, pid, jnp.array(-1, dtype=jnp.int32))
            )
            comets = comets._replace(
                planet_ids=comets.planet_ids.at[g, i].set(jnp.where(expire, -1, pid))
            )
            return planets, initial, comets

        planets, initial, comets = jax.lax.fori_loop(
            0, COMETS_PER_GROUP, comet_body, (planets, initial, comets)
        )
        has_ids = (comets.planet_ids[g] >= 0).any()
        comets = comets._replace(
            group_active=comets.group_active.at[g].set(active & has_ids)
        )
        return planets, initial, comets

    return jax.lax.fori_loop(
        0, MAX_COMET_GROUPS, group_body, (planets, initial_planets, comets)
    )


def _spawn_comet_group(
    game: JaxGameState,
    planets: JaxPlanetState,
    initial_planets: JaxPlanetState,
    comets: JaxCometState,
    spawn_step: jax.Array,
    comet_speed: float,
) -> tuple[JaxPlanetState, JaxPlanetState, JaxCometState]:
    group_slot = comets.group_count

    def try_spawn(_):
        initial_packed = jax.pure_callback(
            _pack_initial_planets_arrays,
            jax.ShapeDtypeStruct((MAX_PLANETS, 7), jnp.float32),
            initial_planets.id,
            initial_planets.owner,
            initial_planets.x,
            initial_planets.y,
            initial_planets.radius,
            initial_planets.ships,
            initial_planets.production,
            initial_planets.active,
            vmap_method="sequential",
        )
        paths_x, paths_y, path_lengths, comet_ships, ok = jax.pure_callback(
            _reference_comet_paths,
            _comet_path_specs(),
            game.episode_seed,
            spawn_step,
            game.angular_velocity,
            _active_comet_planet_ids(comets),
            jnp.array(comet_speed, dtype=jnp.float32),
            initial_packed,
            vmap_method="sequential",
        )

        def place(carry):
            planets0, initial0, comets0 = carry
            g = group_slot
            base_slot = MAX_PLANETS - TOTAL_COMETS + g * COMETS_PER_GROUP
            next_id = jnp.max(jnp.where(planets0.active, planets0.id, 0)) + 1

            def place_comet(i, inner):
                p, initial, comets_local = inner
                slot = base_slot + i
                pid = next_id + i
                p = p._replace(
                    id=p.id.at[slot].set(pid),
                    owner=p.owner.at[slot].set(-1),
                    x=p.x.at[slot].set(COMET_OFF_BOARD),
                    y=p.y.at[slot].set(COMET_OFF_BOARD),
                    radius=p.radius.at[slot].set(COMET_RADIUS),
                    ships=p.ships.at[slot].set(comet_ships),
                    production=p.production.at[slot].set(COMET_PRODUCTION),
                    active=p.active.at[slot].set(True),
                )
                initial = initial._replace(
                    id=initial.id.at[slot].set(pid),
                    owner=initial.owner.at[slot].set(-1),
                    x=initial.x.at[slot].set(COMET_OFF_BOARD),
                    y=initial.y.at[slot].set(COMET_OFF_BOARD),
                    radius=initial.radius.at[slot].set(COMET_RADIUS),
                    ships=initial.ships.at[slot].set(comet_ships),
                    production=initial.production.at[slot].set(COMET_PRODUCTION),
                    active=initial.active.at[slot].set(True),
                )
                comets_local = comets_local._replace(
                    planet_ids=comets_local.planet_ids.at[g, i].set(pid),
                    paths_x=comets_local.paths_x.at[g, i].set(paths_x[i]),
                    paths_y=comets_local.paths_y.at[g, i].set(paths_y[i]),
                    path_lengths=comets_local.path_lengths.at[g, i].set(
                        path_lengths[i]
                    ),
                )
                return p, initial, comets_local

            planets_out, initial_out, comets_out = jax.lax.fori_loop(
                0, COMETS_PER_GROUP, place_comet, (planets0, initial0, comets0)
            )
            comets_out = comets_out._replace(
                group_count=comets_out.group_count + 1,
                path_index=comets_out.path_index.at[g].set(-1),
                group_active=comets_out.group_active.at[g].set(True),
            )
            return planets_out, initial_out, comets_out

        return jax.lax.cond(
            ok,
            place,
            lambda carry: carry,
            (planets, initial_planets, comets),
        )

    room = group_slot < MAX_COMET_GROUPS
    return jax.lax.cond(
        room, try_spawn, lambda _: (planets, initial_planets, comets), None
    )


def _pre_launch_comets(
    game: JaxGameState,
    comet_speed: float,
) -> tuple[JaxPlanetState, JaxPlanetState, JaxCometState]:
    planets = game.planets
    initial = game.initial_planets
    comets = game.comets
    planets, initial, comets = _expire_comets_pre_launch(planets, initial, comets)
    spawn_step = game.step + jnp.array(1, dtype=jnp.int32)
    should_spawn = jnp.any(spawn_step == _COMET_SPAWN_STEPS)

    def spawn(_):
        return _spawn_comet_group(game, planets, initial, comets, spawn_step, comet_speed)

    return jax.lax.cond(
        should_spawn,
        spawn,
        lambda _: (planets, initial, comets),
        None,
    )


def reset(
    key: jax.Array, cfg: TaskConfig
) -> tuple[JaxEnvState, TurnBatch]:
    """Create a deterministic initial board from a JAX PRNG key."""

    fleet_count = max_fleets(cfg)
    key_seed, key_vel = jax.random.split(key)
    seed = jax.random.randint(key_seed, (), 0, 2**31 - 1, dtype=jnp.int32)
    player_count = jnp.array(int(getattr(cfg, "player_count", 2)), dtype=jnp.int32)
    tables = jax.pure_callback(
        _reference_planet_tables,
        _planet_table_specs(),
        seed,
        player_count,
        vmap_method="sequential",
    )
    idx, owner, x, y, radius, ships, production, active = tables
    planets = JaxPlanetState(
        jnp.asarray(idx, dtype=jnp.int32),
        jnp.asarray(owner, dtype=jnp.int32),
        jnp.asarray(x, dtype=jnp.float32),
        jnp.asarray(y, dtype=jnp.float32),
        jnp.asarray(radius, dtype=jnp.float32),
        jnp.asarray(ships, dtype=jnp.float32),
        jnp.asarray(production, dtype=jnp.float32),
        jnp.asarray(active, dtype=bool),
    )
    empty_fleets = JaxFleetState(
        id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        owner=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        x=jnp.zeros((fleet_count,), dtype=jnp.float32),
        y=jnp.zeros((fleet_count,), dtype=jnp.float32),
        angle=jnp.zeros((fleet_count,), dtype=jnp.float32),
        from_planet_id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        ships=jnp.zeros((fleet_count,), dtype=jnp.float32),
        active=jnp.zeros((fleet_count,), dtype=bool),
    )
    game = JaxGameState(
        step=jnp.array(0, dtype=jnp.int32),
        player=jnp.array(0, dtype=jnp.int32),
        angular_velocity=jax.random.uniform(key_vel, (), minval=0.025, maxval=0.05),
        next_fleet_id=jnp.array(0, dtype=jnp.int32),
        episode_seed=seed,
        planets=planets,
        initial_planets=planets,
        fleets=empty_fleets,
        comets=empty_comet_state(),
    )
    history = empty_feature_history(cfg)
    batch, feature_history = encode_learner_turn(game, cfg, history)
    env_state = JaxEnvState(
        game=game,
        learner_player=jnp.array(0, dtype=jnp.int32),
        episode_count=jnp.array(0, dtype=jnp.int32),
        feature_history=feature_history,
    )
    return env_state, batch


def learner_player_for_episode(
    env_index: jax.Array,
    episode_count: jax.Array,
    cfg: TaskConfig,
    alternate_player_sides: bool = True,
) -> jax.Array:
    """Return the learner player id for an environment episode.

    When side alternation is enabled the assignment follows the Python training
    environment convention: ``(env_index + episode_count) % player_count``.
    Otherwise the learner always controls player 0.
    """

    player_count = max(1, int(getattr(cfg, "player_count", 2)))
    if alternate_player_sides:
        return (
            env_index.astype(jnp.int32) + episode_count.astype(jnp.int32)
        ) % jnp.array(player_count, dtype=jnp.int32)
    return jnp.zeros_like(env_index, dtype=jnp.int32)


def assign_learner_players(
    env_state: JaxEnvState,
    env_index: jax.Array,
    episode_count: jax.Array,
    cfg: TaskConfig,
    alternate_player_sides: bool = True,
) -> tuple[JaxEnvState, TurnBatch]:
    """Assign learner player ids and rebuild observations from those perspectives.

    ``reset`` itself creates a neutral player-0 learner state. Batched training
    calls this helper immediately after resets so each vectorized environment
    receives a deterministic learner side based on its slot index and per-slot
    completed episode count.
    """

    learner_player = learner_player_for_episode(
        env_index, episode_count, cfg, alternate_player_sides
    )
    games = jax.vmap(lambda game, player: game._replace(player=player))(
        env_state.game, learner_player.astype(jnp.int32)
    )
    histories = jax.vmap(lambda game: empty_feature_history(cfg))(games)
    turn_batch, feature_histories = jax.vmap(
        lambda game, history: encode_learner_turn(game, cfg, history)
    )(games, histories)
    env_state = env_state._replace(
        game=games,
        learner_player=learner_player.astype(jnp.int32),
        episode_count=episode_count.astype(jnp.int32),
        feature_history=feature_histories,
        decoder_hidden=None,
    )
    return env_state, turn_batch


def _finish_step(
    previous_game: JaxGameState,
    state: JaxEnvState,
    planets: JaxPlanetState,
    fleets: JaxFleetState,
    next_fleet_id: jax.Array,
    cfg: TaskConfig,
    reward_cfg: RewardConfig,
) -> tuple[JaxEnvState, JaxStepResult]:
    planets = planets._replace(
        ships=jnp.where(
            (planets.owner != -1) & planets.active,
            planets.ships + planets.production,
            planets.ships,
        )
    )
    planets, fleets, comets = _move_and_resolve(previous_game, planets, fleets, cfg)

    next_game = previous_game._replace(
        step=previous_game.step + jnp.array(1, dtype=jnp.int32),
        next_fleet_id=next_fleet_id,
        planets=planets,
        fleets=fleets,
        comets=comets,
    )
    (
        done,
        terminal_reward,
        learner_rank,
        learner_placement,
        learner_is_first,
        score_share,
        ship_differential,
        survival_time,
    ) = _terminal(next_game, state.learner_player, cfg, reward_cfg)
    shaping = _shaping(previous_game, next_game, state.learner_player, reward_cfg)
    reward = (
        jnp.where(done, terminal_reward * reward_cfg.reward_terminal_scale, 0.0)
        + shaping[0]
        + shaping[1]
        + shaping[2]
    )
    learner_game = next_game._replace(player=state.learner_player)
    batch, feature_history = encode_learner_turn(
        learner_game, cfg, state.feature_history
    )
    next_state = state._replace(
        game=next_game,
        feature_history=feature_history,
    )
    result = JaxStepResult(
        batch=batch,
        reward=reward,
        done=done,
        terminal_reward=jnp.where(
            done, terminal_reward * reward_cfg.reward_terminal_scale, 0.0
        ),
        shaping_reward=shaping[0] + shaping[1] + shaping[2],
        reward_capture_planet=shaping[0],
        reward_ship_delta=shaping[1],
        reward_production_delta=shaping[2],
        terminal_rank=jnp.where(done, learner_rank, 0.0),
        terminal_placement=jnp.where(done, learner_placement, 0.0),
        terminal_is_first=jnp.where(done, learner_is_first, 0.0),
        terminal_score_share=jnp.where(done, score_share, 0.0),
        terminal_ship_differential=jnp.where(done, ship_differential, 0.0),
        terminal_survival_time=jnp.where(done, survival_time, 0.0),
    )
    return next_state, result


def step(
    state: JaxEnvState,
    learner_action: JaxAction,
    opponent_action: JaxAction,
    cfg: TaskConfig,
    reward_cfg: RewardConfig,
) -> tuple[JaxEnvState, JaxStepResult]:
    """Advance one two-player JAX Orbit Wars environment by one turn.

    Parameters are pure JAX pytrees so this function can be JIT-compiled or
    vectorized. ``learner_action`` is interpreted for ``state.learner_player``;
    ``opponent_action`` is applied to the other side.
    """

    actions0 = jax.tree.map(
        lambda learner, opponent: jnp.where(
            state.learner_player == 0, learner, opponent
        ),
        learner_action,
        opponent_action,
    )
    actions1 = jax.tree.map(
        lambda learner, opponent: jnp.where(
            state.learner_player == 0, opponent, learner
        ),
        learner_action,
        opponent_action,
    )
    player_actions = jax.tree.map(
        lambda action0, action1: jnp.stack([action0, action1], axis=0),
        actions0,
        actions1,
    )
    return step_multi_player(state, player_actions, cfg, reward_cfg)


def step_multi_player(
    state: JaxEnvState,
    player_actions: JaxAction,
    cfg: TaskConfig,
    reward_cfg: RewardConfig,
) -> tuple[JaxEnvState, JaxStepResult]:
    """Advance a multi-player JAX Orbit Wars environment by one turn.

    ``player_actions`` is a :class:`JaxAction` pytree with a leading player
    dimension on each field, e.g. ``source_id.shape == (player_count,
    max_fleets)``. This mirrors Kaggle's interpreter, which processes one
    action list per player before production, movement, and combat. The existing
    :func:`step` helper remains the two-player learner/opponent convenience API.
    """

    previous_game = state.game
    comet_speed = float(getattr(cfg, "comet_speed", COMET_SPEED))
    planets, initial_planets, comets = _pre_launch_comets(previous_game, comet_speed)
    previous_game = previous_game._replace(
        planets=planets, initial_planets=initial_planets, comets=comets
    )
    planets = previous_game.planets
    fleets = previous_game.fleets
    next_fleet_id = previous_game.next_fleet_id
    player_count = int(getattr(cfg, "player_count", 2))

    def launch_player(player_id, carry):
        planets, fleets, next_fleet_id = carry
        action = jax.tree.map(lambda x: jnp.take(x, player_id, axis=0), player_actions)
        planets, fleets, next_fleet_id = _launch_fleets(
            planets, fleets, next_fleet_id, action, player_id, cfg
        )
        return planets, fleets, next_fleet_id

    planets, fleets, next_fleet_id = jax.lax.fori_loop(
        0,
        player_count,
        lambda player_id, carry: launch_player(player_id, carry),
        (planets, fleets, next_fleet_id),
    )
    return _finish_step(previous_game, state, planets, fleets, next_fleet_id, cfg, reward_cfg)


def _launch_fleets(
    planets: JaxPlanetState,
    fleets: JaxFleetState,
    next_fleet_id: jax.Array,
    action: JaxAction,
    player: int,
    cfg: TaskConfig,
):
    """Launch fleets in slot order, matching Kaggle sequential ``process_moves``."""

    fleet_cap = max_fleets(cfg)
    slot_indices = jnp.arange(fleet_cap, dtype=jnp.int32)
    ship_requests = jnp.floor(action.ships).astype(jnp.float32)
    existing_active = fleets.active.astype(jnp.int32).sum()

    def launch_slot(carry, slot):
        planets, next_id, launched_count = carry
        raw_source_id = action.source_id[slot]
        source_idx = jnp.clip(raw_source_id, 0, MAX_PLANETS - 1)
        planet_id_at = planets.id[source_idx]
        ships = ship_requests[slot]
        remaining = planets.ships[source_idx]
        slot_valid = (
            action.valid[slot]
            & (raw_source_id >= 0)
            & (raw_source_id == planet_id_at)
            & planets.active[source_idx]
            & (planets.owner[source_idx] == player)
            & (ships > 0.0)
            & (remaining >= ships)
            & ((existing_active + launched_count) < fleet_cap)
        )
        updated_ships = planets.ships.at[source_idx].add(
            jnp.where(slot_valid, -ships, 0.0)
        )
        planets = planets._replace(ships=updated_ships)
        launched_count = launched_count + slot_valid.astype(jnp.int32)
        launch = (
            slot_valid,
            next_id,
            planets.x[source_idx]
            + jnp.cos(action.angle[slot])
            * (planets.radius[source_idx] + PLANET_LAUNCH_RADIUS_OFFSET),
            planets.y[source_idx]
            + jnp.sin(action.angle[slot])
            * (planets.radius[source_idx] + PLANET_LAUNCH_RADIUS_OFFSET),
            action.angle[slot],
            action.source_id[slot],
            ships,
        )
        next_id = next_id + slot_valid.astype(jnp.int32)
        return (planets, next_id, launched_count), launch

    (planets, next_fleet_id, _), launches = jax.lax.scan(
        launch_slot,
        (planets, next_fleet_id, jnp.array(0, dtype=jnp.int32)),
        slot_indices,
    )
    valid, launch_ids, lx, ly, lang, lsource, lships = launches
    launched = JaxFleetState(
        id=launch_ids,
        owner=jnp.full((fleet_cap,), player, dtype=jnp.int32),
        x=lx,
        y=ly,
        angle=lang,
        from_planet_id=lsource,
        ships=lships,
        active=valid,
    )
    fleets = _compact_fleets(_concat_fleets(fleets, launched), cfg)
    return planets, fleets, next_fleet_id


def _concat_fleets(a: JaxFleetState, b: JaxFleetState) -> JaxFleetState:
    return JaxFleetState(
        *(jnp.concatenate([x, y], axis=0) for x, y in zip(a, b, strict=True))
    )


def _compact_fleets(fleets: JaxFleetState, cfg: TaskConfig) -> JaxFleetState:
    order = jnp.argsort(jnp.where(fleets.active, 0, 1), stable=True)[: max_fleets(cfg)]
    return jax.tree.map(lambda x: jnp.take(x, order, axis=0), fleets)


def _move_and_resolve(
    previous_game: JaxGameState,
    planets: JaxPlanetState,
    fleets: JaxFleetState,
    cfg: TaskConfig,
):
    comets = previous_game.comets
    is_comet = _is_comet_planet(comets, planets.id)
    old_px, old_py = planets.x, planets.y
    init_dx = previous_game.initial_planets.x - BOARD_CENTER[0]
    init_dy = previous_game.initial_planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates = (
        (orbit_radius + planets.radius < ROTATION_RADIUS_LIMIT)
        & planets.active
        & (~is_comet)
    )
    init_angle = jnp.arctan2(init_dy, init_dx)
    cur_angle = init_angle + previous_game.angular_velocity * previous_game.step.astype(
        jnp.float32
    )
    new_px = jnp.where(
        rotates, BOARD_CENTER[0] + orbit_radius * jnp.cos(cur_angle), planets.x
    )
    new_py = jnp.where(
        rotates, BOARD_CENTER[1] + orbit_radius * jnp.sin(cur_angle), planets.y
    )
    new_px, new_py, comets = _advance_comet_positions(
        comets, planets, old_px, old_py, new_px, new_py
    )
    check_collision = jnp.where(is_comet, old_px >= 0.0, True)

    speed = fleet_speed(
        fleets.ships, float(getattr(cfg, "ship_speed", MAX_FLEET_SPEED))
    )
    old_fx, old_fy = fleets.x, fleets.y
    new_fx = fleets.x + jnp.cos(fleets.angle) * speed
    new_fy = fleets.y + jnp.sin(fleets.angle) * speed

    hits = swept_pair_hit(
        old_fx[:, None],
        old_fy[:, None],
        new_fx[:, None],
        new_fy[:, None],
        old_px[None, :],
        old_py[None, :],
        new_px[None, :],
        new_py[None, :],
        planets.radius[None, :],
    )
    hits = (
        hits
        & fleets.active[:, None]
        & planets.active[None, :]
        & check_collision[None, :]
    )
    hit_any = hits.any(axis=1)
    planet_order = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    hit_idx = jnp.min(jnp.where(hits, planet_order[None, :], MAX_PLANETS), axis=1)
    out = (
        (new_fx < 0.0) | (new_fx > BOARD_SIZE) | (new_fy < 0.0) | (new_fy > BOARD_SIZE)
    )
    sun = (
        point_to_segment_distance_xy(
            BOARD_CENTER[0], BOARD_CENTER[1], old_fx, old_fy, new_fx, new_fy
        )
        < SUN_RADIUS
    )
    remove = hit_any | out | sun | (~fleets.active)

    moved_fleets = fleets._replace(x=new_fx, y=new_fy, active=fleets.active & (~remove))
    moved_planets = planets._replace(x=new_px, y=new_py)
    moved_planets = _resolve_combat(moved_planets, fleets, hit_any, hit_idx, cfg)
    moved_planets, initial_planets, comets = _expire_comets_pre_launch(
        moved_planets, previous_game.initial_planets, comets
    )
    return moved_planets, moved_fleets, comets


def _advance_comet_positions(
    comets: JaxCometState,
    planets: JaxPlanetState,
    old_px: jax.Array,
    old_py: jax.Array,
    new_px: jax.Array,
    new_py: jax.Array,
) -> tuple[jax.Array, jax.Array, JaxCometState]:
    def group_body(g, carry):
        new_px, new_py, comets = carry
        active = comets.group_active[g]
        idx = comets.path_index[g] + jnp.where(active, 1, 0)
        comets = comets._replace(path_index=comets.path_index.at[g].set(idx))

        def comet_body(i, inner):
            new_px, new_py, comets = inner
            pid = comets.planet_ids[g, i]
            path_len = comets.path_lengths[g, i]
            on_group = active & (pid >= 0)
            match = (planets.id == pid) & planets.active
            slot = jnp.argmax(match.astype(jnp.int32))
            in_path = on_group & (idx < path_len) & match.any()
            safe_idx = jnp.clip(idx, 0, jnp.maximum(path_len - 1, 0))
            cx = comets.paths_x[g, i, safe_idx]
            cy = comets.paths_y[g, i, safe_idx]
            new_px = jnp.where(in_path, new_px.at[slot].set(cx), new_px)
            new_py = jnp.where(in_path, new_py.at[slot].set(cy), new_py)
            return new_px, new_py, comets

        return jax.lax.fori_loop(
            0, COMETS_PER_GROUP, comet_body, (new_px, new_py, comets)
        )

    return jax.lax.fori_loop(
        0, MAX_COMET_GROUPS, group_body, (new_px, new_py, comets)
    )


def fleet_speed(ships: jax.Array, ship_speed: float = DEFAULT_SHIP_SPEED) -> jax.Array:
    """Compute Orbit Wars fleet speed from ship count using game scaling."""

    safe = jnp.maximum(ships, 1.0)
    speed = 1.0 + (ship_speed - 1.0) * (jnp.log(safe) / jnp.log(1000.0)) ** 1.5
    return jnp.minimum(speed, ship_speed)


def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, radius):
    """Return whether moving points intersect moving circular targets.

    The arguments are broadcastable coordinate arrays for fleet start/end
    points and planet start/end points over a single simulation step.
    """

    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    disc = b * b - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    denom = jnp.maximum(2.0 * a, 1e-12)
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    linear_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    static_hit = c <= 0.0
    return jnp.where(a < 1e-12, static_hit, linear_hit)


def point_to_segment_distance_xy(px, py, vx, vy, wx, wy):
    """Return the distance from point ``p`` to segment ``v``-``w``."""

    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / jnp.maximum(l2, 1e-12)
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    return jnp.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _resolve_combat(
    planets: JaxPlanetState,
    fleets: JaxFleetState,
    hit_any: jax.Array,
    hit_idx: jax.Array,
    cfg: TaskConfig,
) -> JaxPlanetState:
    hit_weights = jax.nn.one_hot(hit_idx, MAX_PLANETS, dtype=jnp.float32) * hit_any[
        :, None
    ].astype(jnp.float32)
    owners = jnp.arange(int(getattr(cfg, "player_count", 2)), dtype=jnp.int32)
    ships_by_owner = jax.vmap(
        lambda owner: (
            hit_weights.T @ jnp.where(fleets.owner == owner, fleets.ships, 0.0)
        )
    )(owners)
    top_owner_idx = jnp.argmax(ships_by_owner, axis=0)
    top_owner = jnp.take(owners, top_owner_idx)
    top = jnp.max(ships_by_owner, axis=0)
    tied_for_top = (ships_by_owner == top[None, :]) & (top[None, :] > 0.0)
    unique_top = tied_for_top.sum(axis=0) == 1
    second = jnp.max(
        jnp.where(owners[:, None] == top_owner[None, :], -jnp.inf, ships_by_owner),
        axis=0,
    )
    second = jnp.where(jnp.isfinite(second), second, 0.0)
    survivors = jnp.where(unique_top, top - second, 0.0)
    has_attack = survivors > 0.0
    same_owner = planets.owner == top_owner
    new_ships_same = planets.ships + survivors
    after_attack = planets.ships - survivors
    captured = after_attack < 0.0
    new_owner = jnp.where(
        has_attack & (~same_owner) & captured, top_owner, planets.owner
    )
    new_ships = jnp.where(has_attack & same_owner, new_ships_same, planets.ships)
    new_ships = jnp.where(
        has_attack & (~same_owner),
        jnp.where(captured, -after_attack, after_attack),
        new_ships,
    )
    return planets._replace(owner=new_owner.astype(jnp.int32), ships=new_ships)


def _terminal(
    game: JaxGameState,
    learner_player: jax.Array,
    cfg: TaskConfig,
    reward_cfg: RewardConfig,
):
    owners = jnp.arange(int(getattr(cfg, "player_count", 2)), dtype=jnp.int32)
    planet_alive = jax.vmap(
        lambda owner: ((game.planets.owner == owner) & game.planets.active).any()
    )(owners)
    fleet_alive = jax.vmap(
        lambda owner: ((game.fleets.owner == owner) & game.fleets.active).any()
    )(owners)
    alive = planet_alive | fleet_alive
    done = (game.step >= MAX_STEPS - 2) | (alive.astype(jnp.int32).sum() <= 1)
    scores = jax.vmap(
        lambda owner: (
            jnp.where(
                (game.planets.owner == owner) & game.planets.active,
                game.planets.ships,
                0.0,
            ).sum()
            + jnp.where(
                (game.fleets.owner == owner) & game.fleets.active,
                game.fleets.ships,
                0.0,
            ).sum()
        )
    )(owners)
    learner_score = jnp.take(scores, jnp.clip(learner_player, 0, owners.shape[0] - 1))
    best_score = jnp.max(scores)
    total_score = jnp.sum(scores)
    rank = 1.0 + (scores > learner_score).astype(jnp.float32).sum()
    tied = (scores == learner_score).astype(jnp.float32).sum()
    placement = rank + (tied - 1.0) * 0.5
    is_first = ((learner_score == best_score) & (learner_score > 0.0)).astype(
        jnp.float32
    )
    score_share = jnp.where(total_score > 0.0, learner_score / total_score, 0.0)
    learner_idx = jnp.clip(learner_player, 0, scores.shape[0] - 1)
    max_other = jnp.max(
        jnp.where(
            jnp.arange(scores.shape[0], dtype=jnp.int32) == learner_idx,
            -jnp.inf,
            scores,
        )
    )
    max_other = jnp.where(jnp.isfinite(max_other), max_other, 0.0)
    ship_denom = learner_score + max_other
    ship_differential = jnp.where(
        ship_denom > 0.0,
        (learner_score - max_other) / ship_denom,
        0.0,
    )
    player_count = jnp.asarray(owners.shape[0], dtype=jnp.float32)
    ranked_reward = jnp.where(
        player_count > 1.0,
        1.0 - 2.0 * (placement - 1.0) / (player_count - 1.0),
        1.0,
    )
    binary_reward = jnp.where(is_first > 0.0, 1.0, -1.0)
    share_reward = score_share
    survival_time = jnp.minimum(
        game.step.astype(jnp.float32) + 1.0,
        jnp.asarray(MAX_STEPS, dtype=jnp.float32),
    ) / jnp.maximum(jnp.asarray(MAX_STEPS, dtype=jnp.float32), 1.0)
    survival_rank_reward = 0.5 * ranked_reward + 0.5 * survival_time
    mode = reward_cfg.terminal_reward_mode.strip().lower()
    if mode == "binary_win":
        reward = binary_reward
    elif mode == "ranked":
        reward = ranked_reward
    elif mode == "score_share":
        reward = share_reward
    elif mode == "survival_plus_rank":
        reward = survival_rank_reward
    elif mode == "normalized_ship_differential":
        reward = ship_differential
    else:
        raise ValueError(
            "reward.terminal_reward_mode must be one of binary_win, ranked, "
            "score_share, survival_plus_rank, or normalized_ship_differential; "
            f"got {mode!r}."
        )
    reward = apply_early_terminal_reward_shaping_jax(reward, game.step, reward_cfg)
    return (
        done,
        reward,
        rank,
        placement,
        is_first,
        score_share,
        ship_differential,
        survival_time,
    )


def _ship_advantage(game: JaxGameState, player: jax.Array):
    mine_p = (game.planets.owner == player) & game.planets.active
    opp_p = (
        (game.planets.owner != -1)
        & (game.planets.owner != player)
        & game.planets.active
    )
    mine_f = (game.fleets.owner == player) & game.fleets.active
    opp_f = (
        (game.fleets.owner != -1) & (game.fleets.owner != player) & game.fleets.active
    )
    return (
        jnp.where(mine_p, game.planets.ships, 0.0).sum()
        + jnp.where(mine_f, game.fleets.ships, 0.0).sum()
        - jnp.where(opp_p, game.planets.ships, 0.0).sum()
        - jnp.where(opp_f, game.fleets.ships, 0.0).sum()
    )


def _shaping(
    previous: JaxGameState,
    current: JaxGameState,
    player: jax.Array,
    cfg: RewardConfig,
):
    captured = (
        (previous.planets.owner != player)
        & (current.planets.owner == player)
        & current.planets.active
    ).sum()
    lost = (
        (previous.planets.owner == player)
        & (current.planets.owner != player)
        & current.planets.active
    ).sum()
    capture_reward = cfg.reward_capture_planet * (captured - lost).astype(jnp.float32)
    ship_reward = cfg.reward_ship_delta * (
        _ship_advantage(current, player) - _ship_advantage(previous, player)
    )
    prev_prod = jnp.where(
        (previous.planets.owner == player) & previous.planets.active,
        previous.planets.production,
        0.0,
    ).sum()
    cur_prod = jnp.where(
        (current.planets.owner == player) & current.planets.active,
        current.planets.production,
        0.0,
    ).sum()
    prod_reward = cfg.reward_production_delta * (cur_prod - prev_prod)
    return capture_reward, ship_reward, prod_reward


batched_reset = jax.vmap(reset, in_axes=(0, None))
batched_step = jax.vmap(step, in_axes=(0, 0, 0, None, None))
batched_step_multi_player = jax.vmap(step_multi_player, in_axes=(0, 0, None, None))


def jit_reset(key: jax.Array, cfg: TaskConfig):
    """JIT-compiled reset helper for a closed-over ``TaskConfig``."""

    return jax.jit(lambda k: reset(k, cfg))(key)


def jit_step(
    state: JaxEnvState,
    learner_action: JaxAction,
    opponent_action: JaxAction,
    cfg: TaskConfig,
    reward_cfg: RewardConfig,
):
    """JIT-compiled step helper for a closed-over ``TaskConfig``."""

    return jax.jit(lambda s, a0, a1: step(s, a0, a1, cfg, reward_cfg))(
        state, learner_action, opponent_action
    )
