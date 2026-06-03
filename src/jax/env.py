"""JAX-native Orbit Wars environment.

This module provides a pure, fixed-shape implementation of the core Orbit Wars
mechanics used by the training code.  It intentionally stores planets and fleets
as padded arrays so ``reset``/``step`` can be composed with ``jax.vmap`` and
``jax.jit``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config import RewardConfig, TaskConfig
from src.game.constants import (
    BOARD_SIZE,
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
    planets: JaxPlanetState
    initial_planets: JaxPlanetState
    fleets: JaxFleetState


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


def reset(
    key: jax.Array, cfg: TaskConfig
) -> tuple[JaxEnvState, TurnBatch]:
    """Create a deterministic initial board from a JAX PRNG key."""

    initial_planet_count = MAX_PLANETS - TOTAL_COMETS
    fleet_count = max_fleets(cfg)
    group_count = max(1, initial_planet_count // 4)
    active_count = group_count * 4

    idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    group = idx // 4
    quadrant = idx % 4
    active = idx < active_count

    key_angle, key_radius, key_prod, key_ships, key_home, key_vel = jax.random.split(
        key, 6
    )
    base_angles = jax.random.uniform(
        key_angle, (group_count,), minval=0.18, maxval=1.39
    )
    # Keep a mix of rotating and static planets while remaining clear of the sun.
    base_orbit = jax.random.uniform(
        key_radius, (group_count,), minval=22.0, maxval=62.0
    )
    prod_group = jax.random.randint(
        key_prod, (group_count,), minval=1, maxval=6
    ).astype(jnp.float32)
    ships_group = jax.random.randint(
        key_ships, (group_count,), minval=5, maxval=31
    ).astype(jnp.float32)
    radius_group = 1.0 + jnp.log(prod_group)

    safe_group = jnp.minimum(group, group_count - 1)
    theta = jnp.take(base_angles, safe_group)
    orbit = jnp.take(base_orbit, safe_group)
    base_x = 50.0 + orbit * jnp.cos(theta)
    base_y = 50.0 + orbit * jnp.sin(theta)
    x = jnp.where(
        quadrant == 0,
        base_y,
        jnp.where(
            quadrant == 1,
            100.0 - base_x,
            jnp.where(quadrant == 2, base_x, 100.0 - base_y),
        ),
    )
    y = jnp.where(
        quadrant == 0,
        base_x,
        jnp.where(
            quadrant == 1,
            base_y,
            jnp.where(quadrant == 2, 100.0 - base_y, 100.0 - base_x),
        ),
    )
    production = jnp.where(active, jnp.take(prod_group, safe_group), 0.0)
    ships = jnp.where(active, jnp.take(ships_group, safe_group), 0.0)
    radius = jnp.where(active, jnp.take(radius_group, safe_group), 0.0)

    owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32)
    home_group = jax.random.randint(key_home, (), minval=0, maxval=group_count)
    home = (group == home_group) & active
    if int(getattr(cfg, "player_count", 2)) == 4:
        owner = jnp.where(home, quadrant, owner)
        ships = jnp.where(home, 10.0, ships)
    else:
        owner = jnp.where(home & (quadrant == 0), 0, owner)
        owner = jnp.where(home & (quadrant == 3), 1, owner)
        ships = jnp.where(home & ((quadrant == 0) | (quadrant == 3)), 10.0, ships)

    planets = JaxPlanetState(idx, owner, x, y, radius, ships, production, active)
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
        planets=planets,
        initial_planets=planets,
        fleets=empty_fleets,
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
    planets, fleets = _move_and_resolve(previous_game, planets, fleets, cfg)

    next_game = previous_game._replace(
        step=previous_game.step + jnp.array(1, dtype=jnp.int32),
        next_fleet_id=next_fleet_id,
        planets=planets,
        fleets=fleets,
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

    previous_game = state.game
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

    planets, fleets, next_fleet_id = _launch_fleets(
        previous_game.planets,
        previous_game.fleets,
        previous_game.next_fleet_id,
        actions0,
        0,
        cfg,
    )
    planets, fleets, next_fleet_id = _launch_fleets(
        planets, fleets, next_fleet_id, actions1, 1, cfg
    )
    return _finish_step(previous_game, state, planets, fleets, next_fleet_id, cfg, reward_cfg)


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
    source_idx = jnp.clip(action.source_id, 0, MAX_PLANETS - 1)
    source_owner = jnp.take(planets.owner, source_idx)
    source_active = jnp.take(planets.active, source_idx)
    source_ships = jnp.take(planets.ships, source_idx)
    valid = (
        action.valid
        & source_active
        & (source_owner == player)
        & (action.ships > 0.0)
        & (source_ships >= action.ships)
    )

    launched_by_planet = jax.nn.one_hot(
        source_idx, MAX_PLANETS, dtype=jnp.float32
    ).T @ jnp.where(valid, action.ships, 0.0)
    planets = planets._replace(
        ships=jnp.where(
            planets.active, planets.ships - launched_by_planet, planets.ships
        )
    )

    start_x = jnp.take(planets.x, source_idx) + jnp.cos(action.angle) * (
        jnp.take(planets.radius, source_idx) + PLANET_LAUNCH_RADIUS_OFFSET
    )
    start_y = jnp.take(planets.y, source_idx) + jnp.sin(action.angle) * (
        jnp.take(planets.radius, source_idx) + PLANET_LAUNCH_RADIUS_OFFSET
    )
    slots = jnp.arange(max_fleets(cfg), dtype=jnp.int32)
    launched = JaxFleetState(
        id=next_fleet_id + slots,
        owner=jnp.full_like(slots, player),
        x=start_x,
        y=start_y,
        angle=action.angle,
        from_planet_id=action.source_id,
        ships=action.ships,
        active=valid,
    )
    fleets = _compact_fleets(_concat_fleets(fleets, launched), cfg)
    return planets, fleets, next_fleet_id + valid.astype(jnp.int32).sum()


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
    old_px, old_py = planets.x, planets.y
    init_dx = previous_game.initial_planets.x - BOARD_CENTER[0]
    init_dy = previous_game.initial_planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates = (orbit_radius + planets.radius < ROTATION_RADIUS_LIMIT) & planets.active
    init_angle = jnp.arctan2(init_dy, init_dx)
    cur_angle = init_angle + previous_game.angular_velocity * (
        previous_game.step + 1
    ).astype(jnp.float32)
    new_px = jnp.where(
        rotates, BOARD_CENTER[0] + orbit_radius * jnp.cos(cur_angle), planets.x
    )
    new_py = jnp.where(
        rotates, BOARD_CENTER[1] + orbit_radius * jnp.sin(cur_angle), planets.y
    )

    speed = fleet_speed(fleets.ships, MAX_FLEET_SPEED)
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
    hits = hits & fleets.active[:, None] & planets.active[None, :]
    hit_any = hits.any(axis=1)
    hit_idx = jnp.argmax(hits, axis=1)
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
    return moved_planets, moved_fleets


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
