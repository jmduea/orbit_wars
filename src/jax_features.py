"""JAX feature encoder for fixed-shape Orbit Wars states."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import EnvConfig
from .features import (
    BOARD_CENTER,
    MAX_OWNER_FEATURE_PLAYERS,
    NO_OP_CANDIDATE_INDEX,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    BASE_CANDIDATE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_DIM,
    BASE_SELF_FEATURE_DIM,
)


class JaxFeatureHistory(NamedTuple):
    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array


class JaxTurnBatch(NamedTuple):
    """Fixed-shape decision batch emitted by the JAX feature encoder.

    Leading dimensions are ``(num_envs, max_planets, ...)`` after batching or
    ``(max_planets, ...)`` for a single environment. ``decision_mask`` marks
    rows that correspond to learner-owned source planets; padding and non-source
    rows remain present to preserve static shapes.
    """

    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    decision_mask: jax.Array
    source_ids: jax.Array
    source_ships: jax.Array
    candidate_ids: jax.Array
    target_angles: jax.Array


def encode_turn(
    game, env_cfg: EnvConfig, history: JaxFeatureHistory | None = None
) -> JaxTurnBatch:
    """Encode a JAX game state into fixed-shape policy inputs.

    The encoder mirrors the Torch/Python feature schema while avoiding dynamic
    lists. Candidate slot ``0`` is reserved for no-op and real targets are sorted
    into the remaining candidate slots using a JAX-friendly distance key.
    """

    planets = game.planets
    player = game.player
    mine = planets.active & (planets.owner == player)
    global_features = _global_features(game, env_cfg)
    self_features = _self_features(planets, game.fleets, player, env_cfg)
    candidate_ids, candidate_features, candidate_mask, target_angles = (
        _candidate_features(planets, player, env_cfg)
    )
    return JaxTurnBatch(
        self_features=_stack_self_history(self_features, history, env_cfg),
        candidate_features=_stack_candidate_history(
            candidate_features, history, env_cfg
        ),
        global_features=jnp.repeat(
            _stack_global_history(global_features, history, env_cfg)[None, :],
            env_cfg.max_planets,
            axis=0,
        ),
        candidate_mask=candidate_mask & mine[:, None],
        decision_mask=mine,
        source_ids=planets.id,
        source_ships=jnp.maximum(planets.ships, 0.0),
        candidate_ids=candidate_ids,
        target_angles=target_angles,
    )


def feature_history_steps(env_cfg: EnvConfig) -> int:
    return max(1, int(getattr(env_cfg, "feature_history_steps", 1)))


def empty_feature_history(env_cfg: EnvConfig) -> JaxFeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    return JaxFeatureHistory(
        self_features=jnp.zeros(
            (steps, env_cfg.max_planets, BASE_SELF_FEATURE_DIM), dtype=jnp.float32
        ),
        candidate_features=jnp.zeros(
            (
                steps,
                env_cfg.max_planets,
                env_cfg.candidate_count,
                BASE_CANDIDATE_FEATURE_DIM,
            ),
            dtype=jnp.float32,
        ),
        global_features=jnp.zeros((steps, BASE_GLOBAL_FEATURE_DIM), dtype=jnp.float32),
    )


def current_feature_snapshot(game, env_cfg: EnvConfig) -> JaxFeatureHistory:
    candidates = _candidate_features(game.planets, game.player, env_cfg)[1]
    return JaxFeatureHistory(
        self_features=_self_features(game.planets, game.fleets, game.player, env_cfg)[
            None, ...
        ],
        candidate_features=candidates[None, ...],
        global_features=_global_features(game, env_cfg)[None, ...],
    )


def append_feature_history(
    history: JaxFeatureHistory | None, game, env_cfg: EnvConfig
) -> JaxFeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return empty_feature_history(env_cfg)
    base = empty_feature_history(env_cfg) if history is None else history
    current = current_feature_snapshot(game, env_cfg)
    return JaxFeatureHistory(
        self_features=jnp.concatenate(
            [base.self_features, current.self_features], axis=0
        )[-steps:],
        candidate_features=jnp.concatenate(
            [base.candidate_features, current.candidate_features], axis=0
        )[-steps:],
        global_features=jnp.concatenate(
            [base.global_features, current.global_features], axis=0
        )[-steps:],
    )


def _stack_self_history(
    current: jax.Array, history: JaxFeatureHistory | None, env_cfg: EnvConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    return jnp.transpose(
        jnp.concatenate([base.self_features, current[None, ...]], axis=0), (1, 0, 2)
    ).reshape(env_cfg.max_planets, -1)


def _stack_candidate_history(
    current: jax.Array, history: JaxFeatureHistory | None, env_cfg: EnvConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    stacked = jnp.concatenate([base.candidate_features, current[None, ...]], axis=0)
    return jnp.transpose(stacked, (1, 2, 0, 3)).reshape(
        env_cfg.max_planets, env_cfg.candidate_count, -1
    )


def _stack_global_history(
    current: jax.Array, history: JaxFeatureHistory | None, env_cfg: EnvConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    return jnp.concatenate([base.global_features, current[None, ...]], axis=0).reshape(
        -1
    )


def _self_features(planets, fleets, player, env_cfg: EnvConfig):
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    my_count = mine.astype(jnp.float32).sum()
    enemy_count = enemy.astype(jnp.float32).sum()
    my_ships = jnp.where(mine, planets.ships, 0.0).sum()
    enemy_ships = jnp.where(enemy, planets.ships, 0.0).sum()
    owner_counts, owner_ships, _owner_fleets, active_mask, player_count_feature = (
        owner_relative_summary(planets, fleets, player, env_cfg)
    )
    p = env_cfg.max_planets
    owner_context = jnp.broadcast_to(
        jnp.concatenate(
            [
                owner_counts,
                owner_ships,
                active_mask,
                jnp.asarray([player_count_feature], dtype=jnp.float32),
            ]
        ),
        (p, 13),
    )
    rotating = is_rotating_xy(planets.x, planets.y, planets.radius)
    base_features = jnp.stack(
        [
            planets.active.astype(jnp.float32),
            planets.x / env_cfg.board_size,
            planets.y / env_cfg.board_size,
            planets.radius / 5.0,
            jnp.minimum(planets.ships, env_cfg.max_ships) / env_cfg.max_ships,
            planets.production / env_cfg.max_production,
            rotating.astype(jnp.float32),
            jnp.full_like(planets.x, my_count / env_cfg.max_planets),
            jnp.full_like(planets.x, enemy_count / env_cfg.max_planets),
            jnp.full_like(
                planets.x, my_ships / (env_cfg.max_planets * env_cfg.max_ships)
            ),
            jnp.full_like(
                planets.x, enemy_ships / (env_cfg.max_planets * env_cfg.max_ships)
            ),
        ],
        axis=-1,
    )
    features = jnp.concatenate([base_features, owner_context], axis=-1)
    return jnp.where(planets.active[:, None], features, jnp.zeros_like(features))


def _candidate_features(planets, player, env_cfg: EnvConfig):
    p = env_cfg.max_planets
    c = env_cfg.candidate_count
    src_x = planets.x[:, None]
    src_y = planets.y[:, None]
    dx = planets.x[None, :] - src_x
    dy = planets.y[None, :] - src_y
    dist = jnp.sqrt(dx * dx + dy * dy)
    valid_target = planets.active[None, :] & (
        planets.id[None, :] != planets.id[:, None]
    )
    # JAX-friendly approximation of the Python encoder's sorted candidate list:
    # closest active non-self planets fill slots 1..N while slot 0 remains no-op.
    sort_distance = jnp.where(valid_target, dist, jnp.inf)
    sort_id = jnp.broadcast_to(planets.id[None, :], dist.shape)
    order = jnp.lexsort((sort_id, sort_distance), axis=1)[:, : max(0, c - 1)]
    pad_width = max(0, c - 1) - order.shape[1]
    if pad_width:
        order = jnp.pad(order, ((0, 0), (0, pad_width)), constant_values=0)

    tgt_x = jnp.take(planets.x, order, axis=0)
    tgt_y = jnp.take(planets.y, order, axis=0)
    tgt_owner = jnp.take(planets.owner, order, axis=0)
    tgt_ships = jnp.take(planets.ships, order, axis=0)
    tgt_prod = jnp.take(planets.production, order, axis=0)
    tgt_radius = jnp.take(planets.radius, order, axis=0)
    tgt_active = jnp.take(planets.active, order, axis=0)
    real_dx = tgt_x - src_x
    real_dy = tgt_y - src_y
    angle = jnp.arctan2(real_dy, real_dx)
    crosses = shot_crosses_sun_xy(
        src_x, src_y, planets.radius[:, None], angle, tgt_x, tgt_y
    )
    base_real_features = jnp.stack(
        [
            tgt_active.astype(jnp.float32),
            (tgt_owner == -1).astype(jnp.float32),
            (tgt_owner == player).astype(jnp.float32),
            ((tgt_owner != -1) & (tgt_owner != player)).astype(jnp.float32),
            tgt_x / env_cfg.board_size,
            tgt_y / env_cfg.board_size,
            real_dx / env_cfg.board_size,
            real_dy / env_cfg.board_size,
            jnp.sqrt(real_dx * real_dx + real_dy * real_dy) / env_cfg.board_size,
            jnp.minimum(tgt_ships, env_cfg.max_ships) / env_cfg.max_ships,
            tgt_prod / env_cfg.max_production,
            is_rotating_xy(tgt_x, tgt_y, tgt_radius).astype(jnp.float32),
            crosses.astype(jnp.float32),
            jnp.broadcast_to(
                jnp.minimum(planets.ships[:, None], env_cfg.max_ships)
                / env_cfg.max_ships,
                tgt_x.shape,
            ),
        ],
        axis=-1,
    )
    real_features = jnp.concatenate(
        [base_real_features, target_owner_one_hot(tgt_owner, player, env_cfg)], axis=-1
    )
    noop_features = jnp.zeros((p, 1, BASE_CANDIDATE_FEATURE_DIM), dtype=jnp.float32)
    features = jnp.concatenate([noop_features, real_features], axis=1)
    noop_ids = jnp.full((p, 1), -1, dtype=jnp.int32)
    ids = jnp.concatenate([noop_ids, jnp.take(planets.id, order, axis=0)], axis=1)
    angles = jnp.concatenate([jnp.zeros((p, 1), dtype=jnp.float32), angle], axis=1)
    noop_mask = (
        jnp.ones((p, 1), dtype=bool)
        if c > NO_OP_CANDIDATE_INDEX
        else jnp.zeros((p, 0), dtype=bool)
    )
    real_mask = tgt_active & (~crosses)
    mask = jnp.concatenate([noop_mask, real_mask], axis=1)[:, :c]
    return ids[:, :c], features[:, :c, :], mask, angles[:, :c]


def _global_features(game, env_cfg: EnvConfig):
    planets = game.planets
    fleets = game.fleets
    player = game.player
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    neutral = planets.active & (planets.owner == -1)
    my_fleet = fleets.active & (fleets.owner == player)
    enemy_fleet = fleets.active & (fleets.owner != player)
    denom = env_cfg.max_planets * env_cfg.max_ships
    owner_counts, owner_ships, owner_fleets, active_mask, player_count_feature = (
        owner_relative_summary(planets, fleets, player, env_cfg)
    )
    base_features = jnp.asarray(
        [
            game.step.astype(jnp.float32) / env_cfg.episode_steps,
            mine.astype(jnp.float32).sum() / env_cfg.max_planets,
            enemy.astype(jnp.float32).sum() / env_cfg.max_planets,
            neutral.astype(jnp.float32).sum() / env_cfg.max_planets,
            jnp.where(mine, planets.ships, 0.0).sum() / denom,
            jnp.where(enemy, planets.ships, 0.0).sum() / denom,
            jnp.where(my_fleet, fleets.ships, 0.0).sum() / denom,
            jnp.where(enemy_fleet, fleets.ships, 0.0).sum() / denom,
        ],
        dtype=jnp.float32,
    )
    return jnp.concatenate(
        [
            base_features,
            owner_counts,
            owner_ships,
            owner_fleets,
            active_mask,
            jnp.asarray([player_count_feature], dtype=jnp.float32),
        ]
    )


def owner_relative_summary(planets, fleets, player, env_cfg: EnvConfig):
    """Return fixed-size owner-relative summaries for up to four players."""

    player_count = clipped_player_count(env_cfg)
    denom = env_cfg.max_planets * env_cfg.max_ships

    planet_slots = relative_owner_slots(planets.owner, player, player_count)
    valid_planets = (
        planets.active & (planets.owner >= 0) & (planets.owner < player_count)
    )
    owner_counts = jnp.bincount(
        planet_slots,
        weights=valid_planets.astype(jnp.float32),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]
    owner_ships = jnp.bincount(
        planet_slots,
        weights=jnp.where(valid_planets, planets.ships, 0.0),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]

    fleet_slots = relative_owner_slots(fleets.owner, player, player_count)
    valid_fleets = fleets.active & (fleets.owner >= 0) & (fleets.owner < player_count)
    owner_fleets = jnp.bincount(
        fleet_slots,
        weights=jnp.where(valid_fleets, fleets.ships, 0.0),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]
    active_mask = (
        jnp.arange(MAX_OWNER_FEATURE_PLAYERS, dtype=jnp.int32) < player_count
    ).astype(jnp.float32)
    return (
        owner_counts / env_cfg.max_planets,
        owner_ships / denom,
        owner_fleets / denom,
        active_mask,
        jnp.asarray(player_count / MAX_OWNER_FEATURE_PLAYERS, dtype=jnp.float32),
    )


def target_owner_one_hot(owner, player, env_cfg: EnvConfig):
    """Encode target owners relative to ``player`` with neutral as all-zero."""

    player_count = clipped_player_count(env_cfg)
    slots = relative_owner_slots(owner, player, player_count)
    valid_owner = (owner >= 0) & (owner < player_count)
    return jax.nn.one_hot(
        jnp.where(valid_owner, slots, 0),
        MAX_OWNER_FEATURE_PLAYERS,
        dtype=jnp.float32,
    ) * valid_owner[..., None].astype(jnp.float32)


def clipped_player_count(env_cfg: EnvConfig) -> int:
    return max(
        1, min(MAX_OWNER_FEATURE_PLAYERS, int(getattr(env_cfg, "player_count", 2)))
    )


def relative_owner_slots(owner, player, player_count: int):
    return (owner.astype(jnp.int32) - player.astype(jnp.int32)) % player_count


def is_rotating_xy(x, y, radius):
    """Return whether planets at ``(x, y)`` are inside the rotating orbit band."""

    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    return jnp.sqrt(dx * dx + dy * dy) + radius < ROTATION_RADIUS_LIMIT


def shot_crosses_sun_xy(src_x, src_y, src_radius, angle, tgt_x, tgt_y):
    """Return whether a launch ray from source to target intersects the sun."""

    start_x = src_x + jnp.cos(angle) * (src_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = src_y + jnp.sin(angle) * (src_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    return (
        point_to_segment_distance_xy(
            BOARD_CENTER[0], BOARD_CENTER[1], start_x, start_y, tgt_x, tgt_y
        )
        < SUN_RADIUS
    )


def point_to_segment_distance_xy(px, py, vx, vy, wx, wy):
    """Return the distance from point ``p`` to segment ``v``-``w``."""

    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / jnp.maximum(l2, 1e-12)
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    return jnp.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
