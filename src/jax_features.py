"""JAX feature encoder for fixed-shape Orbit Wars states."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import EnvConfig
from .features import (
    BOARD_CENTER,
    NO_OP_CANDIDATE_INDEX,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    candidate_feature_dim,
)


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


def encode_turn(game, env_cfg: EnvConfig) -> JaxTurnBatch:
    """Encode a JAX game state into fixed-shape policy inputs.

    The encoder mirrors the Torch/Python feature schema while avoiding dynamic
    lists. Candidate slot ``0`` is reserved for no-op and real targets are sorted
    into the remaining candidate slots using a JAX-friendly distance key.
    """

    planets = game.planets
    player = game.player
    mine = planets.active & (planets.owner == player)
    global_features = _global_features(game, env_cfg)
    self_features = _self_features(planets, player, env_cfg)
    candidate_ids, candidate_features, candidate_mask, target_angles = (
        _candidate_features(planets, player, env_cfg)
    )
    return JaxTurnBatch(
        self_features=self_features,
        candidate_features=candidate_features,
        global_features=jnp.repeat(
            global_features[None, :], env_cfg.max_planets, axis=0
        ),
        candidate_mask=candidate_mask & mine[:, None],
        decision_mask=mine,
        source_ids=planets.id,
        source_ships=jnp.maximum(planets.ships, 0.0),
        candidate_ids=candidate_ids,
        target_angles=target_angles,
    )


def _self_features(planets, player, env_cfg: EnvConfig):
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    my_count = mine.astype(jnp.float32).sum()
    enemy_count = enemy.astype(jnp.float32).sum()
    my_ships = jnp.where(mine, planets.ships, 0.0).sum()
    enemy_ships = jnp.where(enemy, planets.ships, 0.0).sum()
    rotating = is_rotating_xy(planets.x, planets.y, planets.radius)
    features = jnp.stack(
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
    sort_key = jnp.where(
        valid_target, dist * (p + 1.0) + planets.id[None, :].astype(jnp.float32), 1e9
    )
    order = jnp.argsort(sort_key, axis=1)[:, : max(0, c - 1)]
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
    real_features = jnp.stack(
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
    noop_features = jnp.zeros((p, 1, candidate_feature_dim()), dtype=jnp.float32)
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
    return jnp.asarray(
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
