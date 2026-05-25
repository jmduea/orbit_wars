"""JAX geometry and owner-encoding helpers shared by feature catalogs."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config.schema import TaskConfig
from src.game.constants import (
    BOARD_CENTER,
    MAX_OWNER_FEATURE_PLAYERS,
    MAX_PLANETS,
    MAX_PRODUCTION,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)


def clipped_player_count(env_cfg: TaskConfig) -> int:
    return max(
        1, min(MAX_OWNER_FEATURE_PLAYERS, int(getattr(env_cfg, "player_count", 2)))
    )


def relative_owner_slots(owner, player, player_count: int):
    return (owner.astype(jnp.int32) - player.astype(jnp.int32)) % player_count


def target_owner_one_hot(owner, player, env_cfg: TaskConfig):
    """Encode target owners relative to ``player`` with neutral as all-zero."""

    player_count = clipped_player_count(env_cfg)
    slots = relative_owner_slots(owner, player, player_count)
    valid_owner = (owner >= 0) & (owner < player_count)
    return jax.nn.one_hot(
        jnp.where(valid_owner, slots, 0),
        MAX_OWNER_FEATURE_PLAYERS,
        dtype=jnp.float32,
    ) * valid_owner[..., None].astype(jnp.float32)


def incoming_fleet_pressure(x, y, radius, fleets, player):
    target_x = jnp.asarray(x)[..., None]
    target_y = jnp.asarray(y)[..., None]
    target_radius = jnp.asarray(radius)[..., None]
    dx = target_x - fleets.x
    dy = target_y - fleets.y
    cos_a = jnp.cos(fleets.angle)
    sin_a = jnp.sin(fleets.angle)
    forward = dx * cos_a + dy * sin_a
    closest_x = fleets.x + cos_a * forward
    closest_y = fleets.y + sin_a * forward
    cross_track = jnp.sqrt((target_x - closest_x) ** 2 + (target_y - closest_y) ** 2)
    aims_at_target = fleets.active & (forward >= 0.0) & (cross_track <= target_radius)
    friendly = jnp.where(
        aims_at_target & (fleets.owner == player), fleets.ships, 0.0
    ).sum(axis=-1)
    enemy = jnp.where(aims_at_target & (fleets.owner != player), fleets.ships, 0.0).sum(
        axis=-1
    )
    return friendly, enemy


def owner_relative_production(planets, player, env_cfg: TaskConfig):
    player_count = clipped_player_count(env_cfg)
    slots = relative_owner_slots(planets.owner, player, player_count)
    valid_planets = (
        planets.active & (planets.owner >= 0) & (planets.owner < player_count)
    )
    production = jnp.bincount(
        slots,
        weights=jnp.where(valid_planets, planets.production, 0.0),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]
    return production / (MAX_PLANETS * MAX_PRODUCTION)


def is_rotating_xy(x, y, radius):
    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    return jnp.sqrt(dx * dx + dy * dy) + radius < ROTATION_RADIUS_LIMIT


def shot_crosses_sun_xy(src_x, src_y, src_radius, angle, tgt_x, tgt_y):
    start_x = src_x + jnp.cos(angle) * (src_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = src_y + jnp.sin(angle) * (src_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    return (
        point_to_segment_distance_xy(
            BOARD_CENTER[0], BOARD_CENTER[1], start_x, start_y, tgt_x, tgt_y
        )
        < SUN_RADIUS
    )


def point_to_segment_distance_xy(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / jnp.maximum(l2, 1e-12)
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    return jnp.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def rotate_to_learner_frame(x, y, theta_ref):
    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    cos_t = jnp.cos(-theta_ref)
    sin_t = jnp.sin(-theta_ref)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t
    return rx, ry


def canonical_angle(x, y, theta_ref):
    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    angle = jnp.arctan2(dy, dx) - theta_ref
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle)) / jnp.pi


def theta_ref(x, y, owner, player, active):
    owned = active & (owner == player)
    count = owned.astype(jnp.float32).sum()
    cx = jnp.where(count > 0.0, jnp.where(owned, x, 0.0).sum() / count, BOARD_CENTER[0])
    cy = jnp.where(count > 0.0, jnp.where(owned, y, 0.0).sum() / count, BOARD_CENTER[1])
    return jnp.arctan2(cy - BOARD_CENTER[1], cx - BOARD_CENTER[0])
