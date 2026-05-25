"""JAX feature encoder v2: planet tensor + top-K edges + global vector."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config.schema import TaskConfig
from src.features.registry_v2 import (
    GLOBAL_V2_FEATURE_SCHEMA,
    feature_history_steps,
)
from src.game.constants import (
    ANGULAR_VELOCITY_NORM,
    BASE_EDGE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_V2_DIM,
    BOARD_CENTER,
    BOARD_SIZE,
    MAX_FLEET_SPEED,
    MAX_PLANETS,
    MAX_PRODUCTION,
    MAX_STEPS,
)
from src.jax.features import (
    _incoming_fleet_pressure,
    is_rotating_xy,
    owner_relative_production,
    shot_crosses_sun_xy,
    target_owner_one_hot,
)


class JaxFeatureHistoryV2(NamedTuple):
    """Global-only history plus prior planet ship counts for deltas."""

    global_features: jax.Array
    planet_ships: jax.Array
    cursor: jax.Array
    length: jax.Array


class JaxTurnBatchV2(NamedTuple):
    """Fixed-shape v2 policy input batch."""

    planet_features: jax.Array
    planet_mask: jax.Array
    edge_features: jax.Array
    edge_mask: jax.Array
    edge_src_ids: jax.Array
    edge_tgt_ids: jax.Array
    global_features: jax.Array
    theta_ref: jax.Array


def ship_feature_scale(env_cfg: TaskConfig) -> jax.Array:
    scale = float(getattr(env_cfg, "ship_feature_scale", env_cfg.max_ships))
    return jnp.asarray(scale, dtype=jnp.float32)


def encode_turn_v2(
    game, env_cfg: TaskConfig, history: JaxFeatureHistoryV2 | None = None
) -> JaxTurnBatchV2:
    """Encode a JAX game state into v2 planet/edge/global tensors."""

    planets = game.planets
    player = game.player
    scale = ship_feature_scale(env_cfg)
    theta_ref = _theta_ref(planets.x, planets.y, planets.owner, player, planets.active)
    planet_features = _planet_features(
        planets, game.fleets, player, env_cfg, scale, theta_ref, history
    )
    edge_features, edge_mask, edge_tgt_ids = _edge_features(
        planets, game.fleets, player, env_cfg, scale, theta_ref
    )
    global_frame = _global_frame(game, env_cfg, scale, history)
    global_features = _stack_global_history_v2(global_frame, history, env_cfg)
    return JaxTurnBatchV2(
        planet_features=planet_features,
        planet_mask=planets.active.astype(jnp.bool_),
        edge_features=edge_features,
        edge_mask=edge_mask,
        edge_src_ids=planets.id.astype(jnp.int32),
        edge_tgt_ids=edge_tgt_ids,
        global_features=global_features,
        theta_ref=theta_ref,
    )


def empty_feature_history_v2(env_cfg: TaskConfig) -> JaxFeatureHistoryV2:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    return JaxFeatureHistoryV2(
        global_features=jnp.zeros(
            (steps, BASE_GLOBAL_FEATURE_V2_DIM), dtype=jnp.float32
        ),
        planet_ships=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        cursor=jnp.array(0, dtype=jnp.int32),
        length=jnp.array(0, dtype=jnp.int32),
    )


def current_feature_snapshot_v2(game, env_cfg: TaskConfig) -> JaxFeatureHistoryV2:
    global_frame = _global_frame(game, env_cfg, ship_feature_scale(env_cfg), None)
    return JaxFeatureHistoryV2(
        global_features=global_frame[None, :],
        planet_ships=game.planets.ships,
        cursor=jnp.array(0, dtype=jnp.int32),
        length=jnp.array(1, dtype=jnp.int32),
    )


def append_feature_history_v2(
    history: JaxFeatureHistoryV2 | None, game, env_cfg: TaskConfig
) -> JaxFeatureHistoryV2:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return empty_feature_history_v2(env_cfg)
    base = empty_feature_history_v2(env_cfg) if history is None else history
    global_frame = _global_frame(game, env_cfg, ship_feature_scale(env_cfg), None)
    idx = base.cursor % steps
    global_features = base.global_features.at[idx].set(global_frame)
    cursor = (base.cursor + 1).astype(jnp.int32)
    length = jnp.minimum(base.length + 1, steps).astype(jnp.int32)
    return JaxFeatureHistoryV2(
        global_features=global_features,
        planet_ships=game.planets.ships,
        cursor=cursor,
        length=length,
    )


def _theta_ref(x, y, owner, player, active):
    owned = active & (owner == player)
    count = owned.astype(jnp.float32).sum()
    cx = jnp.where(count > 0.0, jnp.where(owned, x, 0.0).sum() / count, BOARD_CENTER[0])
    cy = jnp.where(count > 0.0, jnp.where(owned, y, 0.0).sum() / count, BOARD_CENTER[1])
    return jnp.arctan2(cy - BOARD_CENTER[1], cx - BOARD_CENTER[0])


def _rotate_to_learner_frame(x, y, theta_ref):
    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    cos_t = jnp.cos(-theta_ref)
    sin_t = jnp.sin(-theta_ref)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t
    return rx, ry


def _canonical_angle(x, y, theta_ref):
    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    angle = jnp.arctan2(dy, dx) - theta_ref
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle)) / jnp.pi


def _previous_planet_ships(history: JaxFeatureHistoryV2 | None, current_ships):
    if history is None:
        return current_ships
    return history.planet_ships


def _planet_features(
    planets,
    fleets,
    player,
    env_cfg: TaskConfig,
    scale,
    theta_ref,
    history: JaxFeatureHistoryV2 | None,
):
    owner_slot = target_owner_one_hot(planets.owner, player, env_cfg)
    rotating = is_rotating_xy(planets.x, planets.y, planets.radius)
    sun_dx = planets.x - BOARD_CENTER[0]
    sun_dy = planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(sun_dx * sun_dx + sun_dy * sun_dy) / BOARD_SIZE
    orbit_angle = _canonical_angle(planets.x, planets.y, theta_ref)
    prev_ships = _previous_planet_ships(history, planets.ships)
    ship_delta = (planets.ships - prev_ships) / scale
    incoming_friendly, _incoming_enemy = _incoming_fleet_pressure(
        planets.x, planets.y, planets.radius, fleets, player
    )
    features = jnp.stack(
        [
            planets.active.astype(jnp.float32),
            orbit_radius,
            orbit_angle,
            planets.radius / 5.0,
            jnp.minimum(planets.ships, scale) / scale,
            planets.production / MAX_PRODUCTION,
            rotating.astype(jnp.float32),
            owner_slot[..., 0],
            owner_slot[..., 1],
            owner_slot[..., 2],
            owner_slot[..., 3],
            incoming_friendly / scale,
            ship_delta,
        ],
        axis=-1,
    )
    return jnp.where(planets.active[:, None], features, jnp.zeros_like(features))


def _edge_features(planets, fleets, player, env_cfg: TaskConfig, scale, theta_ref):
    p = MAX_PLANETS
    k = max(0, int(env_cfg.candidate_count) - 1)
    if k == 0:
        return (
            jnp.zeros((p, 0, BASE_EDGE_FEATURE_DIM), dtype=jnp.float32),
            jnp.zeros((p, 0), dtype=jnp.bool_),
            jnp.zeros((p, 0), dtype=jnp.int32),
        )

    src_x = planets.x[:, None]
    src_y = planets.y[:, None]
    dx = planets.x[None, :] - src_x
    dy = planets.y[None, :] - src_y
    dist = jnp.sqrt(dx * dx + dy * dy)
    valid_target = planets.active[None, :] & (
        planets.id[None, :] != planets.id[:, None]
    )
    angle_all = jnp.arctan2(dy, dx)
    crosses_all = shot_crosses_sun_xy(
        src_x,
        src_y,
        planets.radius[:, None],
        angle_all,
        planets.x[None, :],
        planets.y[None, :],
    )
    unblocked_valid = valid_target & (~crosses_all)

    sort_distance = jnp.where(valid_target, dist, jnp.inf)
    sort_blocked = jnp.where(valid_target, crosses_all.astype(jnp.int32), 1)
    sort_id = jnp.broadcast_to(planets.id[None, :], dist.shape)
    order = jnp.lexsort((sort_id, sort_distance, sort_blocked), axis=1)[:, :k]
    pad_width = k - order.shape[1]
    if pad_width > 0:
        order = jnp.pad(order, ((0, 0), (0, pad_width)), constant_values=0)
    ordered_valid = jnp.take_along_axis(valid_target, order, axis=1)

    tgt_x = jnp.take(planets.x, order, axis=0)
    tgt_y = jnp.take(planets.y, order, axis=0)
    tgt_radius = jnp.take(planets.radius, order, axis=0)
    tgt_ships = jnp.take(planets.ships, order, axis=0)
    tgt_owner = jnp.take(planets.owner, order, axis=0)
    tgt_active = jnp.take(planets.active, order, axis=0)

    src_rx, src_ry = _rotate_to_learner_frame(src_x, src_y, theta_ref)
    tgt_rx, tgt_ry = _rotate_to_learner_frame(tgt_x, tgt_y, theta_ref)
    delta_x = (tgt_rx - src_rx) / BOARD_SIZE
    delta_y = (tgt_ry - src_ry) / BOARD_SIZE
    distance = jnp.sqrt(delta_x * delta_x + delta_y * delta_y)

    real_dx = tgt_x - src_x
    real_dy = tgt_y - src_y
    angle = jnp.arctan2(real_dy, real_dx)
    crosses = shot_crosses_sun_xy(
        src_x, src_y, planets.radius[:, None], angle, tgt_x, tgt_y
    )

    owner_slot = target_owner_one_hot(tgt_owner, player, env_cfg)
    incoming_friendly, incoming_enemy = _incoming_fleet_pressure(
        tgt_x, tgt_y, tgt_radius, fleets, player
    )
    turns = distance * BOARD_SIZE / jnp.maximum(MAX_FLEET_SPEED, 1e-6) / MAX_STEPS

    edge_rows = jnp.stack(
        [
            delta_x,
            delta_y,
            distance,
            crosses.astype(jnp.float32),
            jnp.minimum(tgt_ships, scale) / scale,
            owner_slot[..., 0],
            owner_slot[..., 1],
            owner_slot[..., 2],
            owner_slot[..., 3],
            turns,
            incoming_friendly / scale,
            incoming_enemy / scale,
        ],
        axis=-1,
    )
    edge_rows = jnp.where(ordered_valid[..., None], edge_rows, 0.0)
    edge_rows = jnp.where(tgt_active[..., None], edge_rows, 0.0)

    owned_source = planets.active & (planets.owner == player)
    edge_mask = ordered_valid & (~crosses) & owned_source[:, None]

    tgt_ids = jnp.where(ordered_valid, jnp.take(planets.id, order, axis=0), -1)
    return edge_rows, edge_mask, tgt_ids.astype(jnp.int32)


def _global_frame(
    game, env_cfg: TaskConfig, scale, history: JaxFeatureHistoryV2 | None
):
    planets = game.planets
    fleets = game.fleets
    player = game.player
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    neutral = planets.active & (planets.owner == -1)
    my_fleet = fleets.active & (fleets.owner == player)
    enemy_fleet = fleets.active & (fleets.owner != player)
    denom = MAX_PLANETS * scale
    owner_production = owner_relative_production(planets, player, env_cfg)
    previous_global, previous_present = _previous_global_features_v2(history, env_cfg)

    owner_ship_totals_slice = GLOBAL_V2_FEATURE_SCHEMA.base_slice(
        "owner_relative_ship_totals"
    )
    owner_planet_counts_slice = GLOBAL_V2_FEATURE_SCHEMA.base_slice(
        "owner_relative_planet_counts"
    )
    owner_fleet_totals_slice = GLOBAL_V2_FEATURE_SCHEMA.base_slice(
        "owner_relative_fleet_totals"
    )
    owner_production_slice = GLOBAL_V2_FEATURE_SCHEMA.base_slice(
        "owner_relative_production"
    )

    player_count_int = max(1, min(4, int(env_cfg.player_count)))
    player_count = jnp.asarray(player_count_int, dtype=jnp.int32)
    planet_slots = (
        planets.owner.astype(jnp.int32) - player.astype(jnp.int32)
    ) % player_count
    valid_planets = (
        planets.active & (planets.owner >= 0) & (planets.owner < player_count)
    )
    owner_counts_raw = jnp.bincount(
        planet_slots,
        weights=valid_planets.astype(jnp.float32),
        length=4,
    )[:4]
    owner_ships_raw = jnp.bincount(
        planet_slots,
        weights=jnp.where(valid_planets, planets.ships, 0.0),
        length=4,
    )[:4]
    fleet_slots = (
        fleets.owner.astype(jnp.int32) - player.astype(jnp.int32)
    ) % player_count
    valid_fleets = fleets.active & (fleets.owner >= 0) & (fleets.owner < player_count)
    owner_fleets_raw = jnp.bincount(
        fleet_slots,
        weights=jnp.where(valid_fleets, fleets.ships, 0.0),
        length=4,
    )[:4]
    owner_counts = owner_counts_raw / MAX_PLANETS
    owner_ships = owner_ships_raw / denom
    owner_fleets = owner_fleets_raw / denom
    active_mask = (jnp.arange(4, dtype=jnp.int32) < player_count).astype(jnp.float32)
    player_count_feature = jnp.asarray(player_count_int / 4.0, dtype=jnp.float32)

    angular_velocity = game.angular_velocity.astype(jnp.float32) / ANGULAR_VELOCITY_NORM

    return jnp.concatenate(
        [
            jnp.asarray(
                [
                    game.step.astype(jnp.float32) / MAX_STEPS,
                    mine.astype(jnp.float32).sum() / MAX_PLANETS,
                    enemy.astype(jnp.float32).sum() / MAX_PLANETS,
                    neutral.astype(jnp.float32).sum() / MAX_PLANETS,
                    jnp.where(mine, planets.ships, 0.0).sum() / denom,
                    jnp.where(enemy, planets.ships, 0.0).sum() / denom,
                    jnp.where(my_fleet, fleets.ships, 0.0).sum() / denom,
                    jnp.where(enemy_fleet, fleets.ships, 0.0).sum() / denom,
                ],
                dtype=jnp.float32,
            ),
            owner_counts,
            owner_ships,
            owner_fleets,
            active_mask,
            jnp.asarray([player_count_feature], dtype=jnp.float32),
            owner_production,
            (owner_ships - previous_global[owner_ship_totals_slice]) * previous_present,
            (owner_counts - previous_global[owner_planet_counts_slice])
            * previous_present,
            (owner_fleets - previous_global[owner_fleet_totals_slice])
            * previous_present,
            (owner_production - previous_global[owner_production_slice])
            * previous_present,
            jnp.asarray([angular_velocity], dtype=jnp.float32),
        ]
    )


def _previous_global_features_v2(
    history: JaxFeatureHistoryV2 | None, env_cfg: TaskConfig
):
    if history is None or feature_history_steps(env_cfg) <= 1 or history.length <= 0:
        return (
            jnp.zeros((BASE_GLOBAL_FEATURE_V2_DIM,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    ordered = _ordered_global_history_v2(history, env_cfg)
    previous = ordered.global_features[-1]
    present = (jnp.abs(previous).sum() > 0.0).astype(jnp.float32)
    return previous, present


def _ordered_global_history_v2(
    history: JaxFeatureHistoryV2, env_cfg: TaskConfig
) -> JaxFeatureHistoryV2:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return history
    start = (history.cursor - history.length) % steps
    indices = (start + jnp.arange(steps, dtype=jnp.int32)) % steps
    valid = jnp.arange(steps, dtype=jnp.int32) >= (steps - history.length)
    return JaxFeatureHistoryV2(
        global_features=history.global_features[indices] * valid[:, None],
        planet_ships=history.planet_ships,
        cursor=history.cursor,
        length=history.length,
    )


def _stack_global_history_v2(
    current: jax.Array, history: JaxFeatureHistoryV2 | None, env_cfg: TaskConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history_v2(env_cfg) if history is None else history
    ordered = _ordered_global_history_v2(base, env_cfg)
    return jnp.concatenate(
        [ordered.global_features, current[None, ...]], axis=0
    ).reshape(-1)
