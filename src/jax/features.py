"""JAX feature encoder: planet tensor + top-K edges + global vector."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config.schema import TaskConfig
from src.features.catalog._types import EdgeRowAssemblyContext
from src.features.catalog.edge import assemble_edge_rows, edge_feature_catalog_for
from src.features.catalog.global_ import (
    GLOBAL_FEATURE_CATALOG,
    assemble_global_frame,
    build_global_context,
)
from src.features.catalog.planet import assemble_planet_features, build_planet_context
from src.features.registry import feature_history_steps
from src.game.constants import (
    BOARD_CENTER,
    BOARD_SIZE,
    MAX_PLANETS,
    MAX_STEPS,
    ROTATION_RADIUS_LIMIT,
)
from src.jax.feature_primitives import (
    incoming_fleet_pressure,
    orbital_position_at_step_jax,
    rotate_to_learner_frame,
    shot_crosses_sun_xy,
    target_owner_one_hot,
    theta_ref,
)


class FeatureHistory(NamedTuple):
    """Global-only history plus prior planet ship counts for deltas."""

    global_features: jax.Array
    planet_ships: jax.Array
    cursor: jax.Array
    length: jax.Array


class TurnBatch(NamedTuple):
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


def encode_turn(
    game, env_cfg: TaskConfig, history: FeatureHistory | None = None
) -> TurnBatch:
    """Encode a JAX game state into v2 planet/edge/global tensors."""

    planets = game.planets
    player = game.player
    scale = ship_feature_scale(env_cfg)
    theta = theta_ref(planets.x, planets.y, planets.owner, player, planets.active)
    planet_features = _planet_features(
        planets, game.fleets, player, env_cfg, scale, theta, history
    )
    edge_features, edge_mask, edge_tgt_ids = _edge_features(game, env_cfg, scale, theta)
    global_frame = _global_frame(game, env_cfg, scale, history)
    global_features = _stack_global_history(global_frame, history, env_cfg)
    return TurnBatch(
        planet_features=planet_features,
        planet_mask=planets.active.astype(jnp.bool_),
        edge_features=edge_features,
        edge_mask=edge_mask,
        edge_src_ids=planets.id.astype(jnp.int32),
        edge_tgt_ids=edge_tgt_ids,
        global_features=global_features,
        theta_ref=theta,
    )


def empty_feature_history(env_cfg: TaskConfig) -> FeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    return FeatureHistory(
        global_features=jnp.zeros(
            (steps, GLOBAL_FEATURE_CATALOG.base_dim), dtype=jnp.float32
        ),
        planet_ships=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        cursor=jnp.array(0, dtype=jnp.int32),
        length=jnp.array(0, dtype=jnp.int32),
    )


def current_feature_snapshot(game, env_cfg: TaskConfig) -> FeatureHistory:
    global_frame = _global_frame(game, env_cfg, ship_feature_scale(env_cfg), None)
    return FeatureHistory(
        global_features=global_frame[None, :],
        planet_ships=game.planets.ships,
        cursor=jnp.array(0, dtype=jnp.int32),
        length=jnp.array(1, dtype=jnp.int32),
    )


def append_feature_history(
    history: FeatureHistory | None, game, env_cfg: TaskConfig
) -> FeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return empty_feature_history(env_cfg)
    base = empty_feature_history(env_cfg) if history is None else history
    global_frame = _global_frame(game, env_cfg, ship_feature_scale(env_cfg), None)
    idx = base.cursor % steps
    global_features = base.global_features.at[idx].set(global_frame)
    cursor = (base.cursor + 1).astype(jnp.int32)
    length = jnp.minimum(base.length + 1, steps).astype(jnp.int32)
    return FeatureHistory(
        global_features=global_features,
        planet_ships=game.planets.ships,
        cursor=cursor,
        length=length,
    )


def _previous_planet_ships(history: FeatureHistory | None, current_ships):
    if history is None:
        return current_ships
    return history.planet_ships


def _planet_features(
    planets,
    fleets,
    player,
    env_cfg: TaskConfig,
    scale,
    theta_ref_value,
    history: FeatureHistory | None,
):
    context = build_planet_context(
        planets,
        fleets,
        player,
        env_cfg,
        scale,
        theta_ref_value,
        _previous_planet_ships(history, planets.ships),
    )
    return assemble_planet_features(context)


def _intercept_min_distance_matrix(
    game,
    planets,
    env_cfg: TaskConfig,
    *,
    src_x: jax.Array,
    src_y: jax.Array,
    dist: jax.Array,
    valid_target: jax.Array,
) -> jax.Array:
    """Minimum raw intercept distance across anchor speeds for every source-target pair."""

    anchor_speeds = tuple(float(s) for s in env_cfg.intercept_anchors)
    tgt_x = planets.x[None, :]
    tgt_y = planets.y[None, :]
    tgt_initial_x = game.initial_planets.x[None, :]
    tgt_initial_y = game.initial_planets.y[None, :]
    tgt_radius = planets.radius[None, :]
    tgt_active = planets.active[None, :]

    init_dx = tgt_initial_x - BOARD_CENTER[0]
    init_dy = tgt_initial_y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates = (orbit_radius + tgt_radius < ROTATION_RADIUS_LIMIT) & tgt_active
    start_angle = jnp.arctan2(init_dy, init_dx)

    min_dist = jnp.full(dist.shape, jnp.inf, dtype=jnp.float32)
    for anchor_speed in anchor_speeds:
        speed = jnp.asarray(anchor_speed, dtype=jnp.float32)
        tau = jnp.maximum(dist / jnp.maximum(speed, 1e-6), 0.0)
        step_index = game.step.astype(jnp.float32) + tau
        tgt_future_x, tgt_future_y = orbital_position_at_step_jax(
            start_angle,
            orbit_radius,
            game.angular_velocity,
            step_index,
            rotates,
            tgt_x,
            tgt_y,
        )
        raw_future_dx = tgt_future_x - src_x
        raw_future_dy = tgt_future_y - src_y
        intercept_dist = jnp.sqrt(
            raw_future_dx * raw_future_dx + raw_future_dy * raw_future_dy
        )
        min_dist = jnp.minimum(min_dist, intercept_dist.astype(jnp.float32))

    return jnp.where(valid_target, min_dist, jnp.inf)


def _edge_features(game, env_cfg: TaskConfig, scale, theta_ref_value):
    planets = game.planets
    fleets = game.fleets
    player = game.player
    p = MAX_PLANETS
    k = max(0, int(env_cfg.candidate_count) - 1)
    edge_dim = edge_feature_catalog_for(env_cfg).base_dim
    if k == 0:
        return (
            jnp.zeros((p, 0, edge_dim), dtype=jnp.float32),
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

    sort_distance = jnp.where(valid_target, dist, jnp.inf)
    rank_mode = str(env_cfg.edge_rank_mode).strip().lower()
    if rank_mode == "intercept_min":
        sort_distance = _intercept_min_distance_matrix(
            game,
            planets,
            env_cfg,
            src_x=src_x,
            src_y=src_y,
            dist=dist,
            valid_target=valid_target,
        )
    elif rank_mode != "snapshot":
        raise ValueError(
            f"Unsupported task.edge_rank_mode {env_cfg.edge_rank_mode!r}. "
            "Expected 'snapshot' or 'intercept_min'."
        )
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
    tgt_production = jnp.take(planets.production, order, axis=0)
    tgt_owner = jnp.take(planets.owner, order, axis=0)
    tgt_active = jnp.take(planets.active, order, axis=0)

    tgt_initial_x = jnp.take(game.initial_planets.x, order, axis=0)
    tgt_initial_y = jnp.take(game.initial_planets.y, order, axis=0)
    tgt_radius_per_edge = tgt_radius
    tgt_active_per_edge = tgt_active
    init_dx = tgt_initial_x - BOARD_CENTER[0]
    init_dy = tgt_initial_y - BOARD_CENTER[1]
    target_orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates_per_edge = (
        target_orbit_radius + tgt_radius_per_edge < ROTATION_RADIUS_LIMIT
    ) & tgt_active_per_edge
    start_angle_per_edge = jnp.arctan2(init_dy, init_dx)

    src_rx, src_ry = rotate_to_learner_frame(src_x, src_y, theta_ref_value)
    tgt_rx, tgt_ry = rotate_to_learner_frame(tgt_x, tgt_y, theta_ref_value)
    snapshot_delta_x = (tgt_rx - src_rx) / BOARD_SIZE
    snapshot_delta_y = (tgt_ry - src_ry) / BOARD_SIZE

    real_dx = tgt_x - src_x
    real_dy = tgt_y - src_y
    snapshot_distance_raw = jnp.sqrt(real_dx * real_dx + real_dy * real_dy)
    angle = jnp.arctan2(real_dy, real_dx)
    crosses_now = shot_crosses_sun_xy(
        src_x, src_y, planets.radius[:, None], angle, tgt_x, tgt_y
    )

    anchor_speeds = tuple(float(s) for s in env_cfg.intercept_anchors)
    intercept_delta_x_per_anchor: list[jax.Array] = []
    intercept_delta_y_per_anchor: list[jax.Array] = []
    intercept_distance_per_anchor: list[jax.Array] = []
    intercept_turns_per_anchor: list[jax.Array] = []
    sun_cross_at_intercept_per_anchor: list[jax.Array] = []
    tgt_ships_per_anchor: list[jax.Array] = []

    for anchor_speed in anchor_speeds:
        speed = jnp.asarray(anchor_speed, dtype=jnp.float32)
        tau = jnp.maximum(snapshot_distance_raw / jnp.maximum(speed, 1e-6), 0.0)
        projected_ships = tgt_ships + tgt_production * tau
        normalized_projected_ships = (
            jnp.minimum(jnp.maximum(projected_ships, 0.0), scale) / scale
        )
        tgt_ships_per_anchor.append(normalized_projected_ships)
        step_index = game.step.astype(jnp.float32) + tau
        tgt_future_x, tgt_future_y = orbital_position_at_step_jax(
            start_angle_per_edge,
            target_orbit_radius,
            game.angular_velocity,
            step_index,
            rotates_per_edge,
            tgt_x,
            tgt_y,
        )
        tgt_future_rx, tgt_future_ry = rotate_to_learner_frame(
            tgt_future_x, tgt_future_y, theta_ref_value
        )
        intercept_delta_x = (tgt_future_rx - src_rx) / BOARD_SIZE
        intercept_delta_y = (tgt_future_ry - src_ry) / BOARD_SIZE
        intercept_distance = jnp.sqrt(
            intercept_delta_x * intercept_delta_x
            + intercept_delta_y * intercept_delta_y
        )
        raw_future_dx = tgt_future_x - src_x
        raw_future_dy = tgt_future_y - src_y
        raw_future_distance = jnp.sqrt(
            raw_future_dx * raw_future_dx + raw_future_dy * raw_future_dy
        )
        intercept_turns = jnp.clip(
            raw_future_distance / jnp.maximum(speed, 1e-6) / MAX_STEPS,
            0.0,
            1.0,
        )
        future_angle = jnp.arctan2(raw_future_dy, raw_future_dx)
        sun_cross_at_intercept = shot_crosses_sun_xy(
            src_x,
            src_y,
            planets.radius[:, None],
            future_angle,
            tgt_future_x,
            tgt_future_y,
        )
        intercept_delta_x_per_anchor.append(intercept_delta_x)
        intercept_delta_y_per_anchor.append(intercept_delta_y)
        intercept_distance_per_anchor.append(intercept_distance)
        intercept_turns_per_anchor.append(intercept_turns)
        sun_cross_at_intercept_per_anchor.append(
            sun_cross_at_intercept.astype(jnp.float32)
        )

    owner_slot = target_owner_one_hot(tgt_owner, player, env_cfg)
    incoming_friendly, incoming_enemy = incoming_fleet_pressure(
        tgt_x, tgt_y, tgt_radius, fleets, player
    )

    edge_context = EdgeRowAssemblyContext(
        intercept_delta_x_per_anchor=jnp.stack(intercept_delta_x_per_anchor, axis=-1),
        intercept_delta_y_per_anchor=jnp.stack(intercept_delta_y_per_anchor, axis=-1),
        intercept_distance_per_anchor=jnp.stack(intercept_distance_per_anchor, axis=-1),
        intercept_turns_per_anchor=jnp.stack(intercept_turns_per_anchor, axis=-1),
        sun_cross_at_intercept_per_anchor=jnp.stack(
            sun_cross_at_intercept_per_anchor, axis=-1
        ),
        crosses_now=crosses_now.astype(jnp.float32),
        tgt_ships_per_anchor=jnp.stack(tgt_ships_per_anchor, axis=-1),
        owner_slot=owner_slot,
        incoming_friendly=incoming_friendly,
        incoming_enemy=incoming_enemy,
        ordered_valid=ordered_valid,
        tgt_active=tgt_active,
        scale=scale,
    )
    edge_rows = assemble_edge_rows(
        edge_context, catalog=edge_feature_catalog_for(env_cfg)
    )

    owned_source = planets.active & (planets.owner == player)
    edge_mask = ordered_valid & (~crosses_now) & owned_source[:, None]

    tgt_ids = jnp.where(ordered_valid, jnp.take(planets.id, order, axis=0), -1)
    return edge_rows, edge_mask, tgt_ids.astype(jnp.int32)


def _global_frame(game, env_cfg: TaskConfig, scale, history: FeatureHistory | None):
    previous_global, previous_present = _previous_global_features(history, env_cfg)
    context = build_global_context(
        game, env_cfg, scale, previous_global, previous_present
    )
    return assemble_global_frame(context)


def _previous_global_features(history: FeatureHistory | None, env_cfg: TaskConfig):
    if history is None or feature_history_steps(env_cfg) <= 1:
        return (
            jnp.zeros((GLOBAL_FEATURE_CATALOG.base_dim,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    ordered = _ordered_global_history(history, env_cfg)
    previous = ordered.global_features[-1]
    present = (jnp.abs(previous).sum() > 0.0).astype(jnp.float32)
    return previous, present


def _ordered_global_history(
    history: FeatureHistory, env_cfg: TaskConfig
) -> FeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return history
    start = (history.cursor - history.length) % steps
    indices = (start + jnp.arange(steps, dtype=jnp.int32)) % steps
    valid = jnp.arange(steps, dtype=jnp.int32) >= (steps - history.length)
    return FeatureHistory(
        global_features=history.global_features[indices] * valid[:, None],
        planet_ships=history.planet_ships,
        cursor=history.cursor,
        length=history.length,
    )


def _stack_global_history(
    current: jax.Array, history: FeatureHistory | None, env_cfg: TaskConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    ordered = _ordered_global_history(base, env_cfg)
    return jnp.concatenate(
        [ordered.global_features, current[None, ...]], axis=0
    ).reshape(-1)
