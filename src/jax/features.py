"""JAX feature encoder: planet tensor + top-K edges + global vector."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config.schema import TaskConfig
from src.features.catalog._types import EdgeRowAssemblyContext
from src.features.catalog.edge import EDGE_FEATURE_CATALOG, assemble_edge_rows
from src.features.catalog.global_ import (
    GLOBAL_FEATURE_CATALOG,
    assemble_global_frame,
    build_global_context,
)
from src.features.catalog.planet import assemble_planet_features, build_planet_context
from src.features.registry import feature_history_steps
from src.game.constants import BOARD_SIZE, MAX_FLEET_SPEED, MAX_PLANETS, MAX_STEPS
from src.jax.feature_primitives import (
    incoming_fleet_pressure,
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
    edge_features, edge_mask, edge_tgt_ids = _edge_features(
        planets, game.fleets, player, env_cfg, scale, theta
    )
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


def _edge_features(
    planets, fleets, player, env_cfg: TaskConfig, scale, theta_ref_value
):
    p = MAX_PLANETS
    k = max(0, int(env_cfg.candidate_count) - 1)
    edge_dim = EDGE_FEATURE_CATALOG.base_dim
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

    src_rx, src_ry = rotate_to_learner_frame(src_x, src_y, theta_ref_value)
    tgt_rx, tgt_ry = rotate_to_learner_frame(tgt_x, tgt_y, theta_ref_value)
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
    incoming_friendly, incoming_enemy = incoming_fleet_pressure(
        tgt_x, tgt_y, tgt_radius, fleets, player
    )
    turns = distance * BOARD_SIZE / jnp.maximum(MAX_FLEET_SPEED, 1e-6) / MAX_STEPS

    edge_context = EdgeRowAssemblyContext(
        delta_x=delta_x,
        delta_y=delta_y,
        distance=distance,
        crosses=crosses,
        tgt_ships=tgt_ships,
        owner_slot=owner_slot,
        turns=turns,
        incoming_friendly=incoming_friendly,
        incoming_enemy=incoming_enemy,
        ordered_valid=ordered_valid,
        tgt_active=tgt_active,
        scale=scale,
    )
    edge_rows = assemble_edge_rows(edge_context)

    owned_source = planets.active & (planets.owner == player)
    edge_mask = ordered_valid & (~crosses) & owned_source[:, None]

    tgt_ids = jnp.where(ordered_valid, jnp.take(planets.id, order, axis=0), -1)
    return edge_rows, edge_mask, tgt_ids.astype(jnp.int32)


def _global_frame(game, env_cfg: TaskConfig, scale, history: FeatureHistory | None):
    previous_global, previous_present = _previous_global_features(history, env_cfg)
    context = build_global_context(
        game, env_cfg, scale, previous_global, previous_present
    )
    return assemble_global_frame(context)


def _previous_global_features(history: FeatureHistory | None, env_cfg: TaskConfig):
    if history is None or feature_history_steps(env_cfg) <= 1 or history.length <= 0:
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
