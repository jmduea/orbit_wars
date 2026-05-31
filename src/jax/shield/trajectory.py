from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

import jax
from src.game.constants import (
    BOARD_CENTER,
    BOARD_SIZE,
    MAX_FLEET_SPEED,
    MAX_PLANETS,
    MAX_STEPS,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from src.game.shield_config import (
    BOUNDS_REASON,
    HORIZON_REASON,
    SAFE_REASON,
    SUN_REASON,
    UNINTENDED_HIT_REASON,
    trajectory_shield_epsilon,
    trajectory_shield_hit_mode,
    trajectory_shield_mode,
)
from src.jax.action_codec import (
    JaxPolicyOutput,
    ensure_policy_sequence,
)
from src.jax.feature_primitives import orbital_position_at_step_jax

_REASON_TO_CODE = {
    SAFE_REASON: 0,
    SUN_REASON: 1,
    BOUNDS_REASON: 2,
    UNINTENDED_HIT_REASON: 3,
    HORIZON_REASON: 4,
}
_CODE_TO_REASON = {value: key for key, value in _REASON_TO_CODE.items()}


class ShieldDiagnostics(NamedTuple):
    blocked_count: jax.Array
    blocked_sun_count: jax.Array
    blocked_bounds_count: jax.Array
    blocked_unintended_hit_count: jax.Array
    blocked_horizon_count: jax.Array
    fallback_noop_count: jax.Array
    legal_non_noop_count: jax.Array
    original_non_noop_count: jax.Array
    legal_non_noop_rate: jax.Array


class ShieldedBatchResult(NamedTuple):
    batch: Any
    ship_bucket_mask: jax.Array
    diagnostics: ShieldDiagnostics


def ship_count_for_bucket_jax(
    available_ships: jax.Array, bucket: jax.Array, bucket_count: int
) -> jax.Array:
    available = jnp.maximum(available_ships, 0.0)
    fraction = jnp.where(
        bucket <= 0,
        0.0,
        bucket.astype(jnp.float32)
        / jnp.asarray(float(max(bucket_count - 1, 1)), dtype=jnp.float32),
    )
    ships = jnp.ceil(available * fraction)
    ships = jnp.minimum(available, jnp.maximum(1.0, ships))
    return jnp.where((available <= 0.0) | (fraction <= 0.0), 0.0, ships)


def ship_count_for_fraction_jax(
    available_ships: jax.Array, fraction: jax.Array
) -> jax.Array:
    """Map a continuous fraction in ``(0, 1]`` to a discrete launch count."""

    available = jnp.maximum(available_ships, 0.0)
    frac = jnp.clip(fraction.astype(jnp.float32), 1e-6, 1.0)
    ships = jnp.ceil(available * frac)
    ships = jnp.minimum(available, jnp.maximum(1.0, ships))
    return jnp.where(available <= 0.0, 0.0, ships)


def validate_continuous_ship_launch_jax(
    game,
    batch,
    env_cfg: Any,
    planet_ships: jax.Array,
    src_row: jax.Array,
    slot: jax.Array,
    ship_count: jax.Array,
) -> jax.Array:
    """Return True when a continuous ship count passes trajectory shield checks."""

    original_mask = batch.edge_mask[src_row, slot]
    target_id = batch.edge_tgt_ids[src_row, slot]
    source_id = batch.edge_src_ids[src_row]
    angle = _launch_angle_for_edge(game, batch.edge_tgt_ids, src_row, slot)
    source_ships = planet_ships[src_row]
    reason_code = _trajectory_reason_code_jax(
        game,
        source_id,
        target_id,
        angle,
        ship_count,
        game.player,
        env_cfg,
    )
    legal = (reason_code == _REASON_TO_CODE[SAFE_REASON]) & (ship_count <= source_ships)
    return original_mask & (target_id >= 0) & legal & (ship_count > 0.0)


def _planet_positions_at_step_jax(
    game, step_index: jax.Array
) -> tuple[jax.Array, jax.Array]:
    init_dx = game.initial_planets.x - BOARD_CENTER[0]
    init_dy = game.initial_planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates = (
        orbit_radius + game.planets.radius < ROTATION_RADIUS_LIMIT
    ) & game.planets.active
    start_angle = jnp.arctan2(init_dy, init_dx)
    return orbital_position_at_step_jax(
        start_angle,
        orbit_radius,
        game.angular_velocity,
        step_index,
        rotates,
        game.initial_planets.x,
        game.initial_planets.y,
    )


def _moving_circle_hit_time_jax(
    old_fx: jax.Array,
    old_fy: jax.Array,
    new_fx: jax.Array,
    new_fy: jax.Array,
    old_px: jax.Array,
    old_py: jax.Array,
    new_px: jax.Array,
    new_py: jax.Array,
    radius: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    d0x = old_fx - old_px
    d0y = old_fy - old_py
    dvx = (new_fx - old_fx) - (new_px - old_px)
    dvy = (new_fy - old_fy) - (new_py - old_py)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    static_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    denom = jnp.maximum(2.0 * a, 1e-12)
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    hit = static_hit | ((a >= 1e-12) & (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))
    time = jnp.where(static_hit, 0.0, jnp.maximum(t1, 0.0))
    return hit, time


def _line_circle_intersection_time_jax(
    start_x: jax.Array,
    start_y: jax.Array,
    end_x: jax.Array,
    end_y: jax.Array,
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[jax.Array, jax.Array]:
    dx = end_x - start_x
    dy = end_y - start_y
    rel_x = start_x - center_x
    rel_y = start_y - center_y
    a = dx * dx + dy * dy
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    static_hit = c <= 0.0
    b = 2.0 * (rel_x * dx + rel_y * dy)
    disc = b * b - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    denom = jnp.maximum(2.0 * a, 1e-12)
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    hit = static_hit | ((a >= 1e-12) & (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))
    time = jnp.where(static_hit, 0.0, jnp.maximum(t1, 0.0))
    return hit, time


def _bounds_exit_time_jax(
    start_x: jax.Array, start_y: jax.Array, end_x: jax.Array, end_y: jax.Array
) -> tuple[jax.Array, jax.Array]:
    dx = end_x - start_x
    dy = end_y - start_y
    inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    x_upper = jnp.where(
        (dx > 0.0) & (end_x > BOARD_SIZE), (BOARD_SIZE - start_x) / dx, inf
    )
    x_lower = jnp.where((dx < 0.0) & (end_x < 0.0), (0.0 - start_x) / dx, inf)
    y_upper = jnp.where(
        (dy > 0.0) & (end_y > BOARD_SIZE), (BOARD_SIZE - start_y) / dy, inf
    )
    y_lower = jnp.where((dy < 0.0) & (end_y < 0.0), (0.0 - start_y) / dy, inf)
    time = jnp.minimum(jnp.minimum(x_upper, x_lower), jnp.minimum(y_upper, y_lower))
    hit = jnp.isfinite(time) & (time >= 0.0) & (time <= 1.0)
    return hit, time


def _acceptability_mask_jax(
    game, player: jax.Array, target_id: jax.Array, hit_mode: str
) -> jax.Array:
    if hit_mode == "non_friendly":
        return game.planets.owner != player
    return game.planets.id == target_id


def _rotating_planet_mask_jax(game) -> jax.Array:
    init_dx = game.initial_planets.x - BOARD_CENTER[0]
    init_dy = game.initial_planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    return (
        orbit_radius + game.planets.radius < ROTATION_RADIUS_LIMIT
    ) & game.planets.active


def _fleet_speed_for_ships_jax(ships: jax.Array) -> jax.Array:
    speed = (
        1.0
        + (MAX_FLEET_SPEED - 1.0)
        * (jnp.log(jnp.maximum(ships, 1.0)) / jnp.log(1000.0)) ** 1.5
    )
    return jnp.minimum(speed, MAX_FLEET_SPEED)


def _static_pair_fast_path_enabled_jax(
    game,
    source_id: jax.Array,
    target_id: jax.Array,
    angle: jax.Array,
) -> jax.Array:
    rotating = _rotating_planet_mask_jax(game)
    source_index = jnp.clip(source_id.astype(jnp.int32), 0, MAX_PLANETS - 1)
    target_index = jnp.clip(target_id.astype(jnp.int32), 0, MAX_PLANETS - 1)
    source_rotating = jnp.take(rotating, source_index)
    target_rotating = jnp.take(rotating, target_index)
    source_x = jnp.take(game.planets.x, source_index)
    source_y = jnp.take(game.planets.y, source_index)
    source_radius = jnp.take(game.planets.radius, source_index)
    target_x = jnp.take(game.planets.x, target_index)
    target_y = jnp.take(game.planets.y, target_index)
    start_x = source_x + jnp.cos(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = source_y + jnp.sin(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    segment_x = target_x - start_x
    segment_y = target_y - start_y
    segment_len_sq = jnp.maximum(segment_x * segment_x + segment_y * segment_y, 1e-6)
    center_x = jnp.asarray(BOARD_CENTER[0], dtype=jnp.float32)
    center_y = jnp.asarray(BOARD_CENTER[1], dtype=jnp.float32)
    projection = (
        (center_x - start_x) * segment_x + (center_y - start_y) * segment_y
    ) / segment_len_sq
    projection = jnp.clip(projection, 0.0, 1.0)
    closest_x = start_x + projection * segment_x
    closest_y = start_y + projection * segment_y
    distance_to_orbit_center = jnp.sqrt(
        (closest_x - center_x) * (closest_x - center_x)
        + (closest_y - center_y) * (closest_y - center_y)
    )
    init_dx = game.initial_planets.x - center_x
    init_dy = game.initial_planets.y - center_y
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    orbit_band_reaches_segment = (
        jnp.abs(distance_to_orbit_center - orbit_radius)
        <= game.planets.radius + PLANET_LAUNCH_RADIUS_OFFSET
    )
    dynamic_blocker_possible = jnp.any(
        rotating
        & (game.planets.id != source_id)
        & (game.planets.id != target_id)
        & orbit_band_reaches_segment
    )
    return (~source_rotating) & (~target_rotating) & (~dynamic_blocker_possible)


def _static_trajectory_reason_codes_jax(
    game,
    source_id: jax.Array,
    target_id: jax.Array,
    angle: jax.Array,
    ships: jax.Array,
    player: jax.Array,
    env_cfg: Any,
) -> jax.Array:
    epsilon = jnp.asarray(trajectory_shield_epsilon(env_cfg), dtype=jnp.float32)
    hit_mode = trajectory_shield_hit_mode(env_cfg)
    source_index = jnp.clip(source_id.astype(jnp.int32), 0, MAX_PLANETS - 1)
    target_index = jnp.clip(target_id.astype(jnp.int32), 0, MAX_PLANETS - 1)
    horizon = max(int(getattr(env_cfg, "trajectory_shield_horizon", MAX_STEPS)), 1)
    remaining_horizon = jnp.minimum(
        jnp.asarray(horizon, dtype=jnp.float32),
        jnp.maximum(
            jnp.asarray(MAX_STEPS, dtype=jnp.float32) - game.step.astype(jnp.float32),
            0.0,
        ),
    )
    source_x = jnp.take(game.planets.x, source_index)
    source_y = jnp.take(game.planets.y, source_index)
    source_radius = jnp.take(game.planets.radius, source_index)
    target_x = jnp.take(game.planets.x, target_index)
    target_y = jnp.take(game.planets.y, target_index)
    start_x = source_x + jnp.cos(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = source_y + jnp.sin(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    segment_dx = target_x - start_x
    segment_dy = target_y - start_y
    segment_length = jnp.sqrt(segment_dx * segment_dx + segment_dy * segment_dy)

    hit_mask, hit_time = _line_circle_intersection_time_jax(
        start_x,
        start_y,
        target_x,
        target_y,
        game.planets.x,
        game.planets.y,
        game.planets.radius,
    )
    hit_mask = hit_mask & game.planets.active & (game.planets.id != source_id)
    acceptable_mask = hit_mask & _acceptability_mask_jax(
        game, player, target_id, hit_mode
    )
    unacceptable_mask = hit_mask & (
        ~_acceptability_mask_jax(game, player, target_id, hit_mode)
    )
    inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    acceptable_time = jnp.min(jnp.where(acceptable_mask, hit_time, inf))
    unacceptable_time = jnp.min(jnp.where(unacceptable_mask, hit_time, inf))
    sun_hit, sun_time = _line_circle_intersection_time_jax(
        start_x,
        start_y,
        target_x,
        target_y,
        BOARD_CENTER[0],
        BOARD_CENTER[1],
        SUN_RADIUS,
    )
    bounds_hit, bounds_time = _bounds_exit_time_jax(
        start_x, start_y, target_x, target_y
    )

    max_distance = _fleet_speed_for_ships_jax(ships) * remaining_horizon
    acceptable_reachable = acceptable_time * segment_length <= max_distance
    unacceptable_reachable = unacceptable_time * segment_length <= max_distance
    sun_reachable = (sun_time * segment_length <= max_distance) & sun_hit
    bounds_reachable = (bounds_time * segment_length <= max_distance) & bounds_hit
    acceptable_eval = jnp.where(acceptable_reachable, acceptable_time, inf)
    unacceptable_eval = jnp.where(unacceptable_reachable, unacceptable_time, inf)
    sun_eval = jnp.where(sun_reachable, sun_time, inf)
    bounds_eval = jnp.where(bounds_reachable, bounds_time, inf)
    block_time = jnp.minimum(jnp.minimum(sun_eval, bounds_eval), unacceptable_eval)
    safe_hit = jnp.isfinite(acceptable_eval) & (acceptable_eval + epsilon < block_time)
    sun_blocks = (
        jnp.isfinite(sun_eval) & (~safe_hit) & (sun_eval <= acceptable_eval + epsilon)
    )
    bounds_blocks = (
        jnp.isfinite(bounds_eval)
        & (~safe_hit)
        & (~sun_blocks)
        & (bounds_eval <= acceptable_eval + epsilon)
    )
    unintended_blocks = (
        jnp.isfinite(unacceptable_eval)
        & (~safe_hit)
        & (~sun_blocks)
        & (~bounds_blocks)
        & (unacceptable_eval <= acceptable_eval + epsilon)
    )
    reason_code = jnp.where(
        sun_blocks,
        _REASON_TO_CODE[SUN_REASON],
        jnp.where(
            bounds_blocks,
            _REASON_TO_CODE[BOUNDS_REASON],
            jnp.where(
                unintended_blocks,
                _REASON_TO_CODE[UNINTENDED_HIT_REASON],
                jnp.where(
                    safe_hit,
                    _REASON_TO_CODE[SAFE_REASON],
                    _REASON_TO_CODE[HORIZON_REASON],
                ),
            ),
        ),
    )
    return jnp.where(ships <= 0.0, _REASON_TO_CODE[SAFE_REASON], reason_code)


def _trajectory_reason_code_jax(
    game,
    source_id: jax.Array,
    target_id: jax.Array,
    angle: jax.Array,
    ships: jax.Array,
    player: jax.Array,
    env_cfg: Any,
) -> jax.Array:
    epsilon = jnp.asarray(trajectory_shield_epsilon(env_cfg), dtype=jnp.float32)
    hit_mode = trajectory_shield_hit_mode(env_cfg)
    source_index = jnp.clip(source_id.astype(jnp.int32), 0, MAX_PLANETS - 1)
    horizon = max(int(getattr(env_cfg, "trajectory_shield_horizon", MAX_STEPS)), 1)
    remaining_horizon = jnp.minimum(
        jnp.asarray(horizon, dtype=jnp.int32),
        jnp.maximum(
            jnp.asarray(MAX_STEPS, dtype=jnp.int32) - game.step.astype(jnp.int32), 0
        ),
    )
    source_x = jnp.take(game.planets.x, source_index)
    source_y = jnp.take(game.planets.y, source_index)
    source_radius = jnp.take(game.planets.radius, source_index)
    speed = _fleet_speed_for_ships_jax(ships)

    def scan_step(carry, offset):
        old_x, old_y, done, reason_code = carry
        current_step = game.step.astype(jnp.int32) + offset
        active_step = (~done) & (offset < remaining_horizon)
        new_x = old_x + jnp.cos(angle) * speed
        new_y = old_y + jnp.sin(angle) * speed
        old_px, old_py = _planet_positions_at_step_jax(game, current_step)
        new_px, new_py = _planet_positions_at_step_jax(game, current_step + 1)
        hit_mask, hit_time = _moving_circle_hit_time_jax(
            old_x,
            old_y,
            new_x,
            new_y,
            old_px,
            old_py,
            new_px,
            new_py,
            game.planets.radius,
        )
        hit_mask = hit_mask & game.planets.active
        acceptable_mask = hit_mask & _acceptability_mask_jax(
            game, player, target_id, hit_mode
        )
        unacceptable_mask = hit_mask & (
            ~_acceptability_mask_jax(game, player, target_id, hit_mode)
        )
        inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
        acceptable_time = jnp.min(jnp.where(acceptable_mask, hit_time, inf))
        unacceptable_time = jnp.min(jnp.where(unacceptable_mask, hit_time, inf))
        sun_hit, sun_time = _line_circle_intersection_time_jax(
            old_x, old_y, new_x, new_y, BOARD_CENTER[0], BOARD_CENTER[1], SUN_RADIUS
        )
        bounds_hit, bounds_time = _bounds_exit_time_jax(old_x, old_y, new_x, new_y)
        sun_eval = jnp.where(sun_hit, sun_time, inf)
        bounds_eval = jnp.where(bounds_hit, bounds_time, inf)
        block_time = jnp.minimum(jnp.minimum(sun_eval, bounds_eval), unacceptable_time)
        safe_hit = jnp.isfinite(acceptable_time) & (
            acceptable_time + epsilon < block_time
        )
        sun_blocks = (
            jnp.isfinite(sun_eval)
            & (~safe_hit)
            & (sun_eval <= acceptable_time + epsilon)
        )
        bounds_blocks = (
            jnp.isfinite(bounds_eval)
            & (~safe_hit)
            & (~sun_blocks)
            & (bounds_eval <= acceptable_time + epsilon)
        )
        unintended_blocks = (
            jnp.isfinite(unacceptable_time)
            & (~safe_hit)
            & (~sun_blocks)
            & (~bounds_blocks)
            & (unacceptable_time <= acceptable_time + epsilon)
        )
        resolved = active_step & (
            safe_hit | sun_blocks | bounds_blocks | unintended_blocks
        )
        next_reason = jnp.where(
            active_step & sun_blocks,
            _REASON_TO_CODE[SUN_REASON],
            jnp.where(
                active_step & bounds_blocks,
                _REASON_TO_CODE[BOUNDS_REASON],
                jnp.where(
                    active_step & unintended_blocks,
                    _REASON_TO_CODE[UNINTENDED_HIT_REASON],
                    jnp.where(
                        active_step & safe_hit,
                        _REASON_TO_CODE[SAFE_REASON],
                        reason_code,
                    ),
                ),
            ),
        )
        next_done = done | resolved
        return (
            jnp.where(active_step, new_x, old_x),
            jnp.where(active_step, new_y, old_y),
            next_done,
            next_reason,
        ), None

    start_x = source_x + jnp.cos(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = source_y + jnp.sin(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    initial = (
        start_x,
        start_y,
        ships <= 0.0,
        jnp.where(
            ships <= 0.0, _REASON_TO_CODE[SAFE_REASON], _REASON_TO_CODE[HORIZON_REASON]
        ),
    )
    final, _ = jax.lax.scan(scan_step, initial, jnp.arange(horizon, dtype=jnp.int32))
    return final[3]


def trajectory_shield_reason_for_launch_jax(
    game,
    source_id: jax.Array,
    target_id: jax.Array,
    angle: jax.Array,
    ships: jax.Array,
    player: jax.Array,
    env_cfg: Any,
) -> jax.Array:
    return _trajectory_reason_code_jax(
        game, source_id, target_id, angle, ships, player, env_cfg
    )


def trajectory_shield_reason_name(code: int | jax.Array) -> str:
    return _CODE_TO_REASON[int(code)]


def default_edge_action_bucket_mask(
    edge_action_mask: jax.Array, ship_bucket_count: int
) -> jax.Array:
    """Default bucket legality for flat edge actions including trailing NO_OP."""

    if edge_action_mask.ndim == 1:
        edge_action_mask = edge_action_mask[None, :]
    bucket_ids = jnp.arange(max(int(ship_bucket_count), 1), dtype=jnp.int32)
    bucket_is_noop = bucket_ids == 0
    edge_count = edge_action_mask.shape[-1]
    edge_ids = jnp.arange(edge_count, dtype=jnp.int32)
    noop_edge = edge_ids == jnp.maximum(edge_count - 1, 0)
    per_edge_bucket = jnp.where(
        noop_edge[:, None],
        bucket_is_noop[None, :],
        ~bucket_is_noop[None, :],
    )
    return edge_action_mask.astype(bool)[..., None] & per_edge_bucket


def mask_policy_output_for_shield_v2(
    output: JaxPolicyOutput,
    edge_action_mask: jax.Array,
    ship_bucket_count: int,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxPolicyOutput:
    target_logits = ensure_policy_sequence(output.target_logits)
    ship_logits = output.ship_logits
    if ship_logits.ndim == 3:
        ship_logits = ship_logits[:, None, :, :]
    base_mask = edge_action_mask.astype(bool)
    if base_mask.ndim == 1:
        base_mask = base_mask[None, :]
    if ship_bucket_mask is None:
        ship_bucket_mask = default_edge_action_bucket_mask(base_mask, ship_bucket_count)
    elif ship_bucket_mask.ndim == 4 and ship_bucket_mask.shape[0] != base_mask.shape[0]:
        ship_bucket_mask = ship_bucket_mask.reshape(
            -1, ship_bucket_mask.shape[-2], ship_bucket_mask.shape[-1]
        )
    ship_bucket_mask = ship_bucket_mask.astype(bool)
    sequence_k = target_logits.shape[1]
    if ship_bucket_mask.ndim == 3:
        sequence_ship_bucket_mask = jnp.broadcast_to(
            ship_bucket_mask[:, None, :, :],
            (
                base_mask.shape[0],
                sequence_k,
                ship_logits.shape[-2],
                ship_logits.shape[-1],
            ),
        )
    else:
        sequence_ship_bucket_mask = ship_bucket_mask
    target_mask = base_mask[:, None, :] & sequence_ship_bucket_mask.any(axis=-1)
    illegal_logit = jnp.finfo(jnp.float32).min
    masked_target_logits = jnp.where(target_mask, target_logits, illegal_logit)
    masked_ship_logits = jnp.where(
        sequence_ship_bucket_mask, ship_logits, illegal_logit
    )
    return output._replace(
        target_logits=masked_target_logits, ship_logits=masked_ship_logits
    )


def _launch_angle_for_edge(game, edge_tgt_ids, src_row, slot):
    src_x = game.planets.x[src_row]
    src_y = game.planets.y[src_row]
    tgt_id = edge_tgt_ids[src_row, slot]
    match = game.planets.id == tgt_id
    tgt_x = jnp.sum(jnp.where(match, game.planets.x, 0.0))
    tgt_y = jnp.sum(jnp.where(match, game.planets.y, 0.0))
    return jnp.arctan2(tgt_y - src_y, tgt_x - src_x)


def evaluate_edge_pair(
    game,
    batch,
    env_cfg: Any,
    planet_ships: jax.Array,
    src_row: jax.Array,
    slot: jax.Array,
    *,
    bucket_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate trajectory legality for one ``(source row, target slot)`` edge."""

    original_mask = batch.edge_mask[src_row, slot]
    target_id = batch.edge_tgt_ids[src_row, slot]
    source_id = batch.edge_src_ids[src_row]
    angle = _launch_angle_for_edge(game, batch.edge_tgt_ids, src_row, slot)
    source_ships = planet_ships[src_row]
    bucket_ids = jnp.arange(1, bucket_count, dtype=jnp.int32)
    ship_counts = ship_count_for_bucket_jax(
        jnp.broadcast_to(source_ships, bucket_ids.shape), bucket_ids, bucket_count
    )
    static_fast_path_enabled = _static_pair_fast_path_enabled_jax(
        game, source_id, target_id, angle
    )

    def evaluate_static(_):
        return _static_trajectory_reason_codes_jax(
            game,
            source_id,
            target_id,
            angle,
            ship_counts,
            game.player,
            env_cfg,
        )

    def evaluate_dynamic(_):
        return jax.vmap(
            lambda ships: _trajectory_reason_code_jax(
                game,
                source_id,
                target_id,
                angle,
                ships,
                game.player,
                env_cfg,
            )
        )(ship_counts)

    reason_codes = jax.lax.cond(
        static_fast_path_enabled, evaluate_static, evaluate_dynamic, operand=None
    )
    bucket_legal = reason_codes == _REASON_TO_CODE[SAFE_REASON]
    bucket_legal = bucket_legal & (ship_counts <= source_ships)
    legal = jnp.any(bucket_legal)
    legal = original_mask & (target_id >= 0) & legal
    return legal, bucket_legal


def _zero_shield_diagnostics_like(value: jax.Array) -> ShieldDiagnostics:
    zero = jnp.zeros_like(value, dtype=jnp.float32)
    return ShieldDiagnostics(
        blocked_count=zero,
        blocked_sun_count=zero,
        blocked_bounds_count=zero,
        blocked_unintended_hit_count=zero,
        blocked_horizon_count=zero,
        fallback_noop_count=zero,
        legal_non_noop_count=zero,
        original_non_noop_count=zero,
        legal_non_noop_rate=zero,
    )


def _unshielded_factorized_topk_result(
    game,
    batch,
    env_cfg: Any,
    *,
    remaining_planet_ships: jax.Array | None = None,
) -> ShieldedBatchResult:
    """
    Return ordinary edge/bucket legality without trajectory simulation.

    This preserves the source/target top-k mask from feature encoding and only expands it across non-zero ship buckets.
    """
    from src.features.registry import edge_k

    k = edge_k(env_cfg)
    bucket_count = max(int(getattr(env_cfg, "ship_bucket_count", 1)), 1)
    planet_ships = (
        game.planets.ships if remaining_planet_ships is None else remaining_planet_ships
    )

    bucket_ids = jnp.arange(bucket_count, dtype=jnp.int32)
    real_bucket = bucket_ids > 0

    ship_counts = ship_count_for_bucket_jax(
        planet_ships[:, None],
        bucket_ids[None, :],
        bucket_count,
    )
    bucket_has_ships = (ship_counts > 0.0) & (ship_counts <= planet_ships[:, None])

    real_bucket_mask = (
        batch.edge_mask[..., None]
        & real_bucket[None, None, :]
        & bucket_has_ships[:, None, :]
    )

    shielded_edge_mask = real_bucket_mask[..., 1:].any(axis=-1)
    original_legal_total = batch.edge_mask.astype(jnp.float32).sum()
    shielded_legal_total = shielded_edge_mask.astype(jnp.float32).sum()

    diagnostics = ShieldDiagnostics(
        blocked_count=jnp.asarray(0.0, dtype=jnp.float32),
        blocked_sun_count=jnp.asarray(0.0, dtype=jnp.float32),
        blocked_bounds_count=jnp.asarray(0.0, dtype=jnp.float32),
        blocked_unintended_hit_count=jnp.asarray(0.0, dtype=jnp.float32),
        blocked_horizon_count=jnp.asarray(0.0, dtype=jnp.float32),
        fallback_noop_count=(
            (batch.edge_mask.any()) & (~shielded_edge_mask.any())
        ).astype(jnp.float32),
        legal_non_noop_count=shielded_legal_total,
        original_non_noop_count=original_legal_total,
        legal_non_noop_rate=jnp.where(
            original_legal_total > 0.0,
            shielded_legal_total / original_legal_total,
            0.0,
        ),
    )

    return ShieldedBatchResult(
        batch=batch._replace(edge_mask=shielded_edge_mask),
        ship_bucket_mask=real_bucket_mask,
        diagnostics=diagnostics,
    )


def _edge_scalar_feature(batch, feature_name: str) -> jax.Array:
    """Read a scalar edge feature from TurnBatch.edge_features.

    The feature schema is static Python metadata, so the slice is resolved before
    JAX tracing. This avoids adding the feature registry to the JAX data path.
    """

    from src.features.registry import EDGE_FEATURE_SCHEMA

    feature_slice = EDGE_FEATURE_SCHEMA.base_slice(feature_name)
    value = batch.edge_features[..., feature_slice]
    return value.reshape(value.shape[:-1])


def apply_cheap_trajectory_shield_factorized_topk(
    game,
    batch,
    env_cfg: Any,
    remaining_planet_ships: jax.Array | None = None,
) -> ShieldedBatchResult:
    """Cheap shield for factorized top-K decoding.

    This intentionally avoids horizon scans. It uses edge features already
    computed by encode_turn:

      - batch.edge_mask for source ownership, active target, not-current-sun-cross
      - sun_cross_at_intercept_s1 for slow/small launches
      - sun_cross_at_intercept_s6 for fast/large launches
      - per-bucket ship availability

    It is not a perfect physics validator. It is a low-cost mask for PPO
    collection and training.
    """

    bucket_count = max(int(getattr(env_cfg, "ship_bucket_count", 1)), 1)
    planet_ships = (
        game.planets.ships if remaining_planet_ships is None else remaining_planet_ships
    )

    bucket_ids = jnp.arange(bucket_count, dtype=jnp.int32)
    real_bucket = bucket_ids > 0

    # Existing edge schema currently exposes s1 and s6 anchors.
    sun_s1 = _edge_scalar_feature(batch, "sun_cross_at_intercept_s1") > 0.5
    sun_s6 = _edge_scalar_feature(batch, "sun_cross_at_intercept_s6") > 0.5

    # Map lower buckets to slow-anchor risk, upper buckets to fast-anchor risk.
    # Bucket 0 is reserved for no-launch/noop behavior and is never a real launch.
    midpoint = max((bucket_count - 1) // 2, 1)
    use_slow_anchor = bucket_ids <= midpoint
    bucket_sun_blocked = jnp.where(
        use_slow_anchor[None, None, :],
        sun_s1[..., None],
        sun_s6[..., None],
    )

    ship_counts = ship_count_for_bucket_jax(
        planet_ships[:, None],
        bucket_ids[None, :],
        bucket_count,
    )
    bucket_has_ships = (ship_counts > 0.0) & (ship_counts <= planet_ships[:, None])

    bucket_legal = (
        batch.edge_mask[..., None]
        & real_bucket[None, None, :]
        & bucket_has_ships[:, None, :]
        & (~bucket_sun_blocked)
    )

    shielded_edge_mask = bucket_legal[..., 1:].any(axis=-1)

    original_real_mask = batch.edge_mask
    original_legal_total = original_real_mask.astype(jnp.float32).sum()
    shielded_legal_total = shielded_edge_mask.astype(jnp.float32).sum()
    blocked_slots = original_real_mask & (~shielded_edge_mask)

    sun_blocked_slots = original_real_mask & (
        ~(~bucket_sun_blocked[..., 1:]).any(axis=-1)
    )

    diagnostics = ShieldDiagnostics(
        blocked_count=blocked_slots.astype(jnp.float32).sum(),
        blocked_sun_count=(blocked_slots & sun_blocked_slots).astype(jnp.float32).sum(),
        blocked_bounds_count=jnp.asarray(0.0, dtype=jnp.float32),
        blocked_unintended_hit_count=jnp.asarray(0.0, dtype=jnp.float32),
        blocked_horizon_count=(blocked_slots & (~sun_blocked_slots))
        .astype(jnp.float32)
        .sum(),
        fallback_noop_count=(
            (original_real_mask.any()) & (~shielded_edge_mask.any())
        ).astype(jnp.float32),
        legal_non_noop_count=shielded_legal_total,
        original_non_noop_count=original_legal_total,
        legal_non_noop_rate=jnp.where(
            original_legal_total > 0.0,
            shielded_legal_total / original_legal_total,
            0.0,
        ),
    )

    return ShieldedBatchResult(
        batch=batch._replace(edge_mask=shielded_edge_mask),
        ship_bucket_mask=bucket_legal,
        diagnostics=diagnostics,
    )


def apply_configured_trajectory_shield_factorized_topk(
    game,
    batch,
    env_cfg: Any,
    remaining_planet_ships: jax.Array | None = None,
) -> ShieldedBatchResult:
    """Dispatch factorized top-K shielding based on task.trajectory_shield_mode."""

    mode = trajectory_shield_mode(env_cfg)

    if mode == "off":
        return _unshielded_factorized_topk_result(
            game,
            batch,
            env_cfg,
            remaining_planet_ships=remaining_planet_ships,
        )

    if mode in {"cheap", "tiered"}:
        return apply_cheap_trajectory_shield_factorized_topk(
            game,
            batch,
            env_cfg,
            remaining_planet_ships=remaining_planet_ships,
        )

    return apply_trajectory_shield_factorized_topk(
        game,
        batch,
        env_cfg,
        remaining_planet_ships=remaining_planet_ships,
    )


def selected_factored_launch_is_exact_safe_jax(
    game,
    batch,
    env_cfg: Any,
    source_row: jax.Array,
    target_slot: jax.Array,
    ships: jax.Array,
    stop_flag: jax.Array,
    step_active: jax.Array,
) -> jax.Array:
    """Exact-check one selected factorized launch.

    This is intended for tiered mode after cheap sampling. It checks only the
    selected action, not the full source-target-bucket lattice.
    """

    source_id = batch.edge_src_ids[source_row]
    target_id = batch.edge_tgt_ids[source_row, target_slot]
    angle = _launch_angle_for_edge(game, batch.edge_tgt_ids, source_row, target_slot)

    should_check = (
        step_active.astype(bool)
        & (~stop_flag.astype(bool))
        & (ships > 0.0)
        & (target_id >= 0)
    )

    reason_code = _trajectory_reason_code_jax(
        game,
        source_id,
        target_id,
        angle,
        ships,
        game.player,
        env_cfg,
    )

    return (~should_check) | (reason_code == _REASON_TO_CODE[SAFE_REASON])


def apply_trajectory_shield_factorized_topk(
    game,
    batch,
    env_cfg: Any,
    remaining_planet_ships: jax.Array | None = None,
) -> ShieldedBatchResult:
    """Shield factorized top-K actions with a ``(P, K, buckets)`` bucket mask."""

    from src.features.registry import edge_k

    k = edge_k(env_cfg)
    bucket_count = max(int(getattr(env_cfg, "ship_bucket_count", 1)), 1)
    planet_ships = (
        game.planets.ships if remaining_planet_ships is None else remaining_planet_ships
    )
    original_real_mask = batch.edge_mask
    original_legal_total = original_real_mask.astype(jnp.float32).sum()

    if trajectory_shield_mode(env_cfg) == "off" or k == 0:
        unshielded_bucket_mask = jnp.zeros(
            (MAX_PLANETS, k, bucket_count),
            dtype=bool,
        )
        unshielded_bucket_mask = unshielded_bucket_mask.at[..., 0].set(True)
        if bucket_count > 1:
            legal_edges = batch.edge_mask.reshape(MAX_PLANETS, k)
            unshielded_bucket_mask = unshielded_bucket_mask.at[..., 1:].set(
                legal_edges[..., None]
            )
        diagnostics = ShieldDiagnostics(
            blocked_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_sun_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_bounds_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_unintended_hit_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_horizon_count=jnp.asarray(0.0, dtype=jnp.float32),
            fallback_noop_count=jnp.asarray(0.0, dtype=jnp.float32),
            legal_non_noop_count=original_legal_total,
            original_non_noop_count=original_legal_total,
            legal_non_noop_rate=jnp.where(original_legal_total > 0.0, 1.0, 0.0),
        )
        return ShieldedBatchResult(
            batch=batch,
            ship_bucket_mask=unshielded_bucket_mask,
            diagnostics=diagnostics,
        )

    slots = jnp.arange(k, dtype=jnp.int32)

    def evaluate_row(src_row: jax.Array) -> tuple[jax.Array, jax.Array]:
        legal, bucket_legal = jax.vmap(
            lambda slot: evaluate_edge_pair(
                game,
                batch,
                env_cfg,
                planet_ships,
                src_row,
                slot,
                bucket_count=bucket_count,
            )
        )(slots)
        return legal, bucket_legal

    src_rows = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    shielded_edge_mask, legal_bucket_mask = jax.vmap(evaluate_row)(src_rows)
    ship_bucket_mask = jnp.zeros((MAX_PLANETS, k, bucket_count), dtype=bool)
    if bucket_count > 1:
        ship_bucket_mask = ship_bucket_mask.at[..., 1:].set(legal_bucket_mask)

    blocked_slots = original_real_mask & (~shielded_edge_mask)
    shielded_legal_total = shielded_edge_mask.astype(jnp.float32).sum()
    legal_non_noop_rate = jnp.where(
        original_legal_total > 0.0,
        shielded_legal_total / original_legal_total,
        0.0,
    )
    diagnostics = ShieldDiagnostics(
        blocked_count=blocked_slots.astype(jnp.float32).sum(),
        blocked_sun_count=blocked_slots.astype(jnp.float32).sum() * 0.0,
        blocked_bounds_count=blocked_slots.astype(jnp.float32).sum() * 0.0,
        blocked_unintended_hit_count=blocked_slots.astype(jnp.float32).sum() * 0.0,
        blocked_horizon_count=blocked_slots.astype(jnp.float32).sum() * 0.0,
        fallback_noop_count=(
            (original_real_mask.any()) & (~shielded_edge_mask.any())
        ).astype(jnp.float32),
        legal_non_noop_count=shielded_legal_total,
        original_non_noop_count=original_legal_total,
        legal_non_noop_rate=legal_non_noop_rate,
    )
    return ShieldedBatchResult(
        batch=batch._replace(edge_mask=shielded_edge_mask),
        ship_bucket_mask=ship_bucket_mask,
        diagnostics=diagnostics,
    )


def factorized_source_mask_from_shield(
    shielded_edge_mask: jax.Array,
    ship_bucket_mask: jax.Array,
    planet_ships: jax.Array,
) -> jax.Array:
    """Owned planets with ships and at least one shielded non-noop bucket."""

    has_real_bucket = ship_bucket_mask[..., 1:].any(axis=-1)
    row_has_legal = shielded_edge_mask & has_real_bucket
    return (planet_ships > 0.0) & row_has_legal.any(axis=-1)


def apply_trajectory_shield_to_turn_batch_v2(
    game,
    batch,
    env_cfg: Any,
    remaining_planet_ships: jax.Array | None = None,
) -> ShieldedBatchResult:
    from src.features.registry import edge_k

    k = edge_k(env_cfg)
    edge_count = MAX_PLANETS * k + 1
    bucket_count = max(int(getattr(env_cfg, "ship_bucket_count", 1)), 1)
    edge_action_mask = jnp.concatenate(
        [batch.edge_mask.reshape(MAX_PLANETS * k), jnp.ones((1,), dtype=bool)], axis=0
    )
    default_bucket_mask = default_edge_action_bucket_mask(
        edge_action_mask[None, :], bucket_count
    ).squeeze(0)
    original_real_mask = batch.edge_mask.reshape(MAX_PLANETS * k)
    original_legal_total = original_real_mask.astype(jnp.float32).sum()
    planet_ships = (
        game.planets.ships if remaining_planet_ships is None else remaining_planet_ships
    )

    if trajectory_shield_mode(env_cfg) == "off" or k == 0:
        diagnostics = ShieldDiagnostics(
            blocked_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_sun_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_bounds_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_unintended_hit_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_horizon_count=jnp.asarray(0.0, dtype=jnp.float32),
            fallback_noop_count=jnp.asarray(0.0, dtype=jnp.float32),
            legal_non_noop_count=original_legal_total,
            original_non_noop_count=original_legal_total,
            legal_non_noop_rate=jnp.where(original_legal_total > 0.0, 1.0, 0.0),
        )
        return ShieldedBatchResult(
            batch=batch,
            ship_bucket_mask=default_bucket_mask,
            diagnostics=diagnostics,
        )

    def evaluate_flat_edge(flat_idx):
        src_row = flat_idx // k
        slot = flat_idx % k
        return evaluate_edge_pair(
            game,
            batch,
            env_cfg,
            planet_ships,
            src_row,
            slot,
            bucket_count=bucket_count,
        )

    shielded_real_mask, legal_bucket_mask = jax.vmap(evaluate_flat_edge)(
        jnp.arange(MAX_PLANETS * k, dtype=jnp.int32)
    )
    real_bucket_rows = jnp.concatenate(
        [
            jnp.zeros((MAX_PLANETS * k, 1), dtype=bool),
            legal_bucket_mask,
        ],
        axis=-1,
    )
    ship_bucket_mask = jnp.concatenate(
        [default_bucket_mask[: MAX_PLANETS * k], default_bucket_mask[-1:]], axis=0
    )
    ship_bucket_mask = ship_bucket_mask.at[: MAX_PLANETS * k].set(real_bucket_rows)

    blocked_slots = original_real_mask & (~shielded_real_mask)
    shielded_legal_total = shielded_real_mask.astype(jnp.float32).sum()
    legal_non_noop_rate = jnp.where(
        original_legal_total > 0.0,
        shielded_legal_total / original_legal_total,
        0.0,
    )
    diagnostics = ShieldDiagnostics(
        blocked_count=blocked_slots.astype(jnp.float32).sum(),
        blocked_sun_count=(blocked_slots).astype(jnp.float32).sum() * 0.0,
        blocked_bounds_count=(blocked_slots).astype(jnp.float32).sum() * 0.0,
        blocked_unintended_hit_count=(blocked_slots).astype(jnp.float32).sum() * 0.0,
        blocked_horizon_count=(blocked_slots).astype(jnp.float32).sum() * 0.0,
        fallback_noop_count=(
            (original_real_mask.any()) & (~shielded_real_mask.any())
        ).astype(jnp.float32),
        legal_non_noop_count=shielded_legal_total,
        original_non_noop_count=original_legal_total,
        legal_non_noop_rate=legal_non_noop_rate,
    )
    shielded_edge_mask = shielded_real_mask.reshape(batch.edge_mask.shape)
    return ShieldedBatchResult(
        batch=batch._replace(edge_mask=shielded_edge_mask),
        ship_bucket_mask=ship_bucket_mask,
        diagnostics=diagnostics,
    )
