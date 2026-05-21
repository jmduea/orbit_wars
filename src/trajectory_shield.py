from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .constants import (
    BOARD_CENTER,
    BOARD_SIZE,
    MAX_FLEET_SPEED,
    MAX_PLANETS,
    MAX_STEPS,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .game_types import GameState, PlanetState
from .jax_policy import JaxPolicyOutput, action_log_prob_and_entropy, ensure_policy_sequence

SAFE_REASON = "safe"
SUN_REASON = "sun"
BOUNDS_REASON = "bounds"
UNINTENDED_HIT_REASON = "unintended_hit"
HORIZON_REASON = "horizon"

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


class ShieldedActionSample(NamedTuple):
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    entropy: jax.Array


def trajectory_shield_enabled(env_cfg: Any) -> bool:
    return bool(getattr(env_cfg, "trajectory_shield_enabled", True))


def trajectory_shield_horizon(state_step: int, env_cfg: Any) -> int:
    configured = max(int(getattr(env_cfg, "trajectory_shield_horizon", MAX_STEPS)), 1)
    remaining = max(MAX_STEPS - int(state_step), 0)
    return min(configured, remaining)


def trajectory_shield_epsilon(env_cfg: Any) -> float:
    return max(float(getattr(env_cfg, "trajectory_shield_epsilon", 0.0)), 0.0)


def trajectory_shield_hit_mode(env_cfg: Any) -> str:
    return str(getattr(env_cfg, "trajectory_shield_hit_mode", "selected_target")).strip().lower()


def ship_count_for_bucket_py(available_ships: float | int, bucket: int, bucket_count: int) -> int:
    available = max(0, int(available_ships))
    if available <= 0 or bucket <= 0:
        return 0
    fraction = float(bucket) / float(max(bucket_count - 1, 1))
    ships = int(math.ceil(available * fraction))
    return min(available, max(1, ships))


def ship_count_for_bucket_jax(
    available_ships: jax.Array, bucket: jax.Array, bucket_count: int
) -> jax.Array:
    available = jnp.maximum(available_ships, 0.0)
    fraction = jnp.where(
        bucket <= 0,
        0.0,
        bucket.astype(jnp.float32) / jnp.asarray(float(max(bucket_count - 1, 1)), dtype=jnp.float32),
    )
    ships = jnp.ceil(available * fraction)
    ships = jnp.minimum(available, jnp.maximum(1.0, ships))
    return jnp.where((available <= 0.0) | (fraction <= 0.0), 0.0, ships)


def fleet_speed_py(ships: float, ship_speed: float = MAX_FLEET_SPEED) -> float:
    safe = max(float(ships), 1.0)
    speed = 1.0 + (ship_speed - 1.0) * (math.log(safe) / math.log(1000.0)) ** 1.5
    return min(speed, ship_speed)


def _planet_position_at_step(planet: PlanetState, initial_planet: PlanetState, angular_velocity: float, step_index: int) -> tuple[float, float]:
    dx = initial_planet.x - BOARD_CENTER[0]
    dy = initial_planet.y - BOARD_CENTER[1]
    orbital_radius = math.hypot(dx, dy)
    rotates = orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT
    if not rotates:
        return initial_planet.x, initial_planet.y
    start_angle = math.atan2(dy, dx)
    angle = start_angle + angular_velocity * float(step_index)
    return (
        BOARD_CENTER[0] + orbital_radius * math.cos(angle),
        BOARD_CENTER[1] + orbital_radius * math.sin(angle),
    )


def _planet_positions_at_step_jax(game, step_index: jax.Array) -> tuple[jax.Array, jax.Array]:
    init_dx = game.initial_planets.x - BOARD_CENTER[0]
    init_dy = game.initial_planets.y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates = (orbit_radius + game.planets.radius < ROTATION_RADIUS_LIMIT) & game.planets.active
    start_angle = jnp.arctan2(init_dy, init_dx)
    angle = start_angle + game.angular_velocity * step_index.astype(jnp.float32)
    x = jnp.where(
        rotates,
        BOARD_CENTER[0] + orbit_radius * jnp.cos(angle),
        game.initial_planets.x,
    )
    y = jnp.where(
        rotates,
        BOARD_CENTER[1] + orbit_radius * jnp.sin(angle),
        game.initial_planets.y,
    )
    return x, y


def _moving_circle_hit_time_py(
    old_fx: float,
    old_fy: float,
    new_fx: float,
    new_fy: float,
    old_px: float,
    old_py: float,
    new_px: float,
    new_py: float,
    radius: float,
) -> float | None:
    d0x = old_fx - old_px
    d0y = old_fy - old_py
    dvx = (new_fx - old_fx) - (new_px - old_px)
    dvy = (new_fy - old_fy) - (new_py - old_py)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    if c <= 0.0:
        return 0.0
    if a < 1e-12:
        return None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(max(disc, 0.0))
    denom = 2.0 * a
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    if t2 < 0.0 or t1 > 1.0:
        return None
    return max(0.0, t1)


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


def _line_circle_intersection_time_py(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    center_x: float,
    center_y: float,
    radius: float,
) -> float | None:
    dx = end_x - start_x
    dy = end_y - start_y
    rel_x = start_x - center_x
    rel_y = start_y - center_y
    a = dx * dx + dy * dy
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    if c <= 0.0:
        return 0.0
    if a < 1e-12:
        return None
    b = 2.0 * (rel_x * dx + rel_y * dy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(max(disc, 0.0))
    denom = 2.0 * a
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    if t2 < 0.0 or t1 > 1.0:
        return None
    return max(0.0, t1)


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


def _bounds_exit_time_py(start_x: float, start_y: float, end_x: float, end_y: float) -> float | None:
    dx = end_x - start_x
    dy = end_y - start_y
    times: list[float] = []
    if dx > 0.0 and end_x > BOARD_SIZE:
        times.append((BOARD_SIZE - start_x) / dx)
    elif dx < 0.0 and end_x < 0.0:
        times.append((0.0 - start_x) / dx)
    if dy > 0.0 and end_y > BOARD_SIZE:
        times.append((BOARD_SIZE - start_y) / dy)
    elif dy < 0.0 and end_y < 0.0:
        times.append((0.0 - start_y) / dy)
    valid = [time for time in times if 0.0 <= time <= 1.0]
    return min(valid) if valid else None


def _bounds_exit_time_jax(
    start_x: jax.Array, start_y: jax.Array, end_x: jax.Array, end_y: jax.Array
) -> tuple[jax.Array, jax.Array]:
    dx = end_x - start_x
    dy = end_y - start_y
    inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    x_upper = jnp.where((dx > 0.0) & (end_x > BOARD_SIZE), (BOARD_SIZE - start_x) / dx, inf)
    x_lower = jnp.where((dx < 0.0) & (end_x < 0.0), (0.0 - start_x) / dx, inf)
    y_upper = jnp.where((dy > 0.0) & (end_y > BOARD_SIZE), (BOARD_SIZE - start_y) / dy, inf)
    y_lower = jnp.where((dy < 0.0) & (end_y < 0.0), (0.0 - start_y) / dy, inf)
    time = jnp.minimum(jnp.minimum(x_upper, x_lower), jnp.minimum(y_upper, y_lower))
    hit = jnp.isfinite(time) & (time >= 0.0) & (time <= 1.0)
    return hit, time


def _acceptable_planet_py(planet: PlanetState, player: int, target_id: int, hit_mode: str) -> bool:
    if hit_mode == "non_friendly":
        return planet.owner != player
    return planet.id == target_id


def _acceptability_mask_jax(game, player: jax.Array, target_id: jax.Array, hit_mode: str) -> jax.Array:
    if hit_mode == "non_friendly":
        return game.planets.owner != player
    return game.planets.id == target_id


def trajectory_shield_reason_for_launch(
    state: GameState,
    source_id: int,
    target_id: int,
    angle: float,
    ships: int,
    env_cfg: Any,
) -> str:
    if not trajectory_shield_enabled(env_cfg):
        return SAFE_REASON
    if ships <= 0:
        return SAFE_REASON
    planets_by_id = {planet.id: planet for planet in state.planets}
    initial_by_id = {planet.id: planet for planet in state.initial_planets}
    source = planets_by_id.get(source_id)
    target = planets_by_id.get(target_id)
    if source is None or target is None:
        return HORIZON_REASON
    horizon = trajectory_shield_horizon(state.step, env_cfg)
    if horizon <= 0:
        return HORIZON_REASON
    epsilon = trajectory_shield_epsilon(env_cfg)
    hit_mode = trajectory_shield_hit_mode(env_cfg)
    old_x = source.x + math.cos(angle) * (source.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    old_y = source.y + math.sin(angle) * (source.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    speed = fleet_speed_py(float(ships), MAX_FLEET_SPEED)

    for offset in range(horizon):
        current_step = state.step + offset
        new_x = old_x + math.cos(angle) * speed
        new_y = old_y + math.sin(angle) * speed
        acceptable_time: float | None = None
        unacceptable_time: float | None = None
        for planet in state.planets:
            initial_planet = initial_by_id.get(planet.id, planet)
            old_px, old_py = _planet_position_at_step(
                planet, initial_planet, state.angular_velocity, current_step
            )
            new_px, new_py = _planet_position_at_step(
                planet, initial_planet, state.angular_velocity, current_step + 1
            )
            hit_time = _moving_circle_hit_time_py(
                old_x,
                old_y,
                new_x,
                new_y,
                old_px,
                old_py,
                new_px,
                new_py,
                planet.radius,
            )
            if hit_time is None:
                continue
            if _acceptable_planet_py(planet, state.player, target_id, hit_mode):
                acceptable_time = hit_time if acceptable_time is None else min(acceptable_time, hit_time)
            else:
                unacceptable_time = hit_time if unacceptable_time is None else min(unacceptable_time, hit_time)

        sun_time = _line_circle_intersection_time_py(
            old_x,
            old_y,
            new_x,
            new_y,
            BOARD_CENTER[0],
            BOARD_CENTER[1],
            SUN_RADIUS,
        )
        bounds_time = _bounds_exit_time_py(old_x, old_y, new_x, new_y)
        block_time_candidates = [
            time
            for time in (sun_time, bounds_time, unacceptable_time)
            if time is not None
        ]
        block_time = min(block_time_candidates) if block_time_candidates else None
        if acceptable_time is not None and (
            block_time is None or acceptable_time + epsilon < block_time
        ):
            return SAFE_REASON
        if sun_time is not None and (acceptable_time is None or sun_time <= acceptable_time + epsilon):
            return SUN_REASON
        if bounds_time is not None and (acceptable_time is None or bounds_time <= acceptable_time + epsilon):
            return BOUNDS_REASON
        if unacceptable_time is not None and (
            acceptable_time is None or unacceptable_time <= acceptable_time + epsilon
        ):
            return UNINTENDED_HIT_REASON
        old_x = new_x
        old_y = new_y
    return HORIZON_REASON


def is_trajectory_safe_for_launch(
    state: GameState,
    source_id: int,
    target_id: int,
    angle: float,
    ships: int,
    env_cfg: Any,
) -> bool:
    return trajectory_shield_reason_for_launch(
        state, source_id, target_id, angle, ships, env_cfg
    ) == SAFE_REASON


def conservative_target_is_safe(
    state: GameState,
    source_id: int,
    target_id: int,
    angle: float,
    source_ships: int,
    env_cfg: Any,
) -> bool:
    if not trajectory_shield_enabled(env_cfg):
        return True
    bucket_count = max(int(getattr(env_cfg, "ship_bucket_count", 1)), 1)
    for bucket in range(1, bucket_count):
        ships = ship_count_for_bucket_py(source_ships, bucket, bucket_count)
        if ships <= 0:
            continue
        if not is_trajectory_safe_for_launch(
            state, source_id, target_id, angle, ships, env_cfg
        ):
            return False
    return True


def infer_target_id_for_move(state: GameState, source_id: int, angle: float) -> int | None:
    source = next((planet for planet in state.planets if planet.id == source_id), None)
    if source is None:
        return None
    best_target_id: int | None = None
    best_error = math.inf
    for planet in state.planets:
        if planet.id == source.id:
            continue
        target_angle = math.atan2(planet.y - source.y, planet.x - source.x)
        angle_error = abs(math.atan2(math.sin(angle - target_angle), math.cos(angle - target_angle)))
        if angle_error < best_error:
            best_error = angle_error
            best_target_id = planet.id
    if best_error > 1e-4:
        return None
    return best_target_id


def filter_moves_with_trajectory_shield(
    moves: list[list[float | int]],
    state: GameState,
    env_cfg: Any,
) -> list[list[float | int]]:
    if not trajectory_shield_enabled(env_cfg):
        return moves
    filtered: list[list[float | int]] = []
    for move in moves:
        if len(move) < 3:
            continue
        source_id = int(move[0])
        angle = float(move[1])
        ships = int(move[2])
        target_id = infer_target_id_for_move(state, source_id, angle)
        if target_id is None:
            continue
        if is_trajectory_safe_for_launch(state, source_id, target_id, angle, ships, env_cfg):
            filtered.append([source_id, angle, ships])
    return filtered


def mask_policy_output_for_shield(
    output: JaxPolicyOutput,
    candidate_mask: jax.Array,
    ship_bucket_count: int,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxPolicyOutput:
    target_logits = ensure_policy_sequence(output.target_logits)
    ship_logits = output.ship_logits
    if ship_logits.ndim == 3:
        ship_logits = ship_logits[:, None, :, :]
    if candidate_mask.ndim == 3:
        candidate_mask = candidate_mask.reshape(-1, candidate_mask.shape[-1])
    base_mask = candidate_mask.astype(bool)
    if ship_bucket_mask is None:
        ship_bucket_mask = default_ship_bucket_mask(base_mask, ship_bucket_count)
    elif ship_bucket_mask.ndim == 4:
        ship_bucket_mask = ship_bucket_mask.reshape(
            -1, ship_bucket_mask.shape[-2], ship_bucket_mask.shape[-1]
        )
    ship_bucket_mask = ship_bucket_mask.astype(bool)
    sequence_k = target_logits.shape[1]
    first_step_mask = (base_mask & ship_bucket_mask.any(axis=-1))[:, None, :]
    if sequence_k > 1:
        noop_only = jnp.zeros((base_mask.shape[0], sequence_k - 1, base_mask.shape[-1]), dtype=bool)
        noop_only = noop_only.at[..., 0].set(True)
        target_mask = jnp.concatenate([first_step_mask, noop_only], axis=1)
    else:
        target_mask = first_step_mask
    illegal_logit = jnp.finfo(jnp.float32).min
    masked_target_logits = jnp.where(target_mask, target_logits, illegal_logit)

    first_step_ship_mask = ship_bucket_mask[:, None, :, :]
    if sequence_k > 1:
        followup_ship_mask = jnp.zeros(
            (
                base_mask.shape[0],
                sequence_k - 1,
                ship_logits.shape[-2],
                ship_logits.shape[-1],
            ),
            dtype=bool,
        )
        followup_ship_mask = followup_ship_mask.at[:, :, 0, 0].set(True)
        ship_mask = jnp.concatenate([first_step_ship_mask, followup_ship_mask], axis=1)
    else:
        ship_mask = first_step_ship_mask
    masked_ship_logits = jnp.where(ship_mask, ship_logits, illegal_logit)
    return output._replace(target_logits=masked_target_logits, ship_logits=masked_ship_logits)


def default_ship_bucket_mask(candidate_mask: jax.Array, ship_bucket_count: int) -> jax.Array:
    if candidate_mask.ndim == 3:
        candidate_mask = candidate_mask.reshape(-1, candidate_mask.shape[-1])
    bucket_ids = jnp.arange(max(int(ship_bucket_count), 1), dtype=jnp.int32)
    candidate_ids = jnp.arange(candidate_mask.shape[-1], dtype=jnp.int32)
    bucket_is_noop = bucket_ids == 0
    candidate_is_noop = candidate_ids == 0
    per_candidate_bucket = jnp.where(
        candidate_is_noop[:, None],
        bucket_is_noop[None, :],
        ~bucket_is_noop[None, :],
    )
    return candidate_mask.astype(bool)[..., None] & per_candidate_bucket


def sample_shielded_policy_actions(
    key: jax.Array,
    output: JaxPolicyOutput,
    *,
    deterministic: bool = False,
) -> ShieldedActionSample:
    key_target, key_ship = jax.random.split(key)
    target_logits = ensure_policy_sequence(output.target_logits)
    ship_logits = output.ship_logits
    if ship_logits.ndim == 3:
        ship_logits = ship_logits[:, None, :, :]
    if deterministic:
        target_index = jnp.argmax(target_logits, axis=-1)
    else:
        target_index = jax.random.categorical(key_target, target_logits, axis=-1)
    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target_index[..., None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=2,
    ).squeeze(axis=2)
    if deterministic:
        ship_bucket = jnp.argmax(selected_ship_logits, axis=-1)
    else:
        ship_bucket = jax.random.categorical(key_ship, selected_ship_logits, axis=-1)
    log_prob, entropy = action_log_prob_and_entropy(output, target_index, ship_bucket)
    return ShieldedActionSample(
        target_index=target_index,
        ship_bucket=ship_bucket,
        log_prob=log_prob,
        entropy=entropy,
    )


def conservative_decision_mask(decision_mask: jax.Array, sequence_k: int) -> jax.Array:
    mask = jnp.broadcast_to(decision_mask[..., None], decision_mask.shape + (sequence_k,))
    if sequence_k <= 1:
        return mask
    return mask.at[..., 1:].set(False)


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
        jnp.maximum(jnp.asarray(MAX_STEPS, dtype=jnp.int32) - game.step.astype(jnp.int32), 0),
    )
    source_x = jnp.take(game.planets.x, source_index)
    source_y = jnp.take(game.planets.y, source_index)
    source_radius = jnp.take(game.planets.radius, source_index)
    speed = 1.0 + (MAX_FLEET_SPEED - 1.0) * (
        jnp.log(jnp.maximum(ships, 1.0)) / jnp.log(1000.0)
    ) ** 1.5
    speed = jnp.minimum(speed, MAX_FLEET_SPEED)

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
        acceptable_mask = hit_mask & _acceptability_mask_jax(game, player, target_id, hit_mode)
        unacceptable_mask = hit_mask & (~_acceptability_mask_jax(game, player, target_id, hit_mode))
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
        safe_hit = jnp.isfinite(acceptable_time) & (acceptable_time + epsilon < block_time)
        sun_blocks = jnp.isfinite(sun_eval) & (~safe_hit) & (sun_eval <= acceptable_time + epsilon)
        bounds_blocks = jnp.isfinite(bounds_eval) & (~safe_hit) & (~sun_blocks) & (bounds_eval <= acceptable_time + epsilon)
        unintended_blocks = jnp.isfinite(unacceptable_time) & (~safe_hit) & (~sun_blocks) & (~bounds_blocks) & (unacceptable_time <= acceptable_time + epsilon)
        resolved = active_step & (safe_hit | sun_blocks | bounds_blocks | unintended_blocks)
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
        return (jnp.where(active_step, new_x, old_x), jnp.where(active_step, new_y, old_y), next_done, next_reason), None

    start_x = source_x + jnp.cos(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = source_y + jnp.sin(angle) * (source_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    initial = (
        start_x,
        start_y,
        ships <= 0.0,
        jnp.where(ships <= 0.0, _REASON_TO_CODE[SAFE_REASON], _REASON_TO_CODE[HORIZON_REASON]),
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


def apply_trajectory_shield_to_turn_batch(game, batch: Any, env_cfg: Any) -> ShieldedBatchResult:
    slot_count = batch.candidate_ids.shape[-1]
    bucket_count = max(int(getattr(env_cfg, "ship_bucket_count", 1)), 1)
    default_bucket_mask = default_ship_bucket_mask(batch.candidate_mask, bucket_count)
    original_real_mask = batch.candidate_mask[:, 1:] if slot_count > 1 else batch.candidate_mask[:, :0]
    original_legal_total = original_real_mask.astype(jnp.float32).sum()
    if not trajectory_shield_enabled(env_cfg):
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
            batch=batch, ship_bucket_mask=default_bucket_mask, diagnostics=diagnostics
        )

    if slot_count <= 1:
        diagnostics = ShieldDiagnostics(
            blocked_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_sun_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_bounds_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_unintended_hit_count=jnp.asarray(0.0, dtype=jnp.float32),
            blocked_horizon_count=jnp.asarray(0.0, dtype=jnp.float32),
            fallback_noop_count=jnp.asarray(0.0, dtype=jnp.float32),
            legal_non_noop_count=jnp.asarray(0.0, dtype=jnp.float32),
            original_non_noop_count=jnp.asarray(0.0, dtype=jnp.float32),
            legal_non_noop_rate=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return ShieldedBatchResult(
            batch=batch, ship_bucket_mask=default_bucket_mask, diagnostics=diagnostics
        )

    real_candidate_ids = batch.candidate_ids[:, 1:]
    real_angles = batch.target_angles[:, 1:]
    bucket_ids = jnp.arange(1, bucket_count, dtype=jnp.int32)
    if bucket_ids.shape[0] == 0:
        legal_bucket_mask = jnp.zeros(
            real_candidate_ids.shape + (0,), dtype=bool
        )
        shielded_real_mask = jnp.zeros_like(original_real_mask, dtype=bool)
        reason_codes = jnp.full_like(real_candidate_ids, _REASON_TO_CODE[HORIZON_REASON])
    else:
        def evaluate_target(source_id, source_ships, target_id, angle, original_mask):
            ship_counts = ship_count_for_bucket_jax(
                jnp.broadcast_to(source_ships, bucket_ids.shape), bucket_ids, bucket_count
            )
            reason_codes = jax.vmap(
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
            bucket_legal = reason_codes == _REASON_TO_CODE[SAFE_REASON]
            legal = jnp.any(bucket_legal)
            first_failure = jnp.argmax(reason_codes != _REASON_TO_CODE[SAFE_REASON])
            reason_code = jnp.where(
                legal,
                _REASON_TO_CODE[SAFE_REASON],
                jnp.where(
                    jnp.any(reason_codes != _REASON_TO_CODE[SAFE_REASON]),
                    reason_codes[first_failure],
                    _REASON_TO_CODE[SAFE_REASON],
                ),
            )
            legal = original_mask & (target_id >= 0) & legal
            return legal, reason_code, (original_mask & (target_id >= 0) & bucket_legal)

        evaluate_slot = jax.vmap(evaluate_target, in_axes=(None, None, 0, 0, 0))
        shielded_real_mask, reason_codes, legal_bucket_mask = jax.vmap(evaluate_slot, in_axes=(0, 0, 0, 0, 0))(
            batch.source_ids,
            batch.source_ships,
            real_candidate_ids,
            real_angles,
            original_real_mask,
        )

    real_bucket_mask = jnp.concatenate(
        [
            jnp.zeros(real_candidate_ids.shape + (1,), dtype=bool),
            legal_bucket_mask,
        ],
        axis=-1,
    )
    ship_bucket_mask = jnp.concatenate(
        [default_bucket_mask[:, :1, :], real_bucket_mask], axis=1
    )

    shielded_candidate_mask = jnp.concatenate(
        [batch.candidate_mask[:, :1], shielded_real_mask], axis=1
    )
    blocked_slots = original_real_mask & (~shielded_real_mask)
    shielded_legal_total = shielded_real_mask.astype(jnp.float32).sum()
    legal_non_noop_rate = jnp.where(
        original_legal_total > 0.0,
        shielded_legal_total / original_legal_total,
        0.0,
    )
    diagnostics = ShieldDiagnostics(
        blocked_count=blocked_slots.astype(jnp.float32).sum(),
        blocked_sun_count=(blocked_slots & (reason_codes == _REASON_TO_CODE[SUN_REASON])).astype(jnp.float32).sum(),
        blocked_bounds_count=(blocked_slots & (reason_codes == _REASON_TO_CODE[BOUNDS_REASON])).astype(jnp.float32).sum(),
        blocked_unintended_hit_count=(blocked_slots & (reason_codes == _REASON_TO_CODE[UNINTENDED_HIT_REASON])).astype(jnp.float32).sum(),
        blocked_horizon_count=(blocked_slots & (reason_codes == _REASON_TO_CODE[HORIZON_REASON])).astype(jnp.float32).sum(),
        fallback_noop_count=((original_real_mask.any(axis=-1)) & (~shielded_real_mask.any(axis=-1))).astype(jnp.float32).sum(),
        legal_non_noop_count=shielded_legal_total,
        original_non_noop_count=original_legal_total,
        legal_non_noop_rate=legal_non_noop_rate,
    )
    return ShieldedBatchResult(
        batch=batch._replace(candidate_mask=shielded_candidate_mask),
        ship_bucket_mask=ship_bucket_mask,
        diagnostics=diagnostics,
    )
