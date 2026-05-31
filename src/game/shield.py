from __future__ import annotations

import math
from typing import Any

from src.game.constants import (
    BOARD_CENTER,
    BOARD_SIZE,
    MAX_FLEET_SPEED,
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
    trajectory_shield_horizon,
    trajectory_shield_mode,
)
from src.game.types import GameState, PlanetState


def ship_count_for_bucket_py(
    available_ships: float | int, bucket: int, bucket_count: int
) -> int:
    available = max(0, int(available_ships))
    if available <= 0 or bucket <= 0:
        return 0
    fraction = float(bucket) / float(max(bucket_count - 1, 1))
    ships = int(math.ceil(available * fraction))
    return min(available, max(1, ships))


def fleet_speed_py(ships: float, ship_speed: float = MAX_FLEET_SPEED) -> float:
    safe = max(float(ships), 1.0)
    speed = 1.0 + (ship_speed - 1.0) * (math.log(safe) / math.log(1000.0)) ** 1.5
    return min(speed, ship_speed)


def _planet_position_at_step(
    planet: PlanetState,
    initial_planet: PlanetState,
    angular_velocity: float,
    step_index: int,
) -> tuple[float, float]:
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


def _bounds_exit_time_py(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> float | None:
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


def _acceptable_planet_py(
    planet: PlanetState, player: int, target_id: int, hit_mode: str
) -> bool:
    if hit_mode == "non_friendly":
        return planet.owner != player
    return planet.id == target_id


def trajectory_shield_reason_for_launch(
    state: GameState,
    source_id: int,
    target_id: int,
    angle: float,
    ships: int,
    env_cfg: Any,
) -> str:
    if trajectory_shield_mode(env_cfg) == "off":
        return SAFE_REASON
    if ships <= 0:
        return SAFE_REASON
    planets_by_id = {planet.id: planet for planet in state.planets}
    initial_by_id = {planet.id: planet for planet in state.initial_planets}
    source = planets_by_id.get(source_id)
    if source is None or planets_by_id.get(target_id) is None:
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
                acceptable_time = (
                    hit_time
                    if acceptable_time is None
                    else min(acceptable_time, hit_time)
                )
            else:
                unacceptable_time = (
                    hit_time
                    if unacceptable_time is None
                    else min(unacceptable_time, hit_time)
                )

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
        if sun_time is not None and (
            acceptable_time is None or sun_time <= acceptable_time + epsilon
        ):
            return SUN_REASON
        if bounds_time is not None and (
            acceptable_time is None or bounds_time <= acceptable_time + epsilon
        ):
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
    return (
        trajectory_shield_reason_for_launch(
            state, source_id, target_id, angle, ships, env_cfg
        )
        == SAFE_REASON
    )


def infer_target_id_for_move(
    state: GameState, source_id: int, angle: float
) -> int | None:
    source = next((planet for planet in state.planets if planet.id == source_id), None)
    if source is None:
        return None
    best_target_id: int | None = None
    best_error = math.inf
    for planet in state.planets:
        if planet.id == source.id:
            continue
        target_angle = math.atan2(planet.y - source.y, planet.x - source.x)
        angle_error = abs(
            math.atan2(math.sin(angle - target_angle), math.cos(angle - target_angle))
        )
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
    if trajectory_shield_mode(env_cfg) == "off":
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
        if is_trajectory_safe_for_launch(
            state, source_id, target_id, angle, ships, env_cfg
        ):
            filtered.append([source_id, angle, ships])
    return filtered
