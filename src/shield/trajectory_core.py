"""Pure-Python trajectory-shield primitives shared by reference and JAX wrappers."""

from __future__ import annotations

import math

from src.game.constants import BOARD_SIZE, MAX_FLEET_SPEED

# Python reference ↔ JAX trace pairs documented for parity work.
SHIELD_PARITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("ship_count_for_bucket", "ship_count_for_bucket_jax"),
    ("fleet_speed", "_fleet_speed_for_ships_jax"),
    ("moving_circle_hit_time", "_moving_circle_hit_time_jax"),
    ("line_circle_intersection_time", "_line_circle_intersection_time_jax"),
    ("bounds_exit_time", "_bounds_exit_time_jax"),
    ("acceptable_planet_hit", "_acceptability_mask_jax"),
    ("trajectory_shield_reason_for_launch", "trajectory_shield_reason_for_launch_jax"),
    (
        "filter_moves_with_trajectory_shield",
        "apply_trajectory_shield_factorized_topk",
    ),
)


def ship_count_for_bucket(
    available_ships: float | int, bucket: int, bucket_count: int
) -> int:
    """Discrete ship count for a launch bucket index."""

    available = max(0, int(available_ships))
    if available <= 0 or bucket <= 0:
        return 0
    fraction = float(bucket) / float(max(bucket_count - 1, 1))
    ships = int(math.ceil(available * fraction))
    return min(available, max(1, ships))


def fleet_speed(ships: float, *, ship_speed: float = MAX_FLEET_SPEED) -> float:
    """Fleet speed from ship count using the competition log curve."""

    safe = max(float(ships), 1.0)
    speed = 1.0 + (ship_speed - 1.0) * (math.log(safe) / math.log(1000.0)) ** 1.5
    return min(speed, ship_speed)


def moving_circle_hit_time(
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
    """Earliest segment parameter in ``[0, 1]`` where a moving fleet hits a moving circle."""

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


def line_circle_intersection_time(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    center_x: float,
    center_y: float,
    radius: float,
) -> float | None:
    """Earliest segment parameter in ``[0, 1]`` where a segment hits a static circle."""

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


def bounds_exit_time(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> float | None:
    """Earliest segment parameter in ``[0, 1]`` where a segment exits the board."""

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


def acceptable_planet_hit(
    *,
    planet_id: int,
    planet_owner: int,
    player: int,
    target_id: int,
    hit_mode: str,
) -> bool:
    """Return whether hitting ``planet_id`` is allowed for the configured hit mode."""

    if hit_mode == "non_friendly":
        return planet_owner != player
    return planet_id == target_id
