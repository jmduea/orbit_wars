"""Reference Orbit Wars planet layout matching Kaggle ``generate_planets``."""

from __future__ import annotations

import math
import random
from typing import Sequence

from src.game.constants import (
    BOARD_SIZE,
    MAX_PLANETS,
    ROTATION_RADIUS_LIMIT,
)

CENTER = BOARD_SIZE / 2.0
PLANET_CLEARANCE = 7.0
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_planets(rng: random.Random | None = None) -> list[list[float | int]]:
    """Generate symmetric planet groups using the Kaggle reference algorithm."""

    if rng is None:
        rng = random.Random()
    planets: list[list[float | int]] = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        valid = True
        for tp in temp_planets:
            for p in planets:
                if (
                    distance((p[2], p[3]), (tp[2], tp[3]))
                    < p[4] + tp[4] + PLANET_CLEARANCE
                ):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)

        orbital_radius = distance((x, y), (CENTER, CENTER))

        if orbital_radius < 10.0 + r + 10:
            continue

        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue

        valid = True
        ships = rng.randint(5, 30)
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT

                if (
                    distance((p[2], p[3]), (tp[2], tp[3]))
                    < p[4] + tp[4] + PLANET_CLEARANCE
                ):
                    valid = False
                    break

                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break

            if not valid:
                break

        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4

    return planets


def assign_home_planets(
    planets: list[list[float | int]],
    *,
    player_count: int,
    home_group: int,
) -> None:
    """Assign home planets in-place using Kaggle home-group rules."""

    num_groups = len(planets) // 4
    if num_groups <= 0:
        return
    home_group = home_group % num_groups
    base = home_group * 4

    if player_count == 2:
        planets[base][1] = 0
        planets[base][5] = 10
        planets[base + 3][1] = 1
        planets[base + 3][5] = 10
    elif player_count >= 4:
        for j in range(4):
            planets[base + j][1] = j
            planets[base + j][5] = 10


def planets_to_padded_rows(
    planets: list[list[float | int]],
) -> tuple[
    list[int],
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[bool],
]:
    """Pack reference planet rows into fixed ``MAX_PLANETS`` tables."""

    ids = [int(p[0]) for p in planets]
    owner = [int(p[1]) for p in planets]
    x = [float(p[2]) for p in planets]
    y = [float(p[3]) for p in planets]
    radius = [float(p[4]) for p in planets]
    ships = [float(p[5]) for p in planets]
    production = [float(p[6]) for p in planets]
    active = [True] * len(planets)

    pad = MAX_PLANETS - len(planets)
    if pad < 0:
        raise ValueError(
            f"generate_planets produced {len(planets)} planets > MAX_PLANETS"
        )
    next_id = max(ids) + 1 if ids else 0
    ids.extend(range(next_id, next_id + pad))
    owner.extend([-1] * pad)
    x.extend([0.0] * pad)
    y.extend([0.0] * pad)
    radius.extend([0.0] * pad)
    ships.extend([0.0] * pad)
    production.extend([0.0] * pad)
    active.extend([False] * pad)
    return ids, owner, x, y, radius, ships, production, active


__all__ = [
    "assign_home_planets",
    "generate_planets",
    "planets_to_padded_rows",
]
