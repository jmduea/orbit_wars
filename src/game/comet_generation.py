"""Reference Orbit Wars comet path generation matching Kaggle ``generate_comet_paths``."""

from __future__ import annotations

import math
import random
from typing import Sequence

from src.game.constants import (
    BOARD_SIZE,
    COMET_RADIUS,
    COMET_SPEED,
    ROTATION_RADIUS_LIMIT,
)
from src.game.planet_generation import distance

CENTER = BOARD_SIZE / 2.0


def generate_comet_paths(
    initial_planets: Sequence[Sequence[float | int]],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: Sequence[int] | None = None,
    comet_speed: float = COMET_SPEED,
    rng: random.Random | None = None,
) -> list[list[list[float]]] | None:
    """Generate four symmetric elliptical comet paths for one spawn wave.

    Returns four paths (one per quadrant symmetry), each a list of ``[x, y]``
    positions spaced at ``comet_speed`` arc-length, or ``None`` on failure.
    """

    if rng is None:
        rng = random.Random()
    excluded = set(comet_planet_ids or ())
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < 10.0 + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense: list[tuple[float, float]] = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets: list[Sequence[float | int]] = []
        orbiting_planets: list[Sequence[float | int]] = []
        for planet in initial_planets:
            if int(planet[0]) in excluded:
                continue
            pr = distance((float(planet[2]), float(planet[3])), (CENTER, CENTER))
            if pr + float(planet[4]) < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if distance((cx, cy), (CENTER, CENTER)) < 10.0 + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (float(planet[2]), float(planet[3]))) < float(
                        planet[4]
                    ) + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = float(planet[2]) - CENTER
                dy = float(planet[3]) - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < float(planet[4]) + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


__all__ = ["generate_comet_paths"]
