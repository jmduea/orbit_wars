"""Unit tests for reference comet path generation."""

from __future__ import annotations

import random

from src.game.comet_generation import generate_comet_paths
from src.game.constants import BOARD_SIZE
from src.game.planet_generation import generate_planets, planets_to_padded_rows


def _initial_planet_rows(seed: int) -> list[list[float | int]]:
    rng = random.Random(seed)
    planets = generate_planets(rng)
    ids, owner, x, y, radius, ships, production, active = planets_to_padded_rows(
        planets
    )
    rows: list[list[float | int]] = []
    for i in range(len(planets)):
        if not active[i]:
            continue
        rows.append(
            [ids[i], owner[i], x[i], y[i], radius[i], ships[i], production[i]]
        )
    return rows


def test_generate_comet_paths_fixed_seed_produces_four_symmetric_paths():
    initial = _initial_planet_rows(0)
    rng = random.Random("orbit_wars-comet-0-50")
    paths = generate_comet_paths(initial, 0.04, 50, [], comet_speed=4.0, rng=rng)
    assert paths is not None
    assert len(paths) == 4
    for path in paths:
        assert 5 <= len(path) <= 40
        for x, y in path:
            assert 0 <= x <= BOARD_SIZE
            assert 0 <= y <= BOARD_SIZE

    q1 = [[pt[1], pt[0]] for pt in paths[0]]
    assert paths[1] == [[BOARD_SIZE - x, y] for x, y in q1]
    assert paths[2] == [[x, BOARD_SIZE - y] for x, y in q1]
    assert paths[3] == [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in q1]
