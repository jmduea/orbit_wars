"""Offline map-pool entry generation via reference Python generators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.game.comet_generation import generate_comet_paths
from src.game.constants import (
    BOARD_SIZE,
    COMET_SPAWN_STEPS,
    COMETS_PER_GROUP,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_PLANETS,
    ROTATION_RADIUS_LIMIT,
    TOTAL_COMETS,
)
from src.game.planet_generation import (
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    distance,
    generate_planets,
)

CENTER = BOARD_SIZE / 2.0


@dataclass(frozen=True)
class MapPoolEntry:
    """One baked map: neutral planets, angular velocity, and comet schedules."""

    seed: int
    angular_velocity: float
    planet_id: np.ndarray
    planet_owner: np.ndarray
    planet_x: np.ndarray
    planet_y: np.ndarray
    planet_radius: np.ndarray
    planet_ships: np.ndarray
    planet_production: np.ndarray
    planet_active: np.ndarray
    comet_planet_ids: np.ndarray
    comet_paths_x: np.ndarray
    comet_paths_y: np.ndarray
    comet_path_lengths: np.ndarray
    comet_wave_ok: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "seed": np.array(self.seed, dtype=np.int32),
            "angular_velocity": np.array(self.angular_velocity, dtype=np.float32),
            "planet_id": self.planet_id,
            "planet_owner": self.planet_owner,
            "planet_x": self.planet_x,
            "planet_y": self.planet_y,
            "planet_radius": self.planet_radius,
            "planet_ships": self.planet_ships,
            "planet_production": self.planet_production,
            "planet_active": self.planet_active,
            "comet_planet_ids": self.comet_planet_ids,
            "comet_paths_x": self.comet_paths_x,
            "comet_paths_y": self.comet_paths_y,
            "comet_path_lengths": self.comet_path_lengths,
            "comet_wave_ok": self.comet_wave_ok,
        }


class MapPoolBakeError(ValueError):
    """Raised when a seed cannot produce a valid pool entry."""


def _planets_to_rows(planets: list[list[float | int]]) -> tuple[np.ndarray, ...]:
    ids = np.array([int(p[0]) for p in planets], dtype=np.int32)
    owner = np.array([int(p[1]) for p in planets], dtype=np.int32)
    x = np.array([float(p[2]) for p in planets], dtype=np.float32)
    y = np.array([float(p[3]) for p in planets], dtype=np.float32)
    radius = np.array([float(p[4]) for p in planets], dtype=np.float32)
    ships = np.array([float(p[5]) for p in planets], dtype=np.float32)
    production = np.array([float(p[6]) for p in planets], dtype=np.float32)
    active = np.ones(len(planets), dtype=bool)

    pad = MAX_PLANETS - len(planets)
    if pad < 0:
        raise MapPoolBakeError(f"planet count {len(planets)} exceeds MAX_PLANETS")
    next_id = int(ids.max()) + 1 if len(ids) else 0
    ids = np.concatenate([ids, np.arange(next_id, next_id + pad, dtype=np.int32)])
    owner = np.concatenate([owner, np.full(pad, -1, dtype=np.int32)])
    x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    radius = np.concatenate([radius, np.zeros(pad, dtype=np.float32)])
    ships = np.concatenate([ships, np.zeros(pad, dtype=np.float32)])
    production = np.concatenate([production, np.zeros(pad, dtype=np.float32)])
    active = np.concatenate([active, np.zeros(pad, dtype=bool)])
    return ids, owner, x, y, radius, ships, production, active


def validate_planet_tables(
    *,
    planet_id: np.ndarray,
    active: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
) -> None:
    """Mechanical validity checks aligned with pick-4 parity gates."""

    active_count = int(active.sum())
    group_count = active_count // 4
    if group_count < MIN_PLANET_GROUPS or group_count > MAX_PLANET_GROUPS:
        raise MapPoolBakeError(
            f"group count {group_count} outside [{MIN_PLANET_GROUPS}, {MAX_PLANET_GROUPS}]"
        )

    has_orbiting = False
    for i in range(MAX_PLANETS):
        if not active[i]:
            continue
        dist = math.hypot(float(x[i]) - CENTER, float(y[i]) - CENTER)
        if dist + float(radius[i]) < ROTATION_RADIUS_LIMIT:
            has_orbiting = True
        if not (0.0 <= float(x[i]) <= BOARD_SIZE and 0.0 <= float(y[i]) <= BOARD_SIZE):
            raise MapPoolBakeError(f"planet {i} out of bounds")

    if not has_orbiting:
        raise MapPoolBakeError("no orbiting planet group")

    active_ids = [int(planet_id[i]) for i in range(MAX_PLANETS) if active[i]]
    group_ids = sorted({pid // 4 for pid in active_ids})
    for gid in group_ids:
        slots = [np.where(planet_id == gid * 4 + q)[0][0] for q in range(4)]
        if not all(active[s] for s in slots):
            raise MapPoolBakeError(f"incomplete symmetry group {gid}")
        px, py = float(x[slots[0]]), float(y[slots[0]])
        expected = [
            (px, py),
            (BOARD_SIZE - py, px),
            (py, BOARD_SIZE - px),
            (BOARD_SIZE - px, BOARD_SIZE - py),
        ]
        for q, slot in enumerate(slots):
            ex, ey = expected[q]
            if abs(float(x[slot]) - ex) > 1e-3 or abs(float(y[slot]) - ey) > 1e-3:
                raise MapPoolBakeError(f"symmetry mismatch group {gid} quadrant {q}")

    for i in range(MAX_PLANETS):
        if not active[i]:
            continue
        for j in range(i + 1, MAX_PLANETS):
            if not active[j]:
                continue
            sep = distance((float(x[i]), float(y[i])), (float(x[j]), float(y[j])))
            if sep < float(radius[i]) + float(radius[j]) + 0.5:
                raise MapPoolBakeError(f"planet collision {i}/{j}")


def _pack_path(path: list[list[float]]) -> tuple[np.ndarray, np.ndarray, int]:
    xs = np.zeros(MAX_COMET_PATH_LEN, dtype=np.float32)
    ys = np.zeros(MAX_COMET_PATH_LEN, dtype=np.float32)
    length = 0
    for k, point in enumerate(path):
        if k >= MAX_COMET_PATH_LEN:
            break
        xs[k] = float(point[0])
        ys[k] = float(point[1])
        length = k + 1
    return xs, ys, length


def bake_one_entry(seed: int, *, rng=None) -> MapPoolEntry:
    """Bake one full map entry (planets + all comet waves) from an integer seed."""

    import random

    if rng is None:
        rng = random.Random(seed)

    planets = generate_planets(rng)
    (
        planet_id,
        planet_owner,
        planet_x,
        planet_y,
        planet_radius,
        planet_ships,
        planet_production,
        planet_active,
    ) = _planets_to_rows(planets)
    validate_planet_tables(
        planet_id=planet_id,
        active=planet_active,
        x=planet_x,
        y=planet_y,
        radius=planet_radius,
    )

    angular_velocity = rng.uniform(0.025, 0.05)
    planet_rows = [
        [
            int(planet_id[i]),
            int(planet_owner[i]),
            float(planet_x[i]),
            float(planet_y[i]),
            float(planet_radius[i]),
            float(planet_ships[i]),
            float(planet_production[i]),
        ]
        for i in range(MAX_PLANETS)
        if planet_active[i]
    ]

    comet_planet_ids = np.full((MAX_COMET_GROUPS, COMETS_PER_GROUP), -1, dtype=np.int32)
    comet_paths_x = np.zeros(
        (MAX_COMET_GROUPS, COMETS_PER_GROUP, MAX_COMET_PATH_LEN), dtype=np.float32
    )
    comet_paths_y = np.zeros_like(comet_paths_x)
    comet_path_lengths = np.zeros((MAX_COMET_GROUPS, COMETS_PER_GROUP), dtype=np.int32)
    comet_wave_ok = np.zeros((MAX_COMET_GROUPS,), dtype=bool)

    excluded: list[int] = []
    next_comet_id = int(planet_id[planet_active].max()) + 1

    for g, spawn_step in enumerate(COMET_SPAWN_STEPS):
        paths = generate_comet_paths(
            planet_rows,
            angular_velocity,
            spawn_step,
            comet_planet_ids=excluded,
            rng=rng,
        )
        if paths is None:
            raise MapPoolBakeError(
                f"comet wave failed for seed={seed} spawn_step={spawn_step}"
            )
        comet_wave_ok[g] = True
        base_slot = MAX_PLANETS - TOTAL_COMETS + g * COMETS_PER_GROUP
        for i in range(COMETS_PER_GROUP):
            pid = next_comet_id + i
            comet_planet_ids[g, i] = pid
            xs, ys, length = _pack_path(paths[i])
            comet_paths_x[g, i] = xs
            comet_paths_y[g, i] = ys
            comet_path_lengths[g, i] = length
            excluded.append(pid)
            planet_id[base_slot + i] = pid
        next_comet_id += COMETS_PER_GROUP

    if not comet_wave_ok.all():
        raise MapPoolBakeError(f"incomplete comet schedule for seed={seed}")

    return MapPoolEntry(
        seed=seed,
        angular_velocity=angular_velocity,
        planet_id=planet_id,
        planet_owner=planet_owner,
        planet_x=planet_x,
        planet_y=planet_y,
        planet_radius=planet_radius,
        planet_ships=planet_ships,
        planet_production=planet_production,
        planet_active=planet_active,
        comet_planet_ids=comet_planet_ids,
        comet_paths_x=comet_paths_x,
        comet_paths_y=comet_paths_y,
        comet_path_lengths=comet_path_lengths,
        comet_wave_ok=comet_wave_ok,
    )


def stack_entries(entries: list[MapPoolEntry]) -> dict[str, np.ndarray]:
    """Stack single entries into batched arrays for NPZ persistence."""

    if not entries:
        raise MapPoolBakeError("cannot stack empty entry list")
    keys = entries[0].as_dict().keys()
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        stacked[key] = np.stack([entry.as_dict()[key] for entry in entries], axis=0)
    stacked["pool_size"] = np.array(len(entries), dtype=np.int32)
    return stacked


def save_pool_npz(path: str, entries: list[MapPoolEntry]) -> None:
    np.savez_compressed(path, **stack_entries(entries))


def load_pool_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_stacked_pool(arrays: dict[str, np.ndarray]) -> None:
    """Validate every entry in a stacked pool artifact."""

    pool_size = int(np.asarray(arrays.get("pool_size", arrays["seed"].shape[0])).item())
    for idx in range(pool_size):
        validate_planet_tables(
            planet_id=arrays["planet_id"][idx],
            active=arrays["planet_active"][idx],
            x=arrays["planet_x"][idx],
            y=arrays["planet_y"][idx],
            radius=arrays["planet_radius"][idx],
        )
        if not arrays["comet_wave_ok"][idx].all():
            raise MapPoolBakeError(f"entry {idx} missing comet waves")


__all__ = [
    "MapPoolBakeError",
    "MapPoolEntry",
    "bake_one_entry",
    "load_pool_npz",
    "save_pool_npz",
    "stack_entries",
    "validate_planet_tables",
    "validate_stacked_pool",
]
