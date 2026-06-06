"""Tests for offline map-pool bake."""

from __future__ import annotations

import numpy as np
import pytest

from src.jax.map_pool.bake import (
    MapPoolBakeError,
    bake_one_entry,
    save_pool_npz,
    stack_entries,
    validate_planet_tables,
    validate_stacked_pool,
)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 123, 256, 511, 777, 2024])
def test_bake_one_entry_produces_valid_planets_and_comets(seed: int):
    entry = bake_one_entry(seed)
    assert entry.planet_active.sum() % 4 == 0
    group_count = int(entry.planet_active.sum() // 4)
    assert 5 <= group_count <= 10
    validate_planet_tables(
        planet_id=entry.planet_id,
        active=entry.planet_active,
        x=entry.planet_x,
        y=entry.planet_y,
        radius=entry.planet_radius,
    )
    assert entry.comet_wave_ok.all()
    assert (entry.comet_path_lengths > 0).any()


def test_validate_planet_tables_rejects_bad_group_count():
    active = np.zeros(60, dtype=bool)
    active[:12] = True
    with pytest.raises(MapPoolBakeError, match="group count"):
        validate_planet_tables(
            planet_id=np.arange(60, dtype=np.int32),
            active=active,
            x=np.zeros(60, dtype=np.float32),
            y=np.zeros(60, dtype=np.float32),
            radius=np.ones(60, dtype=np.float32),
        )


def test_stack_and_validate_round_trip(tmp_path):
    entries = [bake_one_entry(seed) for seed in (0, 1)]
    stacked = stack_entries(entries)
    assert stacked["seed"].shape == (2,)
    validate_stacked_pool(stacked)
    out = tmp_path / "tiny.npz"
    save_pool_npz(str(out), entries)
    loaded = np.load(out)
    assert int(loaded["pool_size"]) == 2
