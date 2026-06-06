"""Tests for map-pool JAX loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.jax.map_pool.bake import bake_one_entry, save_pool_npz
from src.jax.map_pool.load import (
    load_map_pool,
    load_map_pool_numpy,
    read_manifest_sha256,
)


def test_load_tiny_pool_shapes(tmp_path: Path):
    entries = [bake_one_entry(0), bake_one_entry(1)]
    pool = tmp_path / "tiny.npz"
    save_pool_npz(str(pool), entries)
    constants = load_map_pool(pool)
    assert constants.pool_size == 2
    assert constants.planet_x.shape == (2, 60)
    assert constants.comet_paths_x.shape[0] == 2


def test_load_missing_pool_raises():
    with pytest.raises(FileNotFoundError, match="map pool not found"):
        load_map_pool_numpy("/nonexistent/pool.npz")


def test_read_manifest_sha256_default_v1():
    sha = read_manifest_sha256("data/jax_map_pool/default_v1.npz")
    assert sha == "b48a38e4ad63aab25ffb155af321ab470b4e658cff42eca0369d72048f8d3023"
