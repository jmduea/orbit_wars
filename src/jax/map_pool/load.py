"""Load offline map-pool artifacts into frozen JAX arrays for training init."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import jax
from src.jax.map_pool.bake import load_pool_npz, validate_stacked_pool


class MapPoolConstants(NamedTuple):
    """Stacked pool tensors with leading ``[pool_size, ...]`` dimension."""

    pool_size: int
    sha256: str | None
    angular_velocity: jax.Array
    planet_id: jax.Array
    planet_owner: jax.Array
    planet_x: jax.Array
    planet_y: jax.Array
    planet_radius: jax.Array
    planet_ships: jax.Array
    planet_production: jax.Array
    planet_active: jax.Array
    comet_planet_ids: jax.Array
    comet_paths_x: jax.Array
    comet_paths_y: jax.Array
    comet_path_lengths: jax.Array
    comet_wave_ok: jax.Array


def _resolve_pool_path(path: str | Path) -> Path:
    pool_path = Path(path)
    if pool_path.is_file():
        return pool_path.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / pool_path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"map pool not found: {path}")


def _manifest_path_for(pool_path: Path) -> Path | None:
    stem = pool_path.with_suffix("")
    manifest = Path(f"{stem}.manifest.json")
    return manifest if manifest.is_file() else None


def read_manifest_sha256(path: str | Path) -> str | None:
    """Return sha256 from a pool sidecar manifest when present."""

    pool_path = _resolve_pool_path(path)
    manifest_path = _manifest_path_for(pool_path)
    if manifest_path is None:
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = data.get("sha256")
    return str(sha) if sha else None


def load_map_pool_numpy(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a stacked pool artifact from disk."""

    pool_path = _resolve_pool_path(path)
    arrays = load_pool_npz(str(pool_path))
    validate_stacked_pool(arrays)
    return arrays


def map_pool_constants_from_numpy(
    arrays: dict[str, np.ndarray],
    *,
    sha256: str | None = None,
) -> MapPoolConstants:
    pool_size = int(np.asarray(arrays.get("pool_size", arrays["seed"].shape[0])).item())
    return MapPoolConstants(
        pool_size=pool_size,
        sha256=sha256,
        angular_velocity=jnp.asarray(arrays["angular_velocity"], dtype=jnp.float32),
        planet_id=jnp.asarray(arrays["planet_id"], dtype=jnp.int32),
        planet_owner=jnp.asarray(arrays["planet_owner"], dtype=jnp.int32),
        planet_x=jnp.asarray(arrays["planet_x"], dtype=jnp.float32),
        planet_y=jnp.asarray(arrays["planet_y"], dtype=jnp.float32),
        planet_radius=jnp.asarray(arrays["planet_radius"], dtype=jnp.float32),
        planet_ships=jnp.asarray(arrays["planet_ships"], dtype=jnp.float32),
        planet_production=jnp.asarray(arrays["planet_production"], dtype=jnp.float32),
        planet_active=jnp.asarray(arrays["planet_active"], dtype=bool),
        comet_planet_ids=jnp.asarray(arrays["comet_planet_ids"], dtype=jnp.int32),
        comet_paths_x=jnp.asarray(arrays["comet_paths_x"], dtype=jnp.float32),
        comet_paths_y=jnp.asarray(arrays["comet_paths_y"], dtype=jnp.float32),
        comet_path_lengths=jnp.asarray(arrays["comet_path_lengths"], dtype=jnp.int32),
        comet_wave_ok=jnp.asarray(arrays["comet_wave_ok"], dtype=bool),
    )


def load_map_pool(path: str | Path) -> MapPoolConstants:
    """Load a validated map pool and convert tensors to JAX arrays."""

    pool_path = _resolve_pool_path(path)
    sha256 = read_manifest_sha256(pool_path)
    arrays = load_map_pool_numpy(pool_path)
    return map_pool_constants_from_numpy(arrays, sha256=sha256)


__all__ = [
    "MapPoolConstants",
    "load_map_pool",
    "load_map_pool_numpy",
    "map_pool_constants_from_numpy",
    "read_manifest_sha256",
]
