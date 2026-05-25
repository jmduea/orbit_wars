"""Golden tests for bucket-aware intercept edge features (M4 schema v4)."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from src.config.schema import TaskConfig
from src.features.registry import EDGE_FEATURE_SCHEMA
from src.game.constants import MAX_PLANETS, MAX_STEPS
from src.jax.features import encode_turn

_jax_env = importlib.import_module("src.jax." + "env")
JaxFleetState = _jax_env.JaxFleetState
JaxGameState = _jax_env.JaxGameState
JaxPlanetState = _jax_env.JaxPlanetState


def _cfg(**kwargs) -> TaskConfig:
    base = dict(
        max_fleets=32,
        candidate_count=4,
        player_count=2,
        feature_history_steps=1,
        ship_feature_scale=1000.0,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def _empty_fleets() -> JaxFleetState:
    return JaxFleetState(
        id=jnp.zeros((1,), dtype=jnp.int32),
        owner=jnp.zeros((1,), dtype=jnp.int32),
        x=jnp.zeros((1,), dtype=jnp.float32),
        y=jnp.zeros((1,), dtype=jnp.float32),
        angle=jnp.zeros((1,), dtype=jnp.float32),
        from_planet_id=jnp.zeros((1,), dtype=jnp.int32),
        ships=jnp.zeros((1,), dtype=jnp.float32),
        active=jnp.zeros((1,), dtype=bool),
    )


def _two_planet_game(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    r0: float = 1.0,
    r1: float = 1.0,
    angular_velocity: float = 0.0,
    step: int = 0,
) -> JaxGameState:
    planet_ids = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32).at[0].set(0)
    active = jnp.zeros((MAX_PLANETS,), dtype=bool).at[0].set(True).at[1].set(True)
    x = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    y = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    x = x.at[0].set(x0).at[1].set(x1)
    y = y.at[0].set(y0).at[1].set(y1)
    radius = jnp.full((MAX_PLANETS,), 1.0, dtype=jnp.float32).at[0].set(r0).at[1].set(
        r1
    )
    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=radius,
        ships=jnp.full((MAX_PLANETS,), 100.0, dtype=jnp.float32),
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=active,
    )
    return JaxGameState(
        step=jnp.asarray(step, dtype=jnp.int32),
        player=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=_empty_fleets(),
    )


def _edge_scalar(batch, feature_name: str, src_row: int = 0, slot: int = 0) -> float:
    feature_slice = EDGE_FEATURE_SCHEMA.base_slice(feature_name)
    return float(np.asarray(batch.edge_features[src_row, slot, feature_slice]).reshape(-1)[0])


def _edge_pair(batch, feature_name: str, src_row: int = 0, slot: int = 0) -> np.ndarray:
    feature_slice = EDGE_FEATURE_SCHEMA.base_slice(feature_name)
    return np.asarray(batch.edge_features[src_row, slot, feature_slice])


def test_intercept_static_planet_anchors_match_snapshot_delta() -> None:
    """Non-rotating targets collapse intercept geometry to the snapshot line."""
    cfg = _cfg(candidate_count=4)
    game = _two_planet_game(x0=20.0, y0=50.0, x1=95.0, y1=50.0, r1=6.0)
    batch = encode_turn(game, cfg)

    delta_s1 = _edge_pair(batch, "intercept_delta_coords_s1")
    delta_s6 = _edge_pair(batch, "intercept_delta_coords_s6")
    np.testing.assert_allclose(delta_s1, delta_s6, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(delta_s1[0], -0.75, rtol=0.0, atol=1e-4)


def test_intercept_rotating_planet_anchor_distances_diverge() -> None:
    """Slow and fast anchors disagree when the target orbit is active."""
    cfg = _cfg(candidate_count=4)
    game = _two_planet_game(
        x0=20.0,
        y0=50.0,
        x1=35.0,
        y1=50.0,
        r1=1.0,
        angular_velocity=0.05,
        step=12,
    )
    batch = encode_turn(game, cfg)

    dist_s1 = _edge_scalar(batch, "intercept_distance_s1")
    dist_s6 = _edge_scalar(batch, "intercept_distance_s6")
    assert dist_s1 != dist_s6
    assert dist_s1 > dist_s6


def test_intercept_static_planet_sun_cross_matches_crosses_now() -> None:
    """Snapshot and intercept sun-crossing agree when the target does not rotate."""
    cfg = _cfg(candidate_count=4)
    game = _two_planet_game(x0=20.0, y0=50.0, x1=95.0, y1=60.0, r1=6.0)
    batch = encode_turn(game, cfg)

    crosses_now = _edge_scalar(batch, "crosses_now")
    sun_cross_s1 = _edge_scalar(batch, "sun_cross_at_intercept_s1")
    sun_cross_s6 = _edge_scalar(batch, "sun_cross_at_intercept_s6")
    assert crosses_now == pytest.approx(sun_cross_s1, abs=1e-6)
    assert crosses_now == pytest.approx(sun_cross_s6, abs=1e-6)


def test_intercept_turns_slow_anchor_exceeds_fast_anchor() -> None:
    cfg = _cfg(candidate_count=4)
    game = _two_planet_game(x0=20.0, y0=50.0, x1=80.0, y1=50.0, r1=6.0)
    batch = encode_turn(game, cfg)

    turns_s1 = _edge_scalar(batch, "intercept_turns_s1")
    turns_s6 = _edge_scalar(batch, "intercept_turns_s6")
    assert turns_s1 == pytest.approx(0.12, rel=0.0, abs=1e-4)
    assert turns_s6 == pytest.approx(0.02, rel=0.0, abs=1e-4)
    assert turns_s1 > turns_s6


def test_intercept_turns_clip_saturation_formula() -> None:
    clipped = float(jnp.clip(600.0 / 1.0 / MAX_STEPS, 0.0, 1.0))
    assert clipped == 1.0
