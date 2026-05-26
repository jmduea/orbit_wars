from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np

from src.config.schema import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.game.trajectory_shield import (
    apply_cheap_trajectory_shield_factorized_topk,
    apply_configured_trajectory_shield_factorized_topk,
    trajectory_shield_mode,
)
from src.jax.features import encode_turn

_jax_env = importlib.import_module("src.jax." + "env")
JaxFleetState = _jax_env.JaxFleetState
JaxGameState = _jax_env.JaxGameState
JaxPlanetState = _jax_env.JaxPlanetState


def _cfg(**kwargs) -> TaskConfig:
    base = dict(
        max_fleets=32,
        candidate_count=4,
        ship_bucket_count=8,
        player_count=2,
        feature_history_steps=1,
        ship_feature_scale=1000.0,
        trajectory_shield_enabled=True,
        trajectory_shield_mode="cheap",
        trajectory_shield_horizon=30,
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
    source_ships: float = 100.0,
    target_ships: float = 20.0,
) -> JaxGameState:
    planet_ids = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32).at[0].set(0)
    active = jnp.zeros((MAX_PLANETS,), dtype=bool).at[0].set(True).at[1].set(True)

    x = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    y = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    x = x.at[0].set(x0).at[1].set(x1)
    y = y.at[0].set(y0).at[1].set(y1)

    ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    ships = ships.at[0].set(source_ships).at[1].set(target_ships)

    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=jnp.full((MAX_PLANETS,), 1.0, dtype=jnp.float32),
        ships=ships,
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=active,
    )

    return JaxGameState(
        step=jnp.asarray(0, dtype=jnp.int32),
        player=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(0.0, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=_empty_fleets(),
    )


def test_trajectory_shield_mode_normalizes_disabled() -> None:
    cfg = _cfg(trajectory_shield_enabled=False)
    assert trajectory_shield_mode(cfg) == "off"


def test_cheap_factorized_shield_returns_expected_shapes() -> None:
    cfg = _cfg()
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, cfg)

    result = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)

    assert result.ship_bucket_mask.shape == (
        MAX_PLANETS,
        edge_k(cfg),
        cfg.ship_bucket_count,
    )
    assert result.batch.edge_mask.shape == (MAX_PLANETS, edge_k(cfg))
    assert np.isfinite(float(result.diagnostics.legal_non_noop_rate))


def test_cheap_factorized_shield_allows_safe_nonzero_bucket() -> None:
    cfg = _cfg()
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, cfg)

    result = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)

    assert bool(np.asarray(result.ship_bucket_mask[0, 0, 1:]).any())


def test_cheap_factorized_shield_blocks_when_source_has_no_ships() -> None:
    cfg = _cfg()
    game = _two_planet_game(
        x0=20.0,
        y0=20.0,
        x1=80.0,
        y1=20.0,
        source_ships=0.0,
    )
    batch = encode_turn(game, cfg)

    result = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)

    assert not bool(np.asarray(result.ship_bucket_mask[0, 0, 1:]).any())


def test_configured_factorized_shield_dispatches_cheap_mode() -> None:
    cfg = _cfg(trajectory_shield_mode="cheap")
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, cfg)

    result = apply_configured_trajectory_shield_factorized_topk(game, batch, cfg)

    assert result.ship_bucket_mask.shape == (
        MAX_PLANETS,
        edge_k(cfg),
        cfg.ship_bucket_count,
    )
