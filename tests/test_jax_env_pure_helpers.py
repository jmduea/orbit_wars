"""CPU-light characterization tests for pure JAX env geometry/combat helpers.

Fixtures mirror gold-standard scenarios from
``kaggle_environments.envs.orbit_wars.test_orbit_wars`` (combat resolution,
swept-pair collision, fleet speed scaling).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from src.config import RewardConfig, TaskConfig
from src.game.constants import MAX_PLANETS
from src.game.shield import fleet_speed_py
from src.jax.env import (
    JaxFleetState,
    JaxGameState,
    JaxPlanetState,
    _resolve_combat,
    _shaping,
    _terminal,
    empty_comet_state,
    fleet_speed,
    max_fleets,
    point_to_segment_distance_xy,
    swept_pair_hit,
)

_TEST_CFG = TaskConfig(max_fleets=32, player_count=4)
_FLEET_CAP = max_fleets(_TEST_CFG)


@pytest.mark.parametrize(
    ("ships", "ship_speed"),
    [
        (1, 6.0),
        (10, 6.0),
        (100, 6.0),
        (1000, 6.0),
        (1000, 2.0),
    ],
)
def test_fleet_speed_matches_python_reference(ships: int, ship_speed: float) -> None:
    expected = fleet_speed_py(float(ships), ship_speed)
    actual = float(fleet_speed(jnp.asarray(ships, dtype=jnp.float32), ship_speed))
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-5)


def test_swept_pair_hit_tunnel_case_from_kaggle_gold() -> None:
    """Rotating planet vs horizontal fleet — swept pair catches t=0.5 overlap.

    Mirrors ``TestOrbitWars.test_fleet_does_not_tunnel_through_rotating_planet``.
    """

    hit = swept_pair_hit(
        jnp.asarray(49.0),
        jnp.asarray(50.0),
        jnp.asarray(51.0),
        jnp.asarray(50.0),
        jnp.asarray(50.0),
        jnp.asarray(52.0),
        jnp.asarray(50.0),
        jnp.asarray(48.0),
        jnp.asarray(1.0),
    )
    assert bool(hit)


def test_swept_pair_hit_miss_when_paths_do_not_overlap() -> None:
    hit = swept_pair_hit(
        jnp.asarray(10.0),
        jnp.asarray(10.0),
        jnp.asarray(11.0),
        jnp.asarray(10.0),
        jnp.asarray(90.0),
        jnp.asarray(90.0),
        jnp.asarray(91.0),
        jnp.asarray(90.0),
        jnp.asarray(1.0),
    )
    assert not bool(hit)


def test_point_to_segment_distance_from_kaggle_tunnel_setup() -> None:
    """Static distance cited in the Kaggle tunnel test phase-1 check (= 2.0)."""

    dist = point_to_segment_distance_xy(
        jnp.asarray(50.0),
        jnp.asarray(52.0),
        jnp.asarray(49.0),
        jnp.asarray(50.0),
        jnp.asarray(51.0),
        jnp.asarray(50.0),
    )
    assert float(dist) == pytest.approx(2.0, abs=1e-5)


def _single_planet_state(*, owner: int, ships: float) -> JaxPlanetState:
    active = jnp.zeros((MAX_PLANETS,), dtype=bool).at[0].set(True)
    return JaxPlanetState(
        id=jnp.arange(MAX_PLANETS, dtype=jnp.int32),
        owner=jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32).at[0].set(owner),
        x=jnp.full((MAX_PLANETS,), 80.0, dtype=jnp.float32),
        y=jnp.full((MAX_PLANETS,), 80.0, dtype=jnp.float32),
        radius=jnp.full((MAX_PLANETS,), 5.0, dtype=jnp.float32),
        ships=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32).at[0].set(ships),
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=active,
    )


def _combat_fleets_from_kaggle_user_example() -> JaxFleetState:
    """Fleets from ``TestOrbitWars.test_combat_resolution_user_example``."""

    specs = [
        (0, 41),
        (1, 20),
        (1, 20),
        (2, 42),
    ]
    active = jnp.zeros((_FLEET_CAP,), dtype=bool)
    owner = jnp.full((_FLEET_CAP,), -1, dtype=jnp.int32)
    ships = jnp.zeros((_FLEET_CAP,), dtype=jnp.float32)
    for idx, (fleet_owner, fleet_ships) in enumerate(specs):
        active = active.at[idx].set(True)
        owner = owner.at[idx].set(fleet_owner)
        ships = ships.at[idx].set(float(fleet_ships))
    return JaxFleetState(
        id=jnp.arange(_FLEET_CAP, dtype=jnp.int32),
        owner=owner,
        x=jnp.full((_FLEET_CAP,), 76.0, dtype=jnp.float32),
        y=jnp.full((_FLEET_CAP,), 80.0, dtype=jnp.float32),
        angle=jnp.zeros((_FLEET_CAP,), dtype=jnp.float32),
        from_planet_id=jnp.zeros((_FLEET_CAP,), dtype=jnp.int32),
        ships=ships,
        active=active,
    )


def test_resolve_combat_matches_kaggle_user_example() -> None:
    planets = _single_planet_state(owner=-1, ships=10.0)
    fleets = _combat_fleets_from_kaggle_user_example()
    hit_any = fleets.active
    hit_idx = jnp.zeros((_FLEET_CAP,), dtype=jnp.int32)
    cfg = _TEST_CFG

    resolved = _resolve_combat(planets, fleets, hit_any, hit_idx, cfg)

    assert int(resolved.owner[0]) == -1
    assert float(resolved.ships[0]) == pytest.approx(9.0)


def _minimal_game(planets: JaxPlanetState) -> JaxGameState:
    fleets = JaxFleetState(
        id=jnp.zeros((_FLEET_CAP,), dtype=jnp.int32),
        owner=jnp.full((_FLEET_CAP,), -1, dtype=jnp.int32),
        x=jnp.zeros((_FLEET_CAP,), dtype=jnp.float32),
        y=jnp.zeros((_FLEET_CAP,), dtype=jnp.float32),
        angle=jnp.zeros((_FLEET_CAP,), dtype=jnp.float32),
        from_planet_id=jnp.zeros((_FLEET_CAP,), dtype=jnp.int32),
        ships=jnp.zeros((_FLEET_CAP,), dtype=jnp.float32),
        active=jnp.zeros((_FLEET_CAP,), dtype=bool),
    )
    return JaxGameState(
        step=jnp.asarray(1, dtype=jnp.int32),
        player=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(0.01, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        episode_seed=jnp.asarray(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=fleets,
        comets=empty_comet_state(),
    )


def test_shaping_capture_reward_sign_on_planet_gain() -> None:
    previous = _minimal_game(_single_planet_state(owner=-1, ships=10.0))
    current = _minimal_game(_single_planet_state(owner=0, ships=5.0))
    cfg = RewardConfig(reward_capture_planet=1.0)

    capture_reward, ship_reward, prod_reward = _shaping(
        previous, current, jnp.asarray(0, dtype=jnp.int32), cfg
    )

    assert float(capture_reward) == pytest.approx(1.0)
    assert float(ship_reward) == pytest.approx(0.0)
    assert float(prod_reward) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mode", "learner_wins", "expected_sign"),
    [
        ("binary_win", True, 1.0),
        ("binary_win", False, -1.0),
    ],
)
def test_terminal_binary_win_reward_sign(
    mode: str, learner_wins: bool, expected_sign: float
) -> None:
    if learner_wins:
        planets = _single_planet_state(owner=0, ships=50.0)
    else:
        planets = _single_planet_state(owner=1, ships=50.0)
    game = _minimal_game(planets)
    game = game._replace(step=jnp.asarray(498, dtype=jnp.int32))
    cfg = TaskConfig(player_count=2)
    reward_cfg = RewardConfig(
        terminal_reward_mode=mode,
        early_terminal_reward_shaping_enabled=False,
    )

    done, reward, *_rest = _terminal(
        game, jnp.asarray(0, dtype=jnp.int32), cfg, reward_cfg
    )

    assert bool(done)
    assert float(reward) == pytest.approx(expected_sign)
