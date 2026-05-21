import numpy as np

from src.config import TaskConfig
from src.constants import MAX_PLANETS
from src.features import (
    build_candidate_features,
    build_candidates,
    build_global_features,
    build_self_features,
    candidate_feature_dim,
    global_feature_dim,
    real_candidate_slots,
    self_feature_dim,
)
from src.game_types import GameState, PlanetState, parse_observation


def planet(pid: int, owner: int, x: float, y: float) -> PlanetState:
    return PlanetState(
        id=pid, owner=owner, x=x, y=y, radius=1.0, ships=10, production=1
    )


def test_real_candidate_slots_reserves_index_zero_for_no_op() -> None:
    assert real_candidate_slots(0) == 0
    assert real_candidate_slots(1) == 0
    assert real_candidate_slots(8) == 7


def test_build_candidates_uses_real_slots_and_does_not_cap_enemies_at_one_third() -> (
    None
):
    src = planet(0, 0, 10.0, 10.0)
    enemies = [planet(pid, 1, 12.0 + pid, 10.0) for pid in range(1, 6)]
    neutral = planet(6, -1, 20.0, 10.0)
    friendly = planet(7, 0, 21.0, 10.0)
    state = GameState(
        step=0, player=0, planets=[src, *enemies, neutral, friendly], fleets=[]
    )
    cfg = TaskConfig(candidate_count=8)

    candidates = build_candidates(src, state, cfg)

    assert len(candidates) == 7
    assert sum(candidate.owner == 1 for candidate in candidates) == 5
    assert sum(candidate.owner == -1 for candidate in candidates) == 1
    assert sum(candidate.owner == 0 for candidate in candidates) == 1

def test_owner_relative_features_are_fixed_size_and_player_relative() -> None:
    cfg = TaskConfig(max_fleets=8, candidate_count=4, player_count=4)
    src = PlanetState(id=0, owner=2, x=10.0, y=10.0, radius=1.0, ships=20, production=1)
    targets = [
        PlanetState(id=1, owner=3, x=12.0, y=10.0, radius=1.0, ships=30, production=1),
        PlanetState(id=2, owner=0, x=14.0, y=10.0, radius=1.0, ships=40, production=1),
        PlanetState(id=3, owner=1, x=16.0, y=10.0, radius=1.0, ships=50, production=1),
        PlanetState(id=4, owner=-1, x=18.0, y=10.0, radius=1.0, ships=60, production=1),
    ]
    fleet_type = type("Fleet", (), {})
    fleets = []
    for owner, ships in [(2, 5), (3, 6), (0, 7), (1, 8)]:
        fleet = fleet_type()
        fleet.owner = owner
        fleet.ships = ships
        fleets.append(fleet)
    state = GameState(step=25, player=2, planets=[src, *targets], fleets=fleets)

    self_features = build_self_features(src, state, cfg)
    candidate_features, *_ = build_candidate_features(src, targets, state, cfg)
    global_features = build_global_features(state, cfg)

    assert self_features.shape == (self_feature_dim(),)
    assert candidate_features.shape == (cfg.candidate_count, candidate_feature_dim())
    assert global_features.shape == (global_feature_dim(),)
    np.testing.assert_allclose(self_features[11:15], np.full(4, 1.0 / MAX_PLANETS))
    np.testing.assert_allclose(
        self_features[15:19],
        np.array([20, 30, 40, 50]) / (MAX_PLANETS * cfg.max_ships),
    )
    np.testing.assert_allclose(self_features[19:23], np.ones(4))
    np.testing.assert_allclose(self_features[23], 1.0)
    np.testing.assert_allclose(
        candidate_features[1, 14:18], np.array([0.0, 1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        candidate_features[2, 14:18], np.array([0.0, 0.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(
        candidate_features[3, 14:18], np.array([0.0, 0.0, 0.0, 1.0])
    )
    np.testing.assert_allclose(global_features[8:12], np.full(4, 1.0 / MAX_PLANETS))
    np.testing.assert_allclose(
        global_features[16:20],
        np.array([5, 6, 7, 8]) / (MAX_PLANETS * cfg.max_ships),
    )
    np.testing.assert_allclose(global_features[20:24], np.ones(4))
    np.testing.assert_allclose(global_features[24], 1.0)


def test_parse_observation_preserves_rotation_fields_with_backward_compatible_defaults() -> None:
    parsed = parse_observation(
        {
            "step": 7,
            "player": 1,
            "angular_velocity": 0.125,
            "planets": [[0, 1, 12.0, 13.0, 2.5, 20, 3]],
            "initial_planets": [[0, -1, 10.0, 11.0, 2.5, 18, 3]],
            "fleets": [],
        }
    )

    assert parsed.angular_velocity == 0.125
    assert parsed.initial_planets[0].x == 10.0
    assert parsed.initial_planets[0].owner == -1

    legacy = parse_observation({"step": 3, "player": 0, "planets": [], "fleets": []})

    assert legacy.angular_velocity == 0.0
    assert legacy.initial_planets == []
