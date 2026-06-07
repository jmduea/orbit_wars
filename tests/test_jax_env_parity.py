import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import RewardConfig, TaskConfig
from src.game.constants import MAX_PLANETS
from src.jax.env import (
    BOARD_CENTER,
    JaxAction,
    JaxEnvState,
    JaxFleetState,
    JaxGameState,
    JaxPlanetState,
    empty_action,
    empty_comet_state,
    max_fleets,
    reset,
    reset_with_pool,
    step,
    step_multi_player,
)
from src.jax.map_pool.comets import empty_comet_state


def _cfg(
    *,
    player_count=2,
    max_fleets=16,
    ship_speed=6.0,
    env_parity_mode: str = "train",
):
    return TaskConfig(
        max_fleets=max_fleets,
        candidate_count=4,
        player_count=player_count,
        ship_speed=ship_speed,
        env_parity_mode=env_parity_mode,
    )


def _reward_cfg():
    return RewardConfig()


def _planet_state(rows, cfg):
    rows = [list(row) for row in rows]
    pad = MAX_PLANETS - len(rows)
    assert pad >= 0
    ids = [int(r[0]) for r in rows] + list(range(len(rows), MAX_PLANETS))
    owner = [int(r[1]) for r in rows] + [-1] * pad
    x = [float(r[2]) for r in rows] + [0.0] * pad
    y = [float(r[3]) for r in rows] + [0.0] * pad
    radius = [float(r[4]) for r in rows] + [0.0] * pad
    ships = [float(r[5]) for r in rows] + [0.0] * pad
    production = [float(r[6]) for r in rows] + [0.0] * pad
    active = [True] * len(rows) + [False] * pad
    return JaxPlanetState(
        id=jnp.array(ids, dtype=jnp.int32),
        owner=jnp.array(owner, dtype=jnp.int32),
        x=jnp.array(x, dtype=jnp.float32),
        y=jnp.array(y, dtype=jnp.float32),
        radius=jnp.array(radius, dtype=jnp.float32),
        ships=jnp.array(ships, dtype=jnp.float32),
        production=jnp.array(production, dtype=jnp.float32),
        active=jnp.array(active, dtype=bool),
    )


def _fleet_state(rows, cfg):
    rows = [list(row) for row in rows]
    fleet_count = max_fleets(cfg)
    pad = fleet_count - len(rows)
    assert pad >= 0
    ids = [int(r[0]) for r in rows] + [-1] * pad
    owner = [int(r[1]) for r in rows] + [-1] * pad
    x = [float(r[2]) for r in rows] + [0.0] * pad
    y = [float(r[3]) for r in rows] + [0.0] * pad
    angle = [float(r[4]) for r in rows] + [0.0] * pad
    source = [int(r[5]) for r in rows] + [-1] * pad
    ships = [float(r[6]) for r in rows] + [0.0] * pad
    active = [True] * len(rows) + [False] * pad
    return JaxFleetState(
        id=jnp.array(ids, dtype=jnp.int32),
        owner=jnp.array(owner, dtype=jnp.int32),
        x=jnp.array(x, dtype=jnp.float32),
        y=jnp.array(y, dtype=jnp.float32),
        angle=jnp.array(angle, dtype=jnp.float32),
        from_planet_id=jnp.array(source, dtype=jnp.int32),
        ships=jnp.array(ships, dtype=jnp.float32),
        active=jnp.array(active, dtype=bool),
    )


def _state(
    planets,
    fleets=(),
    *,
    cfg=None,
    step_index=0,
    angular_velocity=0.01,
    learner_player=0,
):
    cfg = cfg or _cfg()
    planet_state = _planet_state(planets, cfg)
    fleet_state = _fleet_state(fleets, cfg)
    game = JaxGameState(
        step=jnp.array(step_index, dtype=jnp.int32),
        player=jnp.array(learner_player, dtype=jnp.int32),
        angular_velocity=jnp.array(angular_velocity, dtype=jnp.float32),
        next_fleet_id=jnp.array(100, dtype=jnp.int32),
        planets=planet_state,
        initial_planets=planet_state,
        fleets=fleet_state,
        comets=empty_comet_state(),
    )
    return JaxEnvState(
        game=game,
        learner_player=jnp.array(learner_player, dtype=jnp.int32),
        episode_count=jnp.array(0, dtype=jnp.int32),
    )


def _advance(state, cfg, reward_cfg=None):
    return step(
        state,
        empty_action(cfg),
        empty_action(cfg),
        cfg,
        reward_cfg or _reward_cfg(),
    )


def _multi_actions(cfg, moves_by_player):
    actions = []
    for player in range(cfg.player_count):
        action = empty_action(cfg)
        for slot, (source_id, angle, ships) in enumerate(
            moves_by_player.get(player, [])
        ):
            action = JaxAction(
                source_id=action.source_id.at[slot].set(source_id),
                angle=action.angle.at[slot].set(angle),
                ships=action.ships.at[slot].set(ships),
                valid=action.valid.at[slot].set(True),
            )
        actions.append(action)
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *actions)


def _planet_rows(state, count):
    p = state.game.planets
    return [
        [
            int(np.asarray(p.id)[i]),
            int(np.asarray(p.owner)[i]),
            float(np.asarray(p.x)[i]),
            float(np.asarray(p.y)[i]),
            float(np.asarray(p.radius)[i]),
            float(np.asarray(p.ships)[i]),
            float(np.asarray(p.production)[i]),
        ]
        for i in range(count)
    ]


def test_reset_generates_fourfold_symmetric_planet_groups_for_two_players():
    cfg = _cfg()
    state, _ = reset(jax.random.PRNGKey(11), cfg)
    active_count = int(np.asarray(state.game.planets.active).sum())
    planets = _planet_rows(state, active_count)

    assert active_count % 4 == 0
    for i in range(0, active_count, 4):
        p0 = planets[i]
        p3 = planets[i + 3]
        assert math.isclose(p0[2] + p3[2], 100.0, abs_tol=1e-5)
        assert math.isclose(p0[3] + p3[3], 100.0, abs_tol=1e-5)
        assert p0[4] == p3[4]


def test_four_player_reset_home_planets_are_rotationally_symmetric():
    cfg = _cfg(player_count=4)
    for seed in range(50):
        state, _ = reset(jax.random.PRNGKey(seed), cfg)
    p = state.game.planets
    owned = np.flatnonzero(np.asarray(p.owner) != -1)

    assert len(owned) == 4
    assert set(np.asarray(p.owner)[owned].tolist()) == {0, 1, 2, 3}
    positions = [(float(np.asarray(p.x)[i]), float(np.asarray(p.y)[i])) for i in owned]
    for x, y in positions:
        rx = BOARD_CENTER[0] - (y - BOARD_CENTER[1])
        ry = BOARD_CENTER[1] + (x - BOARD_CENTER[0])
        assert any(
            math.isclose(rx, px, abs_tol=1e-5) and math.isclose(ry, py, abs_tol=1e-5)
            for px, py in positions
        )


def test_fleet_does_not_tunnel_through_rotating_planet():
    cfg = _cfg(ship_speed=2.0)
    state = _state(
        [[0, -1, 50.0, 52.0, 1.0, 10, 0]],
        [[0, 0, 49.0, 50.0, 0.0, 1, 1000]],
        cfg=cfg,
        step_index=1,
        angular_velocity=math.pi,
    )

    next_state, _ = _advance(state, cfg)

    assert int(np.asarray(next_state.game.fleets.active).sum()) == 0
    assert int(np.asarray(next_state.game.planets.owner)[0]) == 0
    assert float(np.asarray(next_state.game.planets.ships)[0]) == 990.0


def test_fleets_are_removed_when_they_hit_sun_or_leave_board_but_not_inside_board():
    cfg = _cfg()
    cases = [
        ([[0, 0, 80, 50, 3, 50, 1]], [[0, 0, 60, 50, math.pi, 0, 10]], 0),
        ([[0, 0, 80, 50, 3, 50, 1]], [[0, 0, 99.5, 50, 0.0, 0, 10]], 0),
        ([[0, 0, 80, 80, 3, 50, 1]], [[0, 0, 50, 30, 0.0, 0, 10]], 1),
    ]
    for planets, fleets, expected_active in cases:
        next_state, _ = _advance(_state(planets, fleets, cfg=cfg), cfg)
        assert int(np.asarray(next_state.game.fleets.active).sum()) == expected_active


def test_fast_fleet_hits_planet_before_leaving_board_or_sun():
    cfg = _cfg()
    board_state, _ = _advance(
        _state(
            [[0, 1, 98.0, 50.0, 2.0, 50, 1]],
            [[0, 0, 95.0, 50.0, 0.0, 99, 1000]],
            cfg=cfg,
        ),
        cfg,
    )
    sun_state, _ = _advance(
        _state(
            [[0, 1, 62.0, 50.0, 2.0, 50, 1]],
            [[0, 0, 65.0, 50.0, math.pi, 99, 1000]],
            cfg=cfg,
            angular_velocity=0.0,
        ),
        cfg,
    )

    for next_state in (board_state, sun_state):
        assert int(np.asarray(next_state.game.planets.owner)[0]) == 0
        assert float(np.asarray(next_state.game.planets.ships)[0]) == 949.0
        assert int(np.asarray(next_state.game.fleets.active).sum()) == 0


def test_combat_resolution_user_example_for_four_players():
    cfg = _cfg(player_count=4)
    state = _state(
        [[0, -1, 80, 80, 5, 10, 0]],
        [
            [0, 0, 76.0, 80.0, 0.0, 1, 41],
            [1, 1, 76.0, 80.0, 0.0, 2, 20],
            [2, 1, 76.0, 80.0, 0.0, 2, 20],
            [3, 2, 76.0, 80.0, 0.0, 3, 42],
        ],
        cfg=cfg,
    )

    next_state, _ = _advance(state, cfg)

    assert int(np.asarray(next_state.game.planets.owner)[0]) == -1
    assert float(np.asarray(next_state.game.planets.ships)[0]) == 9.0


def test_combat_capture_reinforce_insufficient_tie_and_multi_fleet_cases():
    cfg = _cfg()
    cases = [
        ([[0, -1, 80, 50, 3, 10, 1]], [[0, 0, 76, 50, 0, 99, 30]], 0, 20),
        ([[0, 0, 80, 50, 3, 10, 1]], [[0, 0, 76, 50, 0, 99, 25]], 0, 36),
        ([[0, -1, 80, 50, 3, 20, 1]], [[0, 0, 76, 50, 0, 99, 5]], -1, 15),
        (
            [[0, -1, 80, 50, 3, 10, 1]],
            [[0, 0, 76, 50, 0, 99, 50], [1, 1, 76, 50, 0, 99, 30]],
            0,
            10,
        ),
        (
            [[0, -1, 80, 50, 3, 10, 1]],
            [[0, 0, 76, 50, 0, 99, 30], [1, 1, 76, 50, 0, 99, 30]],
            -1,
            10,
        ),
        (
            [[0, 0, 80, 50, 3, 15, 1]],
            [[0, 0, 76, 50, 0, 99, 40], [1, 1, 76, 50, 0, 99, 25]],
            0,
            31,
        ),
        (
            [[0, 1, 80, 50, 3, 5, 1]],
            [[0, 0, 76, 50, 0, 99, 50], [1, 1, 76, 50, 0, 99, 20]],
            0,
            24,
        ),
        (
            [[0, -1, 80, 50, 3, 10, 1]],
            [[0, 0, 76, 50, 0, 99, 20], [1, 0, 76, 50, 0, 99, 15]],
            0,
            25,
        ),
    ]
    for planets, fleets, expected_owner, expected_ships in cases:
        next_state, _ = _advance(_state(planets, fleets, cfg=cfg), cfg)
        assert int(np.asarray(next_state.game.planets.owner)[0]) == expected_owner
        assert float(np.asarray(next_state.game.planets.ships)[0]) == expected_ships


def test_terminal_rewards_match_reference_for_elimination_max_steps_ties_and_fleets():
    cfg = _cfg()
    reward_cfg = _reward_cfg()
    reward_cfg.early_terminal_reward_shaping_enabled = False
    cases = [
        ([[0, 0, 80, 80, 3, 50, 1], [1, 1, 20, 20, 3, 30, 1]], [], 497, 0, 1.0),
        ([[0, 0, 80, 80, 3, 50, 1]], [], 0, 0, 1.0),
        ([[0, -1, 80, 80, 3, 50, 1]], [], 0, 0, -1.0),
        ([[0, 0, 80, 80, 3, 30, 1], [1, 1, 20, 20, 3, 30, 1]], [], 497, 0, 1.0),
        ([[0, 0, 80, 80, 3, 30, 1], [1, 1, 20, 20, 3, 30, 1]], [], 497, 1, 1.0),
        (
            [[0, 0, 80, 80, 3, 10, 1], [1, 1, 20, 20, 3, 30, 1]],
            [[0, 0, 50, 30, 0, 0, 50]],
            497,
            0,
            1.0,
        ),
        (
            [[0, 0, 80, 80, 3, 10, 1], [1, 1, 20, 20, 3, 30, 1]],
            [[0, 0, 50, 30, 0, 0, 50]],
            497,
            1,
            -1.0,
        ),
    ]
    for planets, fleets, step_index, learner_player, expected_reward in cases:
        next_state, result = _advance(
            _state(
                planets,
                fleets,
                cfg=cfg,
                step_index=step_index,
                learner_player=learner_player,
            ),
            cfg,
            reward_cfg,
        )
        assert bool(np.asarray(result.done))
        assert float(np.asarray(result.terminal_reward)) == expected_reward
        assert int(np.asarray(next_state.game.step)) == step_index + 1


def test_elimination_does_not_end_game_while_player_has_fleet():
    cfg = _cfg()
    _next_state, result = _advance(
        _state(
            [[0, 0, 80, 80, 3, 50, 1]],
            [[0, 1, 30, 30, 0.0, 99, 10]],
            cfg=cfg,
        ),
        cfg,
    )

    assert not bool(np.asarray(result.done))
    assert float(np.asarray(result.terminal_reward)) == 0.0


def test_four_player_terminal_reward_uses_all_players():
    cfg = _cfg(player_count=4)
    reward_cfg = _reward_cfg()
    reward_cfg.early_terminal_reward_shaping_enabled = False
    loser_state, loser_result = _advance(
        _state([[0, 2, 80, 80, 3, 40, 1]], cfg=cfg, learner_player=0),
        cfg,
        reward_cfg,
    )
    winner_state, winner_result = _advance(
        loser_state._replace(
            learner_player=jnp.array(2, dtype=jnp.int32),
            game=loser_state.game._replace(player=jnp.array(2, dtype=jnp.int32)),
        ),
        cfg,
        reward_cfg,
    )

    assert bool(np.asarray(loser_result.done))
    assert float(np.asarray(loser_result.terminal_reward)) == -1.0
    assert bool(np.asarray(winner_result.done))
    assert float(np.asarray(winner_result.terminal_reward)) == 1.0


def test_four_player_step_processes_all_player_action_lists_before_production():
    cfg = _cfg(player_count=4, max_fleets=24)
    state = _state(
        [
            [0, 0, 80, 80, 3, 20, 1],
            [1, 1, 20, 80, 3, 20, 1],
            [2, 2, 20, 20, 3, 20, 1],
            [3, 3, 80, 20, 3, 20, 1],
        ],
        cfg=cfg,
        learner_player=2,
    )
    actions = _multi_actions(
        cfg,
        {
            0: [(0, 0.0, 4)],
            1: [(1, math.pi, 5)],
            2: [(2, math.pi / 2, 6)],
            3: [(3, -math.pi / 2, 7)],
        },
    )

    next_state, result = step_multi_player(state, actions, cfg, _reward_cfg())

    np.testing.assert_allclose(
        np.asarray(next_state.game.planets.ships[:4]),
        np.array([17.0, 16.0, 15.0, 14.0]),
    )
    assert int(np.asarray(next_state.game.fleets.active).sum()) == 4
    assert set(np.asarray(next_state.game.fleets.owner[:4]).tolist()) == {0, 1, 2, 3}
    assert result.batch.planet_features.shape[0] == MAX_PLANETS


def test_four_player_step_rejects_actions_from_planets_not_owned_by_that_player():
    cfg = _cfg(player_count=4, max_fleets=24)
    state = _state(
        [
            [0, 0, 80, 80, 3, 20, 1],
            [1, 1, 20, 80, 3, 20, 1],
            [2, 2, 20, 20, 3, 20, 1],
            [3, 3, 80, 20, 3, 20, 1],
        ],
        cfg=cfg,
    )
    actions = _multi_actions(
        cfg,
        {
            0: [(1, 0.0, 4)],
            1: [(2, 0.0, 5)],
            2: [(3, 0.0, 6)],
            3: [(0, 0.0, 7)],
        },
    )

    next_state, _ = step_multi_player(state, actions, cfg, _reward_cfg())

    np.testing.assert_allclose(
        np.asarray(next_state.game.planets.ships[:4]),
        np.array([21.0, 21.0, 21.0, 21.0]),
    )
    assert int(np.asarray(next_state.game.fleets.active).sum()) == 0


def test_non_pool_comet_spawn_keeps_initial_planets_synced_without_baked_wave():
    cfg = _cfg(env_parity_mode="kaggle")
    state, _ = reset(jax.random.PRNGKey(0), cfg)
    baseline_active = int(np.asarray(state.game.planets.active).sum())
    # Spawn fires when (step + 1) == 50, i.e. on the step that ends at game.step == 50.
    for _ in range(50):
        state, _ = _advance(state, cfg)
    active = np.asarray(state.game.planets.active)
    init_active = np.asarray(state.game.initial_planets.active)
    assert int(np.asarray(state.game.comets.group_active).sum()) == 0
    assert int(active.sum()) == int(init_active.sum())
    assert int(active.sum()) == baseline_active


def test_four_player_step_allows_simultaneous_four_way_combat_from_actions():
    cfg = _cfg(player_count=4, max_fleets=24)
    state = _state(
        [
            [0, 0, 76.0, 80.0, 1.0, 60, 0],
            [1, 1, 76.0, 80.0, 1.0, 60, 0],
            [2, 2, 76.0, 80.0, 1.0, 60, 0],
            [3, 3, 76.0, 80.0, 1.0, 60, 0],
            [4, -1, 80.0, 80.0, 5.0, 10, 0],
        ],
        cfg=cfg,
    )
    actions = _multi_actions(
        cfg,
        {
            0: [(0, 0.0, 41)],
            1: [(1, 0.0, 20), (1, 0.0, 20)],
            2: [(2, 0.0, 42)],
            3: [(3, 0.0, 5)],
        },
    )

    next_state, _ = step_multi_player(state, actions, cfg, _reward_cfg())

    assert int(np.asarray(next_state.game.planets.owner)[4]) == -1
    assert float(np.asarray(next_state.game.planets.ships)[4]) == 9.0
    assert int(np.asarray(next_state.game.fleets.active).sum()) == 0
    np.testing.assert_allclose(
        np.asarray(next_state.game.planets.ships[:4]),
        np.array([19.0, 20.0, 18.0, 55.0]),
    )


@pytest.fixture(scope="module")
def tiny_map_pool():
    from src.jax.map_pool.bake import bake_one_entry, stack_entries
    from src.jax.map_pool.load import map_pool_constants_from_numpy

    entries = [bake_one_entry(seed) for seed in (0, 1, 2)]
    stacked = stack_entries(entries)
    return map_pool_constants_from_numpy(stacked)


def test_pool_reset_produces_valid_planet_group_count(tiny_map_pool):
    cfg = _cfg()
    key = jax.random.PRNGKey(0)
    state, batch = reset_with_pool(
        key, cfg, tiny_map_pool, jnp.array(0, dtype=jnp.int32)
    )
    active = int(np.asarray(state.game.planets.active).sum())
    group_count = active // 4
    assert 5 <= group_count <= 10
    assert batch.planet_mask.ndim >= 1


def test_pool_comet_spawn_keeps_initial_planets_synced_after_fifty_steps():
    from src.jax.map_pool.load import load_map_pool

    cfg = _cfg()
    pool = load_map_pool("data/jax_map_pool/default_v1.npz")
    state, _ = reset_with_pool(
        jax.random.PRNGKey(0), cfg, pool, jnp.array(0, dtype=jnp.int32)
    )
    baseline_active = int(np.asarray(state.game.planets.active).sum())
    for _ in range(50):
        state, _ = _advance(state, cfg)
    active = np.asarray(state.game.planets.active)
    init_active = np.asarray(state.game.initial_planets.active)
    assert int(state.game.comets.group_active.sum()) >= 1
    assert int(active.sum()) == int(init_active.sum())
    assert int(active.sum()) > baseline_active


def test_pool_reset_mid_game_state_remains_valid_after_hundred_steps():
    from src.jax.map_pool.load import load_map_pool

    cfg = _cfg()
    pool = load_map_pool("data/jax_map_pool/default_v1.npz")
    state, _ = reset_with_pool(
        jax.random.PRNGKey(7), cfg, pool, jnp.array(3, dtype=jnp.int32)
    )
    for _ in range(100):
        state, _ = _advance(state, cfg)
    active = np.asarray(state.game.planets.active)
    assert int(active.sum()) >= 20
    assert int(np.asarray(state.game.step)) == 100


def test_eval_paths_do_not_import_map_pool_loader():
    repo = Path(__file__).resolve().parents[1]
    eval_modules = [
        repo / "src/jax/submission_runtime.py",
        repo / "src/cli/eval.py",
        repo / "src/artifacts/tournament/eval.py",
    ]
    forbidden = ("load_map_pool", "map_pool.load")
    for path in eval_modules:
        assert path.is_file(), f"missing eval module: {path}"
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path.name} must not reference {needle!r}"
