"""Unit tests for normalized ship differential terminal reward (JAX canonical)."""

from __future__ import annotations

import numpy as np
import pytest


def _normalized_ship_differential(scores: list[float], learner_index: int) -> float:
    """Reference formula mirrored in src/jax/env._terminal."""

    learner = float(scores[learner_index])
    max_other = max(
        (float(score) for i, score in enumerate(scores) if i != learner_index),
        default=0.0,
    )
    denom = learner + max_other
    if denom <= 0.0:
        return 0.0
    return (learner - max_other) / denom


def test_normalized_ship_differential_elimination_and_tie() -> None:
    assert _normalized_ship_differential([100.0, 0.0], 0) == pytest.approx(1.0)
    assert _normalized_ship_differential([0.0, 100.0], 0) == pytest.approx(-1.0)
    assert _normalized_ship_differential([30.0, 30.0], 0) == pytest.approx(0.0)
    assert _normalized_ship_differential([0.0, 0.0], 0) == pytest.approx(0.0)


def test_normalized_ship_differential_graded_outcomes() -> None:
    assert _normalized_ship_differential([80.0, 20.0], 0) == pytest.approx(0.6)
    assert _normalized_ship_differential([55.0, 45.0], 0) == pytest.approx(0.1)
    assert _normalized_ship_differential([45.0, 55.0], 0) == pytest.approx(-0.1)


def test_normalized_ship_differential_four_player_max_other() -> None:
    reward = _normalized_ship_differential([40.0, 20.0, 20.0, 20.0], 0)
    assert reward == pytest.approx(1.0 / 3.0)


@pytest.mark.jax
def test_jax_terminal_normalized_ship_differential() -> None:
    from src.jax.env import _terminal
    from tests.test_jax_env_parity import _cfg, _reward_cfg, _state

    cfg = _cfg()
    reward_cfg = _reward_cfg()
    reward_cfg.terminal_reward_mode = "normalized_ship_differential"
    reward_cfg.early_terminal_reward_shaping_enabled = False
    planets = [[0, 0, 80, 80, 3, 50, 1], [1, 1, 20, 20, 3, 30, 1]]
    state = _state(planets, [], cfg=cfg, step_index=498, learner_player=0)
    done, reward, *_mid, ship_diff, _survival = _terminal(
        state.game, state.learner_player, cfg, reward_cfg
    )
    expected = _normalized_ship_differential([50.0, 30.0], 0)
    assert bool(np.asarray(done))
    assert float(np.asarray(reward)) == pytest.approx(expected)
    assert float(np.asarray(ship_diff)) == pytest.approx(expected)


@pytest.mark.jax
def test_jax_terminal_tie_returns_zero_not_binary_win() -> None:
    from src.jax.env import _terminal
    from tests.test_jax_env_parity import _cfg, _reward_cfg, _state

    cfg = _cfg()
    reward_cfg = _reward_cfg()
    reward_cfg.terminal_reward_mode = "normalized_ship_differential"
    reward_cfg.early_terminal_reward_shaping_enabled = False
    planets = [[0, 0, 80, 80, 3, 30, 1], [1, 1, 20, 20, 3, 30, 1]]
    state = _state(planets, [], cfg=cfg, step_index=497, learner_player=0)
    _done, reward, _rank, _placement, is_first, _score_share, _ship_diff, _survival = (
        _terminal(state.game, state.learner_player, cfg, reward_cfg)
    )
    assert float(np.asarray(is_first)) == pytest.approx(1.0)
    assert float(np.asarray(reward)) == pytest.approx(0.0)
