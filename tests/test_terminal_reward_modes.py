"""Unit tests for normalized ship differential terminal reward."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import RewardConfig, TaskConfig
from src.game.env import (
    normalized_ship_differential_reward,
    terminal_reward_from_scores,
)
from src.jax.env import _terminal


def test_normalized_ship_differential_elimination_and_tie() -> None:
    assert normalized_ship_differential_reward([100.0, 0.0], 0) == pytest.approx(1.0)
    assert normalized_ship_differential_reward([0.0, 100.0], 0) == pytest.approx(-1.0)
    assert normalized_ship_differential_reward([30.0, 30.0], 0) == pytest.approx(0.0)
    assert normalized_ship_differential_reward([0.0, 0.0], 0) == pytest.approx(0.0)


def test_normalized_ship_differential_graded_outcomes() -> None:
    assert normalized_ship_differential_reward([80.0, 20.0], 0) == pytest.approx(0.6)
    assert normalized_ship_differential_reward([55.0, 45.0], 0) == pytest.approx(0.1)
    assert normalized_ship_differential_reward([45.0, 55.0], 0) == pytest.approx(-0.1)


def test_normalized_ship_differential_four_player_max_other() -> None:
    reward = normalized_ship_differential_reward([40.0, 20.0, 20.0, 20.0], 0)
    assert reward == pytest.approx(1.0 / 3.0)


def test_terminal_reward_from_scores_mode_branch() -> None:
    task = TaskConfig(player_count=2)
    reward_cfg = RewardConfig(terminal_reward_mode="normalized_ship_differential")
    diagnostics = terminal_reward_from_scores(
        [80.0, 20.0], task, reward_cfg, learner_index=0
    )
    assert diagnostics["terminal_reward_unscaled"] == pytest.approx(0.6)
    assert diagnostics["terminal_ship_differential"] == pytest.approx(0.6)
    assert diagnostics["terminal_is_first"] == pytest.approx(1.0)


@pytest.mark.jax
def test_jax_terminal_matches_python_helper() -> None:
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
    expected = normalized_ship_differential_reward([50.0, 30.0], 0)
    assert bool(np.asarray(done))
    assert float(np.asarray(reward)) == pytest.approx(expected)
    assert float(np.asarray(ship_diff)) == pytest.approx(expected)


@pytest.mark.jax
def test_jax_terminal_tie_returns_zero_not_binary_win() -> None:
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
