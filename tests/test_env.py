from __future__ import annotations

from typing import Any

from src.config import TrainConfig
from src.env import OrbitWarsEnv, terminal_reward


def _obs(player: int, step: int = 0) -> dict[str, Any]:
    return {"player": player, "step": step, "planets": [], "fleets": []}


class RecordingOpponent:
    def __init__(self, action: list[list[float | int]]) -> None:
        self.action = action
        self.observations: list[Any] = []

    def act(self, observation: Any) -> list[list[float | int]]:
        self.observations.append(observation)
        return self.action


class SamplingOpponent:
    def __init__(self) -> None:
        self.sampled: list[RecordingOpponent] = []

    def sample_opponent(self) -> RecordingOpponent:
        opponent = RecordingOpponent([[len(self.sampled), 0.25, 10]])
        self.sampled.append(opponent)
        return opponent


class FakeEnv:
    def __init__(self) -> None:
        self.reset_num_agents: int | None = None
        self.actions: list[Any] = []
        self.step_index = 0

    def reset(self, num_agents: int) -> None:
        self.reset_num_agents = num_agents

    def step(self, action: Any) -> list[dict[str, Any]]:
        self.actions.append(action)
        states = []
        for player in range(self.reset_num_agents or 0):
            states.append(
                {
                    "observation": _obs(player, step=self.step_index),
                    "status": "ACTIVE",
                    "reward": 0.0,
                }
            )
        self.step_index += 1
        return states


class FakeMake:
    def __init__(self) -> None:
        self.env = FakeEnv()

    def __call__(self, *_args: Any, **_kwargs: Any) -> FakeEnv:
        return self.env


def test_reset_uses_configured_player_count_and_side_rotation() -> None:
    cfg = TrainConfig()
    cfg.env.player_count = 4
    make = FakeMake()

    env = OrbitWarsEnv(cfg, RecordingOpponent([]), make_fn=make, env_index=2)
    env.reset(seed=123)

    assert make.env.reset_num_agents == 4
    assert make.env.actions[0] == [[], [], [], []]
    assert env.learner_player == 2

    env.reset(seed=124)

    assert env.learner_player == 3


def test_step_builds_joint_action_and_tracks_opponent_observations() -> None:
    cfg = TrainConfig()
    cfg.env.player_count = 4
    make = FakeMake()
    opponent = SamplingOpponent()

    env = OrbitWarsEnv(cfg, opponent, make_fn=make, env_index=1)
    env.reset()
    result = env.step([[99, 0.5, 12]])

    assert len(opponent.sampled) == 3
    assert make.env.actions[-1] == [
        [[0, 0.25, 10]],
        [[99, 0.5, 12]],
        [[1, 0.25, 10]],
        [[2, 0.25, 10]],
    ]
    assert set(env.last_opponent_obs) == {0, 2, 3}
    assert [sample.observations[0]["player"] for sample in opponent.sampled] == [
        0,
        2,
        3,
    ]
    assert result.info["opponent_players"] == [0, 2, 3]
    assert result.info["opponent_statuses"] == {0: "ACTIVE", 2: "ACTIVE", 3: "ACTIVE"}
    assert result.info["opponent_status"] == "ACTIVE"


def test_terminal_reward_treats_any_positive_opponent_reward_as_tie() -> None:
    assert terminal_reward({"reward": 1.0}, [{"reward": -1.0}, {"reward": 0.0}]) == 1.0
    assert terminal_reward({"reward": 1.0}, [{"reward": -1.0}, {"reward": 1.0}]) == 0.0
