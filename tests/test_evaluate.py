from __future__ import annotations

import sys
import types
from typing import Any

import evaluate


class RecordingAgent:
    def __init__(self, action: list[list[float | int]] | None = None) -> None:
        self.action = action or []
        self.observations: list[Any] = []

    def act(self, observation: Any) -> list[list[float | int]]:
        self.observations.append(observation)
        return self.action


class FakeKaggleEnv:
    def __init__(self) -> None:
        self.reset_num_agents: int | None = None
        self.actions: list[Any] = []
        self.step_index = 0

    def reset(self, num_agents: int) -> None:
        self.reset_num_agents = num_agents

    def step(self, action: Any) -> list[dict[str, Any]]:
        self.actions.append(action)
        assert self.reset_num_agents is not None
        status = "ACTIVE" if self.step_index == 0 else "DONE"
        rewards = (
            [0.0] * self.reset_num_agents
            if self.step_index == 0
            else [0.0, 1.0, -1.0, -1.0]
        )
        states = [
            {
                "observation": {
                    "player": player,
                    "step": self.step_index,
                    "planets": [],
                    "fleets": [],
                },
                "status": status,
                "reward": rewards[player],
            }
            for player in range(self.reset_num_agents)
        ]
        self.step_index += 1
        return states


def install_fake_kaggle(monkeypatch: Any, env: FakeKaggleEnv) -> None:
    kaggle_module = types.ModuleType("kaggle_environments")
    kaggle_module.make = lambda *_args, **_kwargs: env
    monkeypatch.setitem(sys.modules, "kaggle_environments", kaggle_module)


def test_parse_formats_accepts_player_counts_and_format_labels() -> None:
    assert evaluate.parse_formats("2,4") == [2, 4]
    assert evaluate.parse_formats("2p,4p,2") == [2, 4]


def test_four_player_evaluation_resets_four_agents_and_uses_three_opponent_slots(
    monkeypatch: Any,
) -> None:
    env = FakeKaggleEnv()
    install_fake_kaggle(monkeypatch, env)
    learner = RecordingAgent([[99, 0.5, 10]])
    opponents = [RecordingAgent([[idx, 0.25, 5]]) for idx in range(3)]

    outcome = evaluate.play_one_game(
        learner,
        opponents,
        seed=7,
        player_count=4,
        learner_seat=1,
    )

    assert env.reset_num_agents == 4
    assert env.actions[0] == [[], [], [], []]
    assert env.actions[1] == [
        [[0, 0.25, 5]],
        [[99, 0.5, 10]],
        [[1, 0.25, 5]],
        [[2, 0.25, 5]],
    ]
    assert [opponent.observations[0]["player"] for opponent in opponents] == [0, 2, 3]
    assert learner.observations[0]["player"] == 1
    assert outcome.first_place is True
    assert outcome.placement == 1.0


def test_four_player_evaluation_rejects_missing_opponent_slots() -> None:
    learner = RecordingAgent()
    opponents = [RecordingAgent(), RecordingAgent()]

    try:
        evaluate.play_one_game(
            learner, opponents, seed=1, player_count=4, learner_seat=0
        )
    except ValueError as exc:
        assert "Expected 3 opponent slot(s)" in str(exc)
    else:
        raise AssertionError("4p evaluation should require three opponent slots")


def test_aggregate_format_reports_four_player_metrics_per_seat() -> None:
    results = [
        evaluate.GameResult("4p", 4, "random", 1, 1, 0, 3, 1.0, "win", 1.0, True, 10),
        evaluate.GameResult(
            "4p", 4, "random", 1, 1, 1, 3, -1.0, "loss", 3.0, False, 12
        ),
    ]

    metrics = evaluate.aggregate_format(results, 4)

    assert metrics.first_place_rate_4p == 0.5
    assert metrics.average_placement_4p == 2.0
    assert metrics.per_seat["0"]["first_place_rate_4p"] == 1.0
    assert metrics.per_seat["1"]["average_placement_4p"] == 3.0


def test_split_result_metrics_separates_wins_and_losses() -> None:
    win = evaluate.GameResult("2p", 2, "random", 1, 1, 0, 1, 1.0, "win", 1.0, True, 10)
    loss = evaluate.GameResult("2p", 2, "random", 2, 2, 0, 1, -1.0, "loss", 2.0, False, 11)
    win.non_noop_actions_per_step = 2.0
    loss.non_noop_actions_per_step = 6.0
    metrics = evaluate.split_result_metrics([win, loss])
    assert metrics["win"]["non_noop_actions_per_step"] == 2.0
    assert metrics["loss"]["non_noop_actions_per_step"] == 6.0


def test_two_player_evaluation_resets_two_agents_and_uses_one_opponent_slot(
    monkeypatch: Any,
) -> None:
    env = FakeKaggleEnv()
    install_fake_kaggle(monkeypatch, env)
    learner = RecordingAgent([[7, 1.25, 3]])
    opponent = RecordingAgent([[8, 0.75, 2]])

    outcome = evaluate.play_one_game(
        learner,
        [opponent],
        seed=9,
        player_count=2,
        learner_seat=0,
    )

    assert env.reset_num_agents == 2
    assert env.actions[0] == [[], []]
    assert env.actions[1] == [[[7, 1.25, 3]], [[8, 0.75, 2]]]
    assert learner.observations[0]["player"] == 0
    assert opponent.observations[0]["player"] == 1
    assert outcome.placement == 2.0


def test_load_checkpoint_if_available_reads_jax_params(tmp_path: Any) -> None:
    import pickle

    cfg = evaluate.load_train_config(evaluate.default_train_config_path())
    ckpt_path = tmp_path / "jax_ckpt.pkl"
    payload = {
        "params": {"dense": {"kernel": np.zeros((1, 1), dtype=np.float32)}},
        "feature_metadata": {
            "self_dim": evaluate.self_feature_dim(cfg.env),
            "candidate_dim": evaluate.candidate_feature_dim(cfg.env),
            "global_dim": evaluate.global_feature_dim(cfg.env),
            "candidate_count": cfg.env.candidate_count,
            "ship_bucket_count": cfg.env.ship_bucket_count,
        },
    }
    with ckpt_path.open("wb") as f:
        pickle.dump(payload, f)

    params = evaluate.load_checkpoint_if_available(None, None, str(ckpt_path), "auto", cfg)

    assert isinstance(params, dict)
    assert "dense" in params


def test_self_play_opponent_act_returns_valid_move_or_pass() -> None:
    from src.opponents import SelfPlayOpponent

    cfg = evaluate.load_train_config(evaluate.default_train_config_path())
    opponent = SelfPlayOpponent(cfg, deterministic=True)
    policy = evaluate.build_policy(cfg)
    import jax
    import jax.numpy as jnp
    dummy = (
        jnp.zeros((1, evaluate.self_feature_dim(cfg.env)), dtype=jnp.float32),
        jnp.zeros((1, cfg.env.candidate_count, evaluate.candidate_feature_dim(cfg.env)), dtype=jnp.float32),
        jnp.zeros((1, evaluate.global_feature_dim(cfg.env)), dtype=jnp.float32),
        jnp.ones((1, cfg.env.candidate_count), dtype=bool),
    )
    params = policy.init(jax.random.PRNGKey(0), *dummy)["params"]
    opponent.sync_from(params, None)

    observation = {"player": 0, "step": 0, "planets": [], "fleets": []}
    moves = opponent.act(observation)
    assert isinstance(moves, list)
