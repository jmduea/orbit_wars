from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

from src.artifacts.tournament.runner import build_checkpoint_agent
from src.config import TrainConfig
from src.jax.submission_runtime import SubmissionReadyAgent


class _FakePolicy:
    pass


def test_jax_replay_actor_uses_submission_runtime_path(monkeypatch, tmp_path: Path) -> None:
    cfg = TrainConfig()
    cfg.task.candidate_count = 4
    cfg.task.ship_bucket_count = 2
    cfg.task.max_fleets = 8
    cfg.task.trajectory_shield_mode = "off"

    checkpoint_path = tmp_path / "jax_ckpt_000100.pkl"
    with checkpoint_path.open("wb") as file:
        pickle.dump({"params": {"fake": "params"}, "config": cfg}, file)

    def fake_act(_observation: object) -> list[list[float | int]]:
        return [[7, 1.0, 3]]

    fake_agent = SubmissionReadyAgent(
        act_fn=fake_act,
        reset_episode=lambda: None,
        warmup=lambda: None,
    )

    with (
        patch(
            "src.artifacts.tournament.runner.build_jax_policy",
            return_value=_FakePolicy(),
        ),
        patch(
            "src.artifacts.tournament.runner.build_submission_ready_agent",
            return_value=fake_agent,
        ),
        patch(
            "src.artifacts.tournament.runner.validate_checkpoint_config_compatibility"
        ),
    ):
        agent = build_checkpoint_agent(cfg, checkpoint_path)
        moves = agent.act_fn({"player": 0, "planets": []})

    assert moves == [[7, 1.0, 3]]
