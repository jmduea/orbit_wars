from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.artifacts.replay import maybe_write_jax_checkpoint_replay
from src.artifacts.tournament.runner import MatchOutcome, build_checkpoint_agent
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


def test_maybe_write_jax_checkpoint_replay_unpacks_run_match_three_tuple(
    tmp_path: Path,
) -> None:
    """#160: caller must unpack run_match as (outcome, env, timing)."""
    cfg = TrainConfig()
    cfg.artifacts.replay.enabled = True
    cfg.artifacts.replay.max_steps = 3
    cfg.training.total_updates = 10
    update = 10

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "jax_ckpt_000010.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    log_path = run_dir / "metrics.jsonl"
    log_path.write_text("{}\n", encoding="utf-8")

    fake_env = MagicMock()
    fake_env.render.return_value = "<html/>"
    outcome = MatchOutcome(
        match_id="replay",
        format_name="2p_random",
        seed=0,
        agent_ids=("seat_0", "seat_1"),
        rewards={"seat_0": 1.0, "seat_1": -1.0},
        results={"seat_0": "win", "seat_1": "loss"},
    )
    learner = SimpleNamespace(act_fn=lambda _obs: [])

    with (
        patch(
            "src.artifacts.replay.run_match",
            return_value=(outcome, fake_env, {"steps": 1}),
        ) as run_match_mock,
        patch(
            "src.artifacts.replay.build_checkpoint_agent",
            return_value=learner,
        ),
        patch(
            "src.artifacts.replay.build_baseline_agent",
            return_value=SimpleNamespace(act_fn=lambda _obs: []),
        ),
    ):
        metadata_path = maybe_write_jax_checkpoint_replay(
            cfg,
            update=update,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            output_dir=tmp_path / "replays",
        )

    assert metadata_path is not None
    assert metadata_path.is_file()
    assert run_match_mock.call_count >= 1


def test_maybe_write_jax_checkpoint_replay_requires_run_match_three_tuple(
    tmp_path: Path,
) -> None:
    cfg = TrainConfig()
    cfg.artifacts.replay.enabled = True
    cfg.training.total_updates = 5
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "jax_ckpt_000005.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    log_path = run_dir / "metrics.jsonl"
    log_path.write_text("{}\n", encoding="utf-8")

    with (
        patch(
            "src.artifacts.replay.run_match",
            return_value=(MatchOutcome(
                match_id="m",
                format_name="2p_random",
                seed=0,
                agent_ids=("seat_0", "seat_1"),
                rewards={"seat_0": 0.0, "seat_1": 0.0},
                results={"seat_0": "tie", "seat_1": "tie"},
            ),),
        ),
        patch(
            "src.artifacts.replay.build_checkpoint_agent",
            return_value=SimpleNamespace(act_fn=lambda _obs: []),
        ),
        patch(
            "src.artifacts.replay.build_baseline_agent",
            return_value=SimpleNamespace(act_fn=lambda _obs: []),
        ),
    ):
        with pytest.raises(ValueError, match="not enough values to unpack"):
            maybe_write_jax_checkpoint_replay(
                cfg,
                update=5,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
                output_dir=tmp_path / "replays",
            )
