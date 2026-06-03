"""Tests for training-time bracket hooks."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.artifacts.tournament.bracket.state import load_bracket_state
from src.config.schema import ArtifactsConfig, BracketTrainingConfig, OutputConfig, TrainConfig
from src.jax.train.bracket_training import bracket_training_tick


def _cfg(tmp_path: Path, *, budget: int = 500) -> TrainConfig:
    cfg = TrainConfig()
    cfg.output = OutputConfig(root=str(tmp_path), campaign="demo")
    cfg.artifacts = ArtifactsConfig(
        bracket_training=BracketTrainingConfig(
            enabled=True,
            qualifier_max_env_steps=budget,
            qualifier_eval_interval_updates=10,
        )
    )
    return cfg


def test_weak_config_at_budget_without_clear(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, budget=1000)
    tick = bracket_training_tick(
        cfg,
        update=50,
        total_env_steps=1000,
        checkpoint_path=None,
        queue_dir=tmp_path / "queue",
        output_root=tmp_path,
    )
    assert tick.weak_config is True
    state = load_bracket_state(
        tmp_path / "campaigns" / "demo" / "bracket" / "state.json"
    )
    assert state.phase == "weak_config"


def test_below_budget_not_weak_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, budget=1_000_000)
    tick = bracket_training_tick(
        cfg,
        update=1,
        total_env_steps=500,
        checkpoint_path=None,
        queue_dir=tmp_path / "queue",
        output_root=tmp_path,
    )
    assert tick.weak_config is False


def test_numbered_checkpoint_path_matches_pipeline(tmp_path: Path) -> None:
    """Async checkpoint jobs use jax_ckpt_{update:06d}.pkl, not jax_ckpt_u{update}.pkl."""
    from src.artifacts.pipeline import CheckpointJob

    job = CheckpointJob(update=50, run_dir=tmp_path, build_payload=lambda: {})
    assert job.numbered_path == tmp_path / "jax_ckpt_000050.pkl"
    assert not (tmp_path / "jax_ckpt_u50.pkl").exists()


def test_interval_queues_qualifier_eval(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ckpt = tmp_path / "ckpt.pkl"
    ckpt.write_bytes(pickle.dumps({"update": 10}))
    tick = bracket_training_tick(
        cfg,
        update=10,
        total_env_steps=100,
        checkpoint_path=ckpt,
        queue_dir=tmp_path / "queue",
        output_root=tmp_path,
    )
    assert tick.qualifier_eval_queued is True
    jobs = list((tmp_path / "queue").glob("*.json"))
    assert jobs
