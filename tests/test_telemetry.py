from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.checkpoint_retention import prune_checkpoints
from src.conf_schema import TrainConfig
from src.config import compose_hydra_train_config, train_config_from_omegaconf
from src.metric_registry import filter_event_record, filter_update_record
from src.telemetry import TelemetryLogger


class _FakeWandbRun:
    def __init__(self) -> None:
        self.config = {}

    def finish(self) -> None:
        pass


class _FakeWandb:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.init_kwargs: dict[str, object] = {}

    def init(self, **kwargs: object) -> _FakeWandbRun:
        self.init_kwargs = kwargs
        return _FakeWandbRun()

    def log(self, record: dict[str, object], step: int | None = None) -> None:
        self.logs.append((record, step))


def test_hydra_config_supports_metric_group_overrides():
    cfg = compose_hydra_train_config(
        [
            "training.total_updates=1",
            "telemetry.metric_groups.trajectory_shield_debug=true",
            "telemetry.metric_groups.losses=false",
        ]
    )

    assert cfg.telemetry.metric_groups.trajectory_shield_debug is True
    assert cfg.telemetry.metric_groups.losses is False
    assert cfg.telemetry.metric_groups.core_progress is True


def test_wandb_logging_uses_requested_steps_when_monotonic(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True

    logger = TelemetryLogger(cfg)
    logger.log({"update": 1, "overall_win_rate": 0.0}, step=1)
    logger.log({"update": 2, "overall_win_rate": 0.5}, step=2)

    assert [step for _record, step in fake_wandb.logs] == [1, 2]


def test_wandb_logging_advances_delayed_steps(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True

    logger = TelemetryLogger(cfg)
    logger.log({"update": 100, "overall_win_rate": 0.0}, step=100)
    logger.log({"update": 101, "overall_win_rate": 0.5}, step=101)
    logger.log({"event": "checkpoint_result", "update": 100}, step=100)
    logger.log({"update": 102, "overall_win_rate": 0.75}, step=102)

    assert [step for _record, step in fake_wandb.logs] == [100, 101, 102, 102]
    assert fake_wandb.logs[-2][0]["update"] == 100


def test_wandb_local_paths_are_configured_before_init(tmp_path: Path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.delenv("WANDB_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("WANDB_DATA_DIR", raising=False)
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True
    wandb_dir = tmp_path / "run" / "cache" / "wandb"
    artifact_dir = tmp_path / "cache" / "wandb-artifacts"
    data_dir = tmp_path / "cache" / "wandb-data"

    TelemetryLogger(
        cfg,
        {
            "wandb_dir": str(wandb_dir),
            "wandb_artifact_dir": str(artifact_dir),
            "wandb_data_dir": str(data_dir),
        },
    )

    assert fake_wandb.init_kwargs["dir"] == str(wandb_dir)
    assert wandb_dir.exists()
    assert artifact_dir.exists()
    assert data_dir.exists()


def test_invalid_plateau_metric_is_rejected():
    cfg = OmegaConf.structured(TrainConfig)
    cfg.training.plateau_metric = "average_episode_reward"

    with pytest.raises(
        ValueError, match="registered canonical scalar telemetry metric"
    ):
        train_config_from_omegaconf(cfg)


def test_non_scalar_plateau_metric_is_rejected():
    cfg = OmegaConf.structured(TrainConfig)
    cfg.training.plateau_metric = "opponent_composition"

    with pytest.raises(
        ValueError, match="registered canonical scalar telemetry metric"
    ):
        train_config_from_omegaconf(cfg)


def test_string_plateau_metric_is_rejected():
    cfg = OmegaConf.structured(TrainConfig)
    cfg.training.plateau_metric = "curriculum_stage_id"

    with pytest.raises(
        ValueError, match="registered canonical scalar telemetry metric"
    ):
        train_config_from_omegaconf(cfg)


def test_string_retention_metric_is_rejected():
    cfg = OmegaConf.structured(TrainConfig)
    cfg.artifacts.checkpoint_retention.best_metric_name = "seed_scheduler_policy"

    with pytest.raises(
        ValueError, match="registered canonical scalar telemetry metric"
    ):
        train_config_from_omegaconf(cfg)


def test_filter_update_record_preserves_configured_retention_metric():
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.losses = False
    cfg.telemetry.metric_groups.opponent_composition = False
    cfg.artifacts.checkpoint_retention.best_metric_name = "total_loss"

    record = {
        "update": 3,
        "total_env_steps": 300,
        "completed_episodes": 7,
        "samples": 128,
        "overall_win_rate": 0.5,
        "win_rate_2p": 0.5,
        "first_place_rate_4p": 0.0,
        "episode_reward_mean": 0.25,
        "env_steps_per_sec": 900.0,
        "total_loss": 1.75,
        "opponent_slots_total": 8.0,
    }

    filtered = filter_update_record(record, cfg)

    assert filtered["total_loss"] == 1.75
    assert "opponent_slots_total" not in filtered


def test_filter_event_record_respects_events_toggle_but_keeps_checkpoint_fields():
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.events = False

    record = {
        "event": "checkpoint_result",
        "update": 9,
        "checkpoint_status": "committed",
        "checkpoint_final": True,
        "checkpoint_reason": "final",
        "checkpoint_error": None,
        "metric": "overall_win_rate",
        "metric_value": 0.9,
    }

    filtered = filter_event_record(record, cfg)

    assert filtered == {
        "event": "checkpoint_result",
        "update": 9,
        "checkpoint_status": "committed",
        "checkpoint_final": True,
        "checkpoint_reason": "final",
        "checkpoint_error": None,
    }


def test_checkpoint_pruning_can_read_preserved_metric_from_filtered_jsonl(
    tmp_path: Path,
):
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.losses = False
    cfg.artifacts.checkpoint_retention.best_metric_name = "total_loss"
    cfg.artifacts.checkpoint_retention.best_metric_mode = "max"

    log_path = tmp_path / "metrics.jsonl"
    records = [
        {
            "update": 1,
            "total_env_steps": 100,
            "completed_episodes": 2,
            "samples": 64,
            "overall_win_rate": 0.25,
            "win_rate_2p": 0.25,
            "first_place_rate_4p": 0.0,
            "episode_reward_mean": 0.1,
            "env_steps_per_sec": 500.0,
            "total_loss": 1.0,
        },
        {
            "update": 2,
            "total_env_steps": 200,
            "completed_episodes": 4,
            "samples": 64,
            "overall_win_rate": 0.5,
            "win_rate_2p": 0.5,
            "first_place_rate_4p": 0.0,
            "episode_reward_mean": 0.3,
            "env_steps_per_sec": 550.0,
            "total_loss": 2.0,
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(filter_update_record(record, cfg)) for record in records)
        + "\n",
        encoding="utf-8",
    )
    for update in (1, 2):
        (tmp_path / f"jax_ckpt_{update:06d}.pkl").write_bytes(b"checkpoint")

    decision = prune_checkpoints(
        tmp_path,
        log_path=log_path,
        keep_last_n=0,
        keep_every_n_updates=0,
        keep_best_k_by_metric=1,
        best_metric_name=cfg.artifacts.checkpoint_retention.best_metric_name,
        best_metric_mode=cfg.artifacts.checkpoint_retention.best_metric_mode,
        min_update_for_pruning=0,
        dry_run_pruning=False,
        protected_paths=None,
    )

    deleted_names = {path.name for path in decision.deleted}
    assert "jax_ckpt_000001.pkl" in deleted_names
    assert (tmp_path / "jax_ckpt_000002.pkl").exists()
