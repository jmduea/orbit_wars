from __future__ import annotations

import sys
from pathlib import Path

from src.config.schema import TrainConfig
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
