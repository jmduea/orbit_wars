from __future__ import annotations

import json
import re
from pathlib import Path

from src import run_paths
from src.config import TrainConfig
from src.run_paths import compose_run_name, resolve_run_paths, write_run_manifests


RUN_NAME_TIMESTAMP_RE = r"\d{8}T\d{6}Z"


def test_fallback_run_context_uses_campaign_envelope(tmp_path: Path) -> None:
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.output.campaign = "capacity"
    cfg.output.run_id = "run-001"

    resolved, context = resolve_run_paths(cfg)

    assert (
        context.run_dir
        == tmp_path / "outputs" / "campaigns" / "capacity" / "runs" / "run-001"
    )
    assert context.checkpoints_dir == context.run_dir / "checkpoints"
    assert context.queue_dir == context.run_dir / "queue" / "optional_jobs"
    assert context.evaluations_dir == context.run_dir / "evaluations"
    assert resolved.artifacts.save_dir == str(context.checkpoints_dir)


def test_write_run_manifests_records_required_paths(tmp_path: Path) -> None:
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.output.campaign = "capacity"
    cfg.output.run_id = "run-001"
    cfg.telemetry.wandb.project = "orbit_wars"
    cfg, context = resolve_run_paths(cfg)

    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-001"
    assert manifest["campaign"] == "capacity"
    assert manifest["model_compatibility_family"] == cfg.model.architecture
    assert manifest["paths"]["checkpoints_dir"] == str(context.checkpoints_dir)
    assert context.campaign_manifest_path.exists()
    assert (context.indexes_dir / "runs.jsonl").exists()


def test_hydra_runtime_output_dir_is_the_run_envelope(
    tmp_path: Path, monkeypatch
) -> None:
    hydra_run_dir = (
        tmp_path / "outputs" / "campaigns" / "scratch" / "runs" / "hydra-run"
    )
    monkeypatch.setattr(run_paths, "_hydra_runtime_output_dir", lambda: hydra_run_dir)
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")

    resolved, context = resolve_run_paths(cfg)

    assert context.run_dir == hydra_run_dir
    assert context.run_id == "hydra-run"
    assert context.checkpoints_dir == hydra_run_dir / "checkpoints"
    assert context.queue_dir == hydra_run_dir / "queue" / "optional_jobs"
    assert context.evaluations_dir == hydra_run_dir / "evaluations"
    assert resolved.artifacts.save_dir == str(context.checkpoints_dir)


def test_compose_run_name_prioritizes_comparison_fields() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "gnn_pointer"
    cfg.format.rollout_groups = [
        {"name": "two_player", "player_count": 2, "num_envs": 8},
        {"name": "four_player", "player_count": 4, "num_envs": 8},
    ]
    cfg.opponents.self_play.enabled = True
    cfg.training.total_updates = 500
    cfg.seed = 42

    run_name = compose_run_name(cfg)

    assert re.fullmatch(
        rf"gnn_pointer-mix2p4p-selfplay-u500-env16-s42-{RUN_NAME_TIMESTAMP_RE}",
        run_name,
    )


def test_compose_run_name_includes_hydra_job_when_present(monkeypatch) -> None:
    monkeypatch.setattr(run_paths, "_hydra_job_num", lambda: 7)
    cfg = TrainConfig()
    cfg.model.architecture = "attention"
    cfg.task.player_count = 2
    cfg.format.rollout_groups = []
    cfg.opponents.self_play.enabled = False
    cfg.opponents.mix.weights = {"random": 1.0, "latest": 0.0}
    cfg.training.total_updates = 1000
    cfg.training.num_envs = 16
    cfg.seed = 123

    run_name = compose_run_name(cfg)

    assert re.fullmatch(
        rf"attention-2p-random-u1000-env16-s123-job0007-{RUN_NAME_TIMESTAMP_RE}",
        run_name,
    )
