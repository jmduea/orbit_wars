from __future__ import annotations

import json
from pathlib import Path

from src import run_paths
from src.config import TrainConfig
from src.run_paths import resolve_run_paths, write_run_manifests


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


def test_hydra_runtime_output_dir_is_the_run_envelope(tmp_path: Path, monkeypatch) -> None:
    hydra_run_dir = tmp_path / "outputs" / "campaigns" / "scratch" / "runs" / "hydra-run"
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
