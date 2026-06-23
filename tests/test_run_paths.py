from __future__ import annotations

import json
import re
from pathlib import Path

from src.artifacts import run_paths
from src.artifacts.run_paths import (
    compose_run_name,
    resolve_run_paths,
    write_run_manifests,
)

def _configure_rollout_groups(cfg, groups):
    if not groups:
        cfg.training.format_weights = {int(cfg.task.player_count): 1.0}
        return
    active = [group for group in groups if int(group.get("num_envs", 0)) > 0]
    if len(active) == 1:
        group = active[0]
        cfg.training.num_envs = int(group["num_envs"])
        cfg.training.format_weights = {int(group["player_count"]): 1.0}
        return
    total = sum(int(group["num_envs"]) for group in active)
    cfg.training.num_envs = total
    cfg.training.rotate_format_rollouts = False
    cfg.training.format_weights = {
        int(group["player_count"]): int(group["num_envs"]) / float(total)
        for group in active
    }

from src.config import TrainConfig
from src.config.runtime import _orbit_sweep_subdir

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
    assert manifest["pointer_decoder"] == "factorized_topk"
    assert manifest["action_layout_version"] == 2
    assert manifest["paths"]["checkpoints_dir"] == str(context.checkpoints_dir)
    assert context.campaign_manifest_path.exists()
    campaign_manifest = json.loads(
        context.campaign_manifest_path.read_text(encoding="utf-8")
    )
    assert campaign_manifest["promotion_metric_name"] == "episode_reward_mean"
    assert (context.indexes_dir / "runs.jsonl").exists()


def test_orbit_sweep_subdir_uses_override_dirname_when_present() -> None:
    assert _orbit_sweep_subdir(3, "training.learning_rate=0.001", "run-abc") == (
        "runs/training.learning_rate=0.001"
    )


def test_orbit_sweep_subdir_falls_back_to_job_num_and_run_id() -> None:
    assert _orbit_sweep_subdir(7, "", "20260530T120000Z-s42-deadbeef") == (
        "runs/7_20260530T120000Z-s42-deadbeef"
    )


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
    cfg.model.architecture = "planet_graph_transformer"
    _configure_rollout_groups(cfg, [
        {"name": "two_player", "player_count": 2, "num_envs": 8},
        {"name": "four_player", "player_count": 4, "num_envs": 8},
    ])
    cfg.opponents.self_play.enabled = True
    cfg.training.total_updates = 500
    cfg.seed = 42

    run_name = compose_run_name(cfg)

    assert re.fullmatch(
        rf"planet_graph_transformer-mix2p4p-selfplay-u500-env16-s42-{RUN_NAME_TIMESTAMP_RE}",
        run_name,
    )


def test_compose_run_name_includes_hydra_job_when_present(monkeypatch) -> None:
    monkeypatch.setattr(run_paths, "_hydra_job_num", lambda: 7)
    cfg = TrainConfig()
    cfg.model.architecture = "attention"
    cfg.task.player_count = 2
    _configure_rollout_groups(cfg, [])
    cfg.opponents.self_play.enabled = False
    cfg.curriculum.enabled = False
    cfg.curriculum.stages = [
        {"id": "random_only", "opponent_families": {"random": 1.0, "latest": 0.0}}
    ]
    cfg.training.total_updates = 1000
    cfg.training.num_envs = 16
    cfg.seed = 123

    run_name = compose_run_name(cfg)

    assert re.fullmatch(
        rf"attention-2p-random-u1000-env16-s123-job0007-{RUN_NAME_TIMESTAMP_RE}",
        run_name,
    )
