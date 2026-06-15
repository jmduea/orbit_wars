"""Preflight W&B sweep recipe composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.make_wandb_sweep import compose_sweep_gen, write_wandb_sweep


def test_preflight_wandb_sweep_compose() -> None:
    cfg = compose_sweep_gen(["wandb_sweep=preflight"])
    assert cfg["name"] == "preflight"
    assert cfg["metric"]["name"] == "preflight_sweep_score"
    assert cfg["metric"]["goal"] == "maximize"
    params = cfg["parameters"]
    tags = params["telemetry.wandb.tags"]["value"]
    assert "preflight" in tags
    assert params["telemetry.wandb.log_artifacts"]["value"] is True
    assert params["telemetry.metric_groups.action_decision"]["value"] is True
    assert params["telemetry.metric_groups.losses"]["value"] is True
    assert params["task"]["value"] == "rollout_selected_validate"
    assert params["training.total_updates"]["value"] == 100
    assert params["training.reseed_every_updates"]["value"] == 25
    assert params["train_bundle"]["value"] in (
        "opponent_recovery_floor",
        "production_mix",
    )
    assert params["artifacts"]["value"] == "default"
    assert cfg["method"] == "bayes"
    assert params["training.lr"]["distribution"] == "log_uniform_values"
    assert params["training.lr"]["min"] == pytest.approx(5e-5)
    assert params["training.lr"]["max"] == pytest.approx(3e-4)
    assert params["training.epochs"]["values"] == [1, 2]


def test_preflight_wandb_sweep_writes_yaml(tmp_path: Path) -> None:
    cfg = compose_sweep_gen(["wandb_sweep=preflight"])
    cfg["out_dir"] = str(tmp_path)
    out = write_wandb_sweep(cfg)
    assert out.name == "preflight.yaml"
    text = out.read_text(encoding="utf-8")
    assert "preflight_sweep_score" in text
    assert "telemetry.metric_groups.losses" in text
    assert "telemetry.metric_groups.action_decision" in text
