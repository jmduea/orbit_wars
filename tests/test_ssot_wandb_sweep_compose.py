"""SSOT W&B preflight sweep recipe composition."""

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
    assert params["telemetry.wandb.tags"]["value"] == [
        "preflight",
        "production_mix",
        "2p4p_32_split",
        "100u",
        "25u_reseed",
    ]
    assert params["telemetry.wandb.log_artifacts"]["value"] is True
    assert params["telemetry.metric_groups.losses"]["value"] is True
    assert params["training.total_updates"]["value"] == 100
    assert params["train_bundle"]["value"] == "production_mix"
    assert cfg["method"] == "bayes"
    assert params["training.lr"]["distribution"] == "log_uniform_values"
    assert params["training.lr"]["min"] == pytest.approx(5e-5)
    assert params["training.lr"]["max"] == pytest.approx(3e-4)


def test_ssot_preflight_wandb_sweep_compose() -> None:
    cfg = compose_sweep_gen(["wandb_sweep=ssot_preflight"])
    assert cfg["name"] == "ssot_preflight"
    assert cfg["metric"]["name"] == "ssot_preflight_sweep_score"
    assert cfg["metric"]["goal"] == "maximize"
    params = cfg["parameters"]
    assert params["telemetry.wandb.tags"]["value"] == [
        "ssot_preflight",
        "gates_2_3",
        "mix2p4p_selfplay",
    ]
    assert params["telemetry.wandb.log_artifacts"]["value"] is True
    assert params["training.total_updates"]["value"] == 50
    assert cfg["method"] == "bayes"
    assert params["opponents"]["value"] == "self_play_only"
    assert params["curriculum"]["value"] == "off"
    assert params["training.lr"]["distribution"] == "log_uniform_values"
    assert params["training.lr"]["min"] == pytest.approx(5e-5)
    assert params["training.lr"]["max"] == pytest.approx(3e-4)


def test_preflight_wandb_sweep_writes_yaml(tmp_path: Path) -> None:
    cfg = compose_sweep_gen(["wandb_sweep=preflight"])
    cfg["out_dir"] = str(tmp_path)
    out = write_wandb_sweep(cfg)
    assert out.name == "preflight.yaml"
    text = out.read_text(encoding="utf-8")
    assert "preflight_sweep_score" in text
    assert "telemetry.metric_groups.losses" in text


def test_ssot_preflight_wandb_sweep_writes_yaml(tmp_path: Path) -> None:
    cfg = compose_sweep_gen(["wandb_sweep=ssot_preflight"])
    cfg["out_dir"] = str(tmp_path)
    out = write_wandb_sweep(cfg)
    assert out.name == "ssot_preflight.yaml"
    text = out.read_text(encoding="utf-8")
    assert "ssot_preflight_sweep_score" in text
