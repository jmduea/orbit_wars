"""JAX training-loop smoke: seed scheduler emits reseed_events in JSONL (slow tier)."""

from __future__ import annotations

import json

import pytest


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
from src.jax.train import run_jax_training


def _minimal_reseed_training_cfg(*, reseed_every_updates: int) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.player_count = 2
    cfg.task.max_fleets = 8
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    _configure_rollout_groups(cfg, [
        {"name": "two_player", "player_count": 2, "num_envs": 1},
    ])
    cfg.training.num_envs = 1
    cfg.training.rollout_steps = 1
    cfg.training.total_updates = 4
    cfg.training.epochs = 1
    cfg.training.update_chunk_rows = 32
    cfg.training.rollout_microbatch_envs = 1
    cfg.training.reseed_every_updates = reseed_every_updates
    cfg.artifacts.checkpoint_every = 100
    cfg.artifacts.artifact_pipeline.enabled = False
    cfg.artifacts.replay.enabled = False
    cfg.telemetry.wandb.enabled = False
    cfg.run_name = "seed_scheduler_smoke"
    return cfg


def _load_jsonl_records(tmp_path) -> list[dict[str, object]]:
    log_path = next(tmp_path.glob("campaigns/*/runs/*/logs/*_jax.jsonl"))
    return [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.jax
@pytest.mark.slow
def test_jax_training_logs_periodic_reseed_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")
    cfg = _minimal_reseed_training_cfg(reseed_every_updates=2)
    cfg.output.root = str(tmp_path)
    cfg.artifacts.save_dir = str(tmp_path)

    run_jax_training(cfg)

    records = _load_jsonl_records(tmp_path)
    training_records = [record for record in records if "reseed_events" in record]
    reseed_by_update = {
        int(record["update"]): record["reseed_events"] for record in training_records
    }

    assert reseed_by_update[1] == []
    assert len(reseed_by_update[2]) == 1
    assert len(reseed_by_update[4]) == 1

    for update in (2, 4):
        event = reseed_by_update[update][0]
        assert event["reason"] == "periodic"
        assert event["policy"] == "random_jump"
        assert event["old_seed"] != event["new_seed"]
        assert event["update"] == update


@pytest.mark.jax
@pytest.mark.slow
def test_jax_training_skips_reseed_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")
    cfg = _minimal_reseed_training_cfg(reseed_every_updates=0)
    cfg.output.root = str(tmp_path)
    cfg.artifacts.save_dir = str(tmp_path)

    run_jax_training(cfg)

    records = _load_jsonl_records(tmp_path)
    training_records = [record for record in records if "reseed_events" in record]
    assert all(not record["reseed_events"] for record in training_records)
