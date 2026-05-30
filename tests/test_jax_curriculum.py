"""JAX curriculum training-loop integration smokes (slow tier).

Ownership:
- End-to-end staged self-play training logs and promotion are asserted here.
- Per-family opponent-slot rollout metrics live in ``tests/test_curriculum.py`` (CPU/JAX
  collect) and ``tests/test_jax_scripted_opponents.py`` (scripted families).
"""

import json

import pytest

from src.config import TrainConfig
from src.jax.train import run_jax_training


def _self_play_staged_v2_stages() -> list[dict[str, object]]:
    promote = {
        "metric": "overall_win_rate",
        "op": ">=",
        "value": 0.0,
        "window_updates": 1,
    }
    return [
        {
            "id": "bootstrap_random",
            "min_updates": 1,
            "promote_if": promote,
            "opponent_families": {
                "latest": 0.0,
                "historical": 0.0,
                "random": 1.0,
                "noop": 0.0,
                "nearest_sniper": 0.0,
                "turtle": 0.0,
                "opportunistic": 0.0,
            },
        },
        {
            "id": "mixed_exploiters",
            "min_updates": 1,
            "promote_if": promote,
            "opponent_families": {
                "latest": 0.55,
                "historical": 0.20,
                "random": 0.05,
                "noop": 0.05,
                "nearest_sniper": 0.07,
                "turtle": 0.04,
                "opportunistic": 0.04,
            },
        },
        {
            "id": "self_play_pressure",
            "min_updates": 1,
            "opponent_families": {
                "latest": 0.45,
                "historical": 0.35,
                "random": 0.03,
                "noop": 0.02,
                "nearest_sniper": 0.07,
                "turtle": 0.04,
                "opportunistic": 0.04,
            },
        },
    ]


def _v2_curriculum_training_cfg(*, player_count: int, four_player_num_envs: int) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.player_count = player_count
    cfg.curriculum.enabled = True
    cfg.curriculum.stages = _self_play_staged_v2_stages()
    cfg.opponents.mode.opponent = "self"
    cfg.opponents.self_play.enabled = True
    cfg.opponents.snapshot.pool_size = 2
    cfg.opponents.snapshot.interval_updates = 1
    cfg.run_name = f"curriculum_v2_{player_count}p"
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.model.max_moves_k = 2
    cfg.format.rollout_groups = [
        {"name": "two_player", "player_count": 2, "num_envs": 1 if player_count == 2 else 0},
        {
            "name": "four_player",
            "player_count": 4,
            "num_envs": four_player_num_envs,
        },
    ]
    cfg.training.num_envs = 1
    cfg.training.rollout_steps = 1
    cfg.training.total_updates = 3
    cfg.training.epochs = 1
    cfg.training.minibatch_size = 32
    cfg.training.rollout_microbatch_envs = 1
    cfg.artifacts.checkpoint_every = 100
    cfg.artifacts.artifact_pipeline.enabled = False
    cfg.artifacts.replay.enabled = False
    cfg.telemetry.wandb.enabled = False
    return cfg


def _assert_v2_curriculum_training_logs(tmp_path, cfg: TrainConfig) -> None:
    cfg.output.root = str(tmp_path)
    cfg.artifacts.save_dir = str(tmp_path)
    run_jax_training(cfg)

    log_path = next(tmp_path.glob("campaigns/*/runs/*/logs/*_jax.jsonl"))
    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    promoted = False
    for record in records:
        events = record.get("curriculum_phase_events", [])
        if any(event.get("event") == "curriculum_stage_promoted" for event in events):
            promoted = True

    final_record = next(record for record in records if record.get("update") == cfg.training.total_updates)
    assert promoted
    assert final_record["curriculum_phase_id"] == "self_play_pressure"
    for key in (
        "rollout_seconds_2p",
        "rollout_seconds_4p",
        "env_steps_per_sec_2p",
        "env_steps_per_sec_4p",
        "rollout_env_steps_per_sec_2p",
        "rollout_env_steps_per_sec_4p",
        "samples_per_sec_2p",
        "samples_per_sec_4p",
        "rollout_samples_per_sec_2p",
        "rollout_samples_per_sec_4p",
        "update_time_rollout_fraction",
        "update_time_ppo_fraction",
    ):
        assert isinstance(final_record[key], float)
        assert final_record[key] >= 0.0
        assert final_record[key] == final_record[key]


@pytest.mark.jax
def test_v2_training_loop_self_play_staged_2p_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")
    cfg = _v2_curriculum_training_cfg(player_count=2, four_player_num_envs=0)
    _assert_v2_curriculum_training_logs(tmp_path, cfg)


@pytest.mark.jax
def test_v2_training_loop_self_play_staged_4p_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")
    cfg = _v2_curriculum_training_cfg(player_count=4, four_player_num_envs=1)
    cfg.format.rollout_groups = [
        {"name": "two_player", "player_count": 2, "num_envs": 0},
        {"name": "four_player", "player_count": 4, "num_envs": 1},
    ]
    _assert_v2_curriculum_training_logs(tmp_path, cfg)
