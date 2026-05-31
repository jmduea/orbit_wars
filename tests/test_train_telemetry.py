import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp

from src.jax.train_checkpoint import HistoricalSnapshotPool
from src.jax.train_telemetry import (
    build_update_record,
    split_debug_update_record,
    write_filtered_update_records,
)


def test_split_debug_update_record_separates_debug_keys() -> None:
    lean, debug = split_debug_update_record(
        {
            "update": 1,
            "loss": 0.5,
            "debug_rollout_scan": {"a": 1},
            "debug/parity": 2,
        }
    )

    assert lean == {"update": 1, "loss": 0.5}
    assert debug == {"debug_rollout_scan": {"a": 1}, "debug/parity": 2}


def test_build_update_record_applies_rollout_metric_filtering() -> None:
    cfg = SimpleNamespace(model=SimpleNamespace(max_moves_k=4))
    cfg.telemetry = SimpleNamespace(metric_groups={})
    historical_pool = HistoricalSnapshotPool(
        params={"w": jnp.zeros((2, 2))},
        snapshot_ids=jnp.zeros((2,), dtype=jnp.int32),
        snapshot_updates=jnp.zeros((2,), dtype=jnp.int32),
        valid_mask=jnp.array([True, False]),
    )
    rollout_scalars = {
        "env_steps": 10.0,
        "episode_done": 2.0,
        "win_rate_2p": 0.5,
        "first_place_rate_4p": 0.0,
        "average_placement_4p": 0.0,
        "overall_win_rate": 0.5,
        "survival_time": 1.0,
        "score_share": 0.25,
        "average_reward": 0.1,
        "episode_reward_mean": 0.2,
        "mean_active_launches_per_turn": 2.0,
        "debug_rollout_scan": 99.0,
    }

    record = build_update_record(
        update=1,
        total_env_steps=10,
        completed_episodes=2,
        rollout_samples=100,
        rollout_scalars=rollout_scalars,
        metrics_host={"total_loss": 1.0},
        update_seconds=1.0,
        rollout_seconds=0.5,
        ppo_seconds=0.4,
        train_start_time=0.0,
        per_format_timing_metrics={"rollout_seconds_2p": 0.5},
        curriculum_telemetry={"curriculum_stage_index": 0},
        reseed_events=[],
        update_events=[],
        historical_pool=historical_pool,
        gpu_update_metrics={},
        seed_scheduler_policy="fixed",
        plateau_metric="overall_win_rate",
        cfg=cfg,
    )

    assert record["historical_pool_size"] == 1
    assert record["stop_utilization_ratio"] == 0.5
    assert record["episode_reward_mean"] == 0.2
    assert "average_reward" not in record
    assert "debug_rollout_scan" not in record


def test_write_filtered_update_records_respects_disabled_groups(
    tmp_path: Path,
) -> None:
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.telemetry.metric_groups = {"rollout": False}
    log_path = tmp_path / "metrics.jsonl"
    debug_log_path = tmp_path / "debug.jsonl"
    telemetry = MagicMock()
    record = {
        "update": 1,
        "win_rate_2p": 0.5,
        "debug_rollout_scan": {"chunk": 1},
    }

    write_filtered_update_records(
        log_path=log_path,
        debug_log_path=debug_log_path,
        record=record,
        cfg=cfg,
        telemetry=telemetry,
        update=1,
    )

    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert "debug_rollout_scan" not in logged
    assert debug_log_path.exists()
    telemetry.log.assert_called_once()
