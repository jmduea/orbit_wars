"""Tests for preflight calibration signal extraction and threshold derivation."""

from __future__ import annotations

import json

from src.jax.preflight_calibration import (
    PREFLIGHT_TRAIN_BASE,
    derive_thresholds,
    discover_calibration_snapshots,
    extract_training_signals,
    summarize_calibration,
)


def test_preflight_train_base_logs_every_update() -> None:
    assert "training.log_every=1" in PREFLIGHT_TRAIN_BASE


def test_extract_training_signals_computes_trend() -> None:
    records = [
        {
            "update": index,
            "overall_win_rate": 0.1 if index <= 10 else 0.7,
            "mean_active_launches_per_turn": 0.2 if index <= 10 else 1.0,
            "approx_kl": 0.01,
            "entropy": 0.05,
        }
        for index in range(1, 21)
    ]
    snapshot = extract_training_signals(
        records,
        opponent="noop_only",
        seed=42,
        total_updates=20,
    )
    assert snapshot.win_rate_delta is not None
    assert snapshot.win_rate_delta > 0.5


def test_derive_thresholds_prefers_trend_plus_tournament() -> None:
    records = [
        {
            "update": index,
            "overall_win_rate": 0.15 if index <= 10 else 0.35,
            "mean_active_launches_per_turn": 1.0 if index <= 10 else 7.0,
            "approx_kl": 0.01,
            "entropy": 0.05,
        }
        for index in range(1, 21)
    ]
    snapshot = extract_training_signals(
        records,
        opponent="noop_only",
        seed=42,
        total_updates=20,
    )
    summary = summarize_calibration([snapshot])
    thresholds = derive_thresholds(summary)
    assert thresholds["mode"] == "trend_plus_tournament"
    learning = thresholds["learning_signal"]
    assert isinstance(learning, dict)
    assert float(learning["min_win_rate_delta"]) <= snapshot.win_rate_delta
    win_proof = thresholds["win_proof_tournament"]
    assert isinstance(win_proof, dict)
    assert float(win_proof["noop_min_win_rate"]) < 0.85


def test_discover_calibration_snapshots_reads_campaign_layout(tmp_path) -> None:
    campaign = tmp_path / "campaigns" / "preflight_calibrate_noop_s42_u200"
    run_dir = campaign / "runs" / "20260531T000000Z-s42-deadbeef"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    records = [
        {
            "update": index,
            "overall_win_rate": 0.1 if index <= 10 else 0.4,
            "mean_active_launches_per_turn": 1.0,
        }
        for index in range(1, 21)
    ]
    log_path = logs_dir / "run_jax.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    snapshots = discover_calibration_snapshots(tmp_path)
    assert len(snapshots) == 1
    assert snapshots[0].opponent == "noop_only"
    assert snapshots[0].seed == 42
    assert snapshots[0].total_updates == 200
    assert snapshots[0].win_rate_delta is not None
