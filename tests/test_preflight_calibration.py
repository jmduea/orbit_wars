"""Tests for preflight calibration signal extraction and threshold derivation."""

from __future__ import annotations

import json

from src.jax.preflight_calibration import (
    PREFLIGHT_TRAIN_BASE,
    calibration_train_overrides,
    derive_thresholds,
    discover_calibration_snapshots,
    extract_training_signals,
    summarize_calibration,
)


def test_preflight_train_base_logs_every_update() -> None:
    assert "training.log_every=1" in PREFLIGHT_TRAIN_BASE


def test_planet_flow_calibration_overrides_use_p0_guards() -> None:
    overrides = calibration_train_overrides(
        "noop_only",
        seed=42,
        total_updates=20,
        model="planet_flow_target_heatmap",
    )

    assert "artifacts=planet_flow_proof" in overrides
    assert "training=planet_flow" in overrides
    assert "training.rollout_steps=128" not in overrides
    assert "curriculum=off" in overrides
    assert "telemetry.metric_groups.action_decision=true" in overrides


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


def test_planet_flow_calibration_writes_planet_flow_thresholds() -> None:
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
        model="planet_flow_target_heatmap",
    )
    thresholds = derive_thresholds(summarize_calibration([snapshot]))

    planet_flow = thresholds["planet_flow_learning_signal"]
    learning = thresholds["learning_signal"]
    assert isinstance(planet_flow, dict)
    assert isinstance(learning, dict)
    for key, value in learning.items():
        assert planet_flow[key] == value
    assert planet_flow["max_post_mask_unreachable_demand_rate"] == 0.05


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


def test_discover_calibration_snapshots_ignores_newer_empty_runs(tmp_path) -> None:
    campaign = tmp_path / "campaigns" / "preflight_calibrate_noop_s42_u200"
    older_run = campaign / "runs" / "20260531T000000Z-s42-complete"
    older_logs = older_run / "logs"
    older_logs.mkdir(parents=True)
    records = [
        {
            "update": index,
            "overall_win_rate": 0.1 if index <= 10 else 0.4,
            "mean_active_launches_per_turn": 1.0,
        }
        for index in range(1, 21)
    ]
    (older_logs / "run_jax.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    newer_run = campaign / "runs" / "20260531T010000Z-s42-empty"
    newer_logs = newer_run / "logs"
    newer_logs.mkdir(parents=True)
    (newer_logs / "run_jax.jsonl").write_text("", encoding="utf-8")
    older_run.touch()
    newer_run.touch()

    snapshots = discover_calibration_snapshots(tmp_path)

    assert len(snapshots) == 1
    assert snapshots[0].run_dir == str(older_run)
    assert snapshots[0].win_rate_delta is not None


def test_unified_tournament_section_in_committed_calibration_parses() -> None:
    from pathlib import Path

    from src.artifacts.tournament.unified.spec import load_unified_tournament_spec

    path = Path("docs/benchmarks/preflight-calibration.json")
    spec = load_unified_tournament_spec(path)
    assert not spec.needs_calibration
    assert spec.stage1.floors["noop"] == 0.7
    assert spec.stage1.floors["random"] == 0.58
    assert not spec.enforcement


def test_refresh_agents_md_thresholds_replaces_block(tmp_path) -> None:
    from src.jax.preflight_calibration import (
        PREFLIGHT_THRESHOLDS_END,
        PREFLIGHT_THRESHOLDS_START,
        format_agents_md_threshold_block,
        refresh_agents_md_thresholds,
    )

    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(
        "\n".join(
            [
                "# Guide",
                "- **Verification thresholds:** policy line.",
                PREFLIGHT_THRESHOLDS_START,
                "- old bullet",
                PREFLIGHT_THRESHOLDS_END,
                "- **Metric gates:** next section.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "commit_sha": "abc123def456",
        "thresholds": {
            "learning_signal": {
                "window_updates": 10,
                "min_win_rate_delta": 0.05,
                "max_approx_kl": 0.15,
                "min_entropy": 0.0001,
            },
            "win_proof_tournament": {
                "noop_min_win_rate": 0.7,
                "random_min_win_rate": 0.58,
            },
        },
    }
    block = format_agents_md_threshold_block(report)
    assert "min_win_rate_delta=0.05" in block
    assert refresh_agents_md_thresholds(tmp_path, report, agents_md_path=agents_md)
    updated = agents_md.read_text(encoding="utf-8")
    assert "old bullet" not in updated
    assert "noop_min_win_rate=0.7" in updated
    assert not refresh_agents_md_thresholds(tmp_path, report, agents_md_path=agents_md)
