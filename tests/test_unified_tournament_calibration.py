from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifacts.tournament.unified.reporting import (
    UnifiedLadderVerdict,
    UnifiedStageResult,
)
from src.artifacts.tournament.unified.scoring import UnifiedOpponentScore
from src.jax.unified_tournament_calibration import (
    UnifiedCalSnapshot,
    build_calibrated_unified_section,
    build_unified_calibration_report,
    derive_unified_floors,
    discover_unified_cal_snapshots,
    pick_games_per_pair,
    run_unified_calibration_sweep,
    unified_cal_campaign,
    verification_passes_at_derived_floors,
)
from src.jax.unified_tournament_calibration import UnifiedCalibrationPlan


def _snapshot(
    *,
    games: int,
    noop: float,
    random: float,
    seconds: float | None = 1.0,
) -> UnifiedCalSnapshot:
    return UnifiedCalSnapshot(
        checkpoint_path="/tmp/ckpt.pkl",
        games_per_pair=games,
        campaign=unified_cal_campaign(games),
        output_dir="/tmp/out",
        seconds_total=seconds,
        noop_combined=noop,
        random_combined=random,
        noop_win_rate_2p=noop,
        noop_win_rate_4p=noop,
        random_win_rate_2p=random,
        random_win_rate_4p=random,
        stage1_passed=True,
        reason="stage1_calibration_only",
    )


def test_pick_games_per_pair_prefers_higher_min_combined() -> None:
    decision = pick_games_per_pair(
        [
            _snapshot(games=2, noop=0.8, random=0.6),
            _snapshot(games=4, noop=0.95, random=0.75),
            _snapshot(games=8, noop=0.94, random=0.74),
        ]
    )
    assert decision["chosen_games_per_pair"] == 4


def test_derive_unified_floors_applies_margin() -> None:
    floors = derive_unified_floors(
        [_snapshot(games=4, noop=0.90, random=0.70)],
        games_per_pair=4,
        margin_fraction=0.05,
    )
    assert floors["noop_min_combined"] == pytest.approx(0.855)
    assert floors["random_min_combined"] == pytest.approx(0.665)


def test_build_calibrated_section_enables_enforcement_when_verified() -> None:
    snapshots = [_snapshot(games=4, noop=0.90, random=0.70)]
    section, decision = build_calibrated_unified_section(
        snapshots,
        base_section={"incumbent_bootstrap_opponent": "nearest_sniper"},
        enable_enforcement=True,
    )
    assert section["games_per_pair"] == 4
    assert section["enforcement"] is True
    assert float(section["noop_min_combined"]) < 0.90
    assert decision["enforcement"] is True


def test_verification_passes_at_derived_floors() -> None:
    snapshots = [_snapshot(games=4, noop=0.90, random=0.70)]
    floors = derive_unified_floors(snapshots, games_per_pair=4)
    assert verification_passes_at_derived_floors(
        snapshots,
        games_per_pair=4,
        noop_floor=float(floors["noop_min_combined"]),
        random_floor=float(floors["random_min_combined"]),
    )


def test_discover_unified_cal_snapshots_reads_verdict(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    games = 4
    output_dir = (
        output_root
        / "campaigns"
        / unified_cal_campaign(games)
        / "evaluations"
        / "cal_jax_ckpt_last"
    )
    output_dir.mkdir(parents=True)
    verdict = {
        "reason": "stage1_calibration_only",
        "stages": [
            {
                "name": "stage1_prerequisites",
                "passed": True,
                "opponents": [
                    {
                        "opponent": "noop",
                        "combined": 0.92,
                        "win_rate_2p": 0.9,
                        "win_rate_4p": 0.94,
                    },
                    {
                        "opponent": "random",
                        "combined": 0.71,
                        "win_rate_2p": 0.7,
                        "win_rate_4p": 0.72,
                    },
                ],
            }
        ],
    }
    (output_dir / "unified_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8"
    )
    ckpt = tmp_path / "jax_ckpt_last.pkl"
    ckpt.write_bytes(b"x")
    found = discover_unified_cal_snapshots(
        output_root,
        games_per_pair_candidates=(4,),
        checkpoint_paths=(ckpt,),
    )
    assert len(found) == 1
    assert found[0].noop_combined == pytest.approx(0.92)
    assert found[0].random_combined == pytest.approx(0.71)


def test_run_unified_calibration_sweep_dry_run(tmp_path: Path) -> None:
    ckpt = tmp_path / "jax_ckpt_last.pkl"
    ckpt.write_bytes(b"ckpt")
    plan = UnifiedCalibrationPlan(
        checkpoint_paths=(ckpt,),
        games_per_pair_candidates=(2, 4),
        dry_run=True,
        output_root=tmp_path / "outputs",
    )
    snapshots = run_unified_calibration_sweep(
        plan=plan,
        repo_root=tmp_path,
        base_section=None,
    )
    assert len(snapshots) == 2


def test_build_unified_calibration_report_dry_run(tmp_path: Path) -> None:
    report = build_unified_calibration_report(
        repo_root=tmp_path,
        plan=UnifiedCalibrationPlan(
            checkpoint_paths=(tmp_path / "c.pkl",),
            games_per_pair_candidates=(4,),
            dry_run=True,
        ),
        snapshots=[],
        analyze_only=False,
        seconds_total=0.01,
        base_section=None,
        enable_enforcement=False,
    )
    assert report["gate"] == "unified_tournament_calibration"
    assert report["unified_tournament"]["enforcement"] is False


def test_calibrate_unified_tournament_cli_dry_run(tmp_path: Path) -> None:
    from src.cli.benchmark import run_calibrate_unified_tournament_cli
    import argparse

    ckpt = tmp_path / "jax_ckpt_last.pkl"
    ckpt.write_bytes(b"ckpt")
    args = argparse.Namespace(
        out=tmp_path / "preflight.json",
        artifact_out=tmp_path / "artifact.json",
        output_root=tmp_path / "outputs",
        checkpoint=[ckpt],
        games_per_pair="4",
        analyze_only=False,
        dry_run=True,
        write_stub=False,
    )
    assert run_calibrate_unified_tournament_cli(args) == 0
    assert (tmp_path / "artifact.json").is_file()


@patch("src.jax.unified_tournament_calibration.run_unified_ladder")
def test_run_unified_calibration_arm_records_scores(
    mock_ladder: object, tmp_path: Path
) -> None:
    from src.jax.unified_tournament_calibration import run_unified_calibration_arm

    mock_ladder.return_value = UnifiedLadderVerdict(
        passed=True,
        reason="stage1_calibration_only",
        stages=(
            UnifiedStageResult(
                name="stage1_prerequisites",
                passed=True,
                opponents=(
                    UnifiedOpponentScore(
                        opponent="noop",
                        win_rate_2p=0.9,
                        win_rate_4p=0.8,
                        combined=0.85,
                        passed=True,
                    ),
                    UnifiedOpponentScore(
                        opponent="random",
                        win_rate_2p=0.7,
                        win_rate_4p=0.6,
                        combined=0.65,
                        passed=True,
                    ),
                ),
            ),
        ),
        challenger_checkpoint="/tmp/ckpt.pkl",
    )
    ckpt = tmp_path / "jax_ckpt_last.pkl"
    ckpt.write_bytes(b"ckpt")
    snapshot = run_unified_calibration_arm(
        checkpoint_path=ckpt,
        games_per_pair=4,
        output_root=tmp_path / "outputs",
        repo_root=tmp_path,
        base_section=None,
        dry_run=False,
    )
    assert snapshot.noop_combined == pytest.approx(0.85)
    assert snapshot.random_combined == pytest.approx(0.65)
