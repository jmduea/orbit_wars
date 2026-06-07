from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.artifacts.checkpoint_eval import run_checkpoint_eval_job
from src.cli import benchmark as benchmark_cli


def test_tournament_proof_dry_run_documents_docker_first_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"stub")
    cal = tmp_path / "calibration.json"
    cal.write_text(
        json.dumps(
            {
                "unified_tournament": {
                    "enforcement": False,
                    "noop_min_combined": 0.7,
                    "random_min_combined": 0.58,
                    "games_per_pair": 4,
                    "prerequisite_seeds": [0],
                    "incumbent_seeds": [0],
                    "four_p_baseline_fillers": ["noop", "random", "random"],
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "tournament.json"
    assert (
        benchmark_cli.main(
            [
                "tournament-proof",
                "--eval-checkpoint",
                str(checkpoint),
                "--out",
                str(out),
                "--thresholds-path",
                str(cal),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["submit_valid_order"] == [
        "docker_validation",
        "unified_tournament_ladder",
    ]


@patch("src.artifacts.tournament.unified.ladder.run_unified_ladder")
@patch("src.artifacts.docker_validation.run_submit_valid_docker_gate")
def test_tournament_proof_runs_docker_before_ladder(
    mock_docker,
    mock_ladder,
    tmp_path: Path,
) -> None:
    from src.artifacts.tournament.unified.reporting import (
        UnifiedLadderVerdict,
        UnifiedStageResult,
    )

    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"stub")
    cal = tmp_path / "calibration.json"
    cal.write_text(
        json.dumps(
            {
                "unified_tournament": {
                    "enforcement": True,
                    "noop_min_combined": 0.7,
                    "random_min_combined": 0.58,
                    "games_per_pair": 2,
                    "prerequisite_seeds": [0],
                    "incumbent_seeds": [0],
                    "four_p_baseline_fillers": ["noop", "random", "random"],
                }
            }
        ),
        encoding="utf-8",
    )
    call_order: list[str] = []

    def _docker(**_kwargs: object) -> dict[str, object]:
        call_order.append("docker")
        return {"validation_ok": True}

    def _ladder(*_args: object, **_kwargs: object) -> UnifiedLadderVerdict:
        call_order.append("ladder")
        return UnifiedLadderVerdict(
            passed=True,
            reason="pass",
            stages=(UnifiedStageResult(name="stage1_prerequisites", passed=True),),
            challenger_checkpoint=str(checkpoint),
        )

    mock_docker.side_effect = _docker
    mock_ladder.side_effect = _ladder

    out = tmp_path / "proof.json"
    assert (
        benchmark_cli.main(
            [
                "tournament-proof",
                "--eval-checkpoint",
                str(checkpoint),
                "--out",
                str(out),
                "--thresholds-path",
                str(cal),
            ]
        )
        == 0
    )
    assert call_order == ["docker", "ladder"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["docker_validation_ok"] is True
    assert payload["submit_valid_order"] == [
        "docker_validation",
        "unified_tournament_ladder",
    ]


@patch("src.artifacts.tournament.unified.ladder.run_unified_ladder")
@patch("src.artifacts.docker_validation.run_submit_valid_docker_gate")
def test_tournament_proof_skips_ladder_when_docker_fails(
    mock_docker,
    mock_ladder,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"stub")
    cal = tmp_path / "calibration.json"
    cal.write_text(
        json.dumps(
            {
                "unified_tournament": {
                    "enforcement": True,
                    "noop_min_combined": 0.7,
                    "random_min_combined": 0.58,
                    "games_per_pair": 2,
                    "prerequisite_seeds": [0],
                    "incumbent_seeds": [0],
                    "four_p_baseline_fillers": ["noop", "random", "random"],
                }
            }
        ),
        encoding="utf-8",
    )
    mock_docker.side_effect = RuntimeError("docker down")

    out = tmp_path / "proof.json"
    assert (
        benchmark_cli.main(
            [
                "tournament-proof",
                "--eval-checkpoint",
                str(checkpoint),
                "--out",
                str(out),
                "--thresholds-path",
                str(cal),
            ]
        )
        == 1
    )
    mock_ladder.assert_not_called()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tournament_skipped"] is True
    assert payload["tournament_skipped_reason"] == "docker_validation_failed"


@patch("src.artifacts.checkpoint_eval.run_tournament_promotion_job")
@patch("src.artifacts.checkpoint_eval.run_docker_gate_for_job")
def test_checkpoint_eval_runs_docker_before_tournament(
    mock_docker,
    mock_tournament,
    tmp_path: Path,
) -> None:
    from src.artifacts.tournament.types import TournamentResult

    call_order: list[str] = []

    def _docker_gate(
        _job: dict[str, object], *, result_dir: Path, repo_root: Path
    ) -> tuple[dict[str, object], bool]:
        call_order.append("docker")
        return {"validation_ok": True}, True

    def _tournament(_job: dict[str, object], *, result_dir: Path):
        call_order.append("tournament")
        return (
            TournamentResult(
                tournament_id="t-1",
                output_dir=result_dir,
                outcomes=(),
                leaderboard=(),
            ),
            None,
        )

    mock_docker.side_effect = _docker_gate
    mock_tournament.side_effect = _tournament

    run_checkpoint_eval_job(
        {"checkpoint_path": str(tmp_path / "ckpt.pkl")},
        result_dir=tmp_path / "result",
    )
    assert call_order == ["docker", "tournament"]


@patch("src.artifacts.checkpoint_eval.run_tournament_promotion_job")
@patch("src.artifacts.checkpoint_eval.run_docker_gate_for_job")
def test_checkpoint_eval_skips_tournament_when_docker_manifest_not_ok(
    mock_docker,
    mock_tournament,
    tmp_path: Path,
) -> None:
    mock_docker.return_value = ({"validation_ok": False}, False)

    result = run_checkpoint_eval_job(
        {"checkpoint_path": str(tmp_path / "ckpt.pkl")},
        result_dir=tmp_path / "result",
    )

    mock_tournament.assert_not_called()
    assert result["validation_ok"] is False
    assert result["tournament_skipped"] is True
