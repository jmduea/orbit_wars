"""Composite checkpoint eval: Docker submission validation then tournament promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts.docker_validation import run_docker_gate_for_job
from src.artifacts.tournament.worker import run_tournament_promotion_job

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_checkpoint_eval_job(
    job: dict[str, object],
    *,
    result_dir: Path,
) -> dict[str, Any]:
    """Run docker validation, then tournament promotion, under one result directory."""

    _, validation_ok = run_docker_gate_for_job(
        job, result_dir=result_dir, repo_root=REPO_ROOT
    )
    if not validation_ok:
        return {
            "validation_ok": False,
            "docker_manifest_path": str(result_dir / "docker_manifest.json"),
            "docker_output_dir": str(result_dir / "docker_validation"),
            "tournament_skipped": True,
            "tournament_skipped_reason": "docker_validation_failed",
            "promoted": False,
            "promotion_reason": "docker_validation_failed",
        }

    tournament_dir = result_dir / "tournament"
    tournament, promotion_attempt = run_tournament_promotion_job(
        job, result_dir=tournament_dir
    )

    promoted = bool(promotion_attempt and promotion_attempt.promoted)
    unified_verdict_path = tournament_dir / "unified_verdict.json"
    return {
        "validation_ok": validation_ok,
        "docker_manifest_path": str(result_dir / "docker_manifest.json"),
        "docker_output_dir": str(result_dir / "docker_validation"),
        "tournament_id": tournament.tournament_id,
        "leaderboard_path": str(tournament_dir / "leaderboard.json"),
        "tournament_unified_verdict_path": str(unified_verdict_path)
        if unified_verdict_path.is_file()
        else None,
        "tournament_unified_passed": (
            json.loads(unified_verdict_path.read_text(encoding="utf-8")).get("passed")
            if unified_verdict_path.is_file()
            else None
        ),
        "promoted": promoted,
        "promotion_reason": promotion_attempt.reason
        if promotion_attempt
        else "no_passing_row",
        "tournament": {
            "tournament_id": tournament.tournament_id,
            "leaderboard_rows": len(tournament.leaderboard),
        },
    }
