"""Composite checkpoint eval: Docker submission validation then tournament promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts.docker_validation import run_docker_validation_subprocess
from src.artifacts.run_paths import atomic_write_json
from src.artifacts.tournament.types import TournamentResult
from src.artifacts.tournament.worker import run_tournament_promotion_job

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_checkpoint_eval_job(
    job: dict[str, object],
    *,
    result_dir: Path,
) -> dict[str, Any]:
    """Run docker validation, then tournament promotion, under one result directory."""

    checkpoint_path = Path(str(job["checkpoint_path"]))
    docker_output_dir = result_dir / "docker_validation"
    docker_manifest = run_docker_validation_subprocess(
        checkpoint_path=checkpoint_path,
        output_dir=docker_output_dir,
        repo_root=REPO_ROOT,
        docker_image=str(
            job.get("docker_image", "gcr.io/kaggle-images/python-simulations")
        ),
        seed=int(job.get("seed", 42)),
        player_count=str(job.get("player_count", "both")),
        per_step_seconds=float(job.get("per_step_seconds", 1.0)),
        overage_budget_seconds=float(job.get("overage_budget_seconds", 60.0)),
        episode_steps=int(job.get("episode_steps", 500)),
    )
    atomic_write_json(result_dir / "docker_manifest.json", docker_manifest)

    tournament_dir = result_dir / "tournament"
    tournament, promotion_attempt = run_tournament_promotion_job(
        job, result_dir=tournament_dir
    )

    promoted = bool(promotion_attempt and promotion_attempt.promoted)
    unified_verdict_path = tournament_dir / "unified_verdict.json"
    return {
        "validation_ok": True,
        "docker_manifest_path": str(result_dir / "docker_manifest.json"),
        "docker_output_dir": str(docker_output_dir),
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
        "tournament": _tournament_summary(tournament),
    }


def _tournament_summary(tournament: TournamentResult) -> dict[str, object]:
    return {
        "tournament_id": tournament.tournament_id,
        "leaderboard_rows": len(tournament.leaderboard),
    }
