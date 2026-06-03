"""Artifact worker handler for async tournament promotion jobs."""

from __future__ import annotations

import json
from pathlib import Path

from src.artifacts.promotion import PromotionAttempt
from src.artifacts.tournament.eval import run_tournament
from src.artifacts.tournament.promotion import (
    promote_from_tournament,
    promote_from_unified_ladder,
    top_passing_row,
)
from src.artifacts.tournament.resolve import (
    agent_from_checkpoint,
    load_train_config_from_checkpoint,
    resolve_promoted_agent,
    run_context_for_agent,
)
from src.artifacts.tournament.types import TournamentResult
from src.artifacts.tournament.unified.ladder import run_unified_ladder
from src.artifacts.tournament.unified.spec import load_unified_tournament_spec
from src.jax.preflight_calibration import default_calibration_json_path


def _unified_enabled(cfg) -> bool:
    return bool(getattr(cfg.artifacts, "unified_tournament", None) and cfg.artifacts.unified_tournament.enabled)


def _tournament_result_from_unified(output_dir: Path) -> TournamentResult:
    manifest_path = output_dir / "manifest.json"
    tournament_id = "unified"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("tournament_id"):
            tournament_id = str(payload["tournament_id"])
    return TournamentResult(
        tournament_id=tournament_id,
        output_dir=output_dir,
        outcomes=(),
        leaderboard=(),
    )


def run_tournament_promotion_job(
    job: dict[str, object],
    *,
    result_dir: Path,
) -> tuple[TournamentResult, PromotionAttempt | None]:
    """Execute a queued tournament promotion job."""

    checkpoint_path = Path(str(job["checkpoint_path"]))
    cfg = load_train_config_from_checkpoint(checkpoint_path)
    if cfg.artifacts.promotion.strategy in {"hybrid", "tournament"}:
        cfg.artifacts.tournament.enabled = True

    candidate = agent_from_checkpoint(
        checkpoint_path,
        agent_id=str(job.get("run_id", "candidate")),
    )
    campaign = str(job.get("campaign", cfg.output.campaign))
    output_root = Path(str(job.get("output_root", cfg.output.root)))

    if _unified_enabled(cfg):
        repo_root = Path(__file__).resolve().parents[3]
        spec = load_unified_tournament_spec(
            default_calibration_json_path(repo_root),
            hydra=cfg.artifacts.unified_tournament,
        )
        verdict = run_unified_ladder(
            checkpoint_path,
            spec,
            result_dir,
            campaign=campaign,
            output_root=output_root,
        )
        tournament = _tournament_result_from_unified(result_dir)
        if not verdict.passed:
            return tournament, None
        context = run_context_for_agent(
            candidate, campaign=campaign, output_root=str(output_root)
        )
        attempt = promote_from_unified_ladder(
            cfg,
            context,
            challenger=candidate,
            verdict=verdict,
            tournament_output_dir=result_dir,
            update=int(job["update"]) if job.get("update") is not None else None,
        )
        return tournament, attempt if attempt.promoted else None

    incumbent = resolve_promoted_agent(campaign, cfg.output.root)
    tournament = run_tournament(
        (candidate,),
        cfg=cfg.artifacts.tournament,
        output_dir=result_dir,
        incumbent=incumbent,
        promotion_gates=cfg.artifacts.promotion.tournament,
    )
    passing = top_passing_row(tournament)
    if passing is None:
        return tournament, None

    context = run_context_for_agent(candidate, campaign=campaign, output_root=cfg.output.root)
    attempt = promote_from_tournament(
        cfg,
        context,
        row=passing,
        tournament=tournament,
        update=int(job["update"]),
    )
    return tournament, attempt
