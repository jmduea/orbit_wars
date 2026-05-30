"""Artifact worker handler for async tournament promotion jobs."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.promotion import PromotionAttempt
from src.artifacts.tournament.eval import run_tournament
from src.artifacts.tournament.promotion import promote_from_tournament, top_passing_row
from src.artifacts.tournament.resolve import (
    agent_from_checkpoint,
    load_train_config_from_checkpoint,
    resolve_promoted_agent,
    run_context_for_agent,
)
from src.artifacts.tournament.types import TournamentResult


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
