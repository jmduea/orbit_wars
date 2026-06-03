"""Incumbent resolution and swap for unified tournament ladder."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.promotion_manifest import promoted_manifest_path
from src.artifacts.tournament.resolve import (
    agent_from_baseline,
    resolve_promoted_agent,
    run_context_for_agent,
)
from src.artifacts.tournament.types import AgentEntry
from src.artifacts.tournament.unified.reporting import UnifiedLadderVerdict
from src.artifacts.tournament.unified.spec import UnifiedTournamentSpec
from src.config import TrainConfig


def resolve_incumbent(
    spec: UnifiedTournamentSpec,
    *,
    campaign: str | None,
    output_root: Path,
) -> AgentEntry | None:
    """Resolve incumbent: campaign promoted manifest, then scripted bootstrap."""

    if campaign:
        incumbent = resolve_promoted_agent(campaign, str(output_root))
        if incumbent is not None:
            return incumbent
    if spec.incumbent_bootstrap_opponent is not None:
        return agent_from_baseline(spec.incumbent_bootstrap_opponent, agent_id="incumbent")
    return None


def swap_incumbent_on_unified_pass(
    cfg: TrainConfig,
    *,
    challenger: AgentEntry,
    verdict: UnifiedLadderVerdict,
    campaign: str,
    output_root: Path,
    update: int | None = None,
    tournament_output_dir: Path,
) -> bool:
    """Write promoted manifest when Stage 2 R9 passes."""

    if not verdict.passed or not verdict.incumbent_swap:
        return False
    from src.artifacts.tournament.promotion import promote_from_unified_ladder

    context = run_context_for_agent(
        challenger, campaign=campaign, output_root=str(output_root)
    )
    attempt = promote_from_unified_ladder(
        cfg,
        context,
        challenger=challenger,
        verdict=verdict,
        tournament_output_dir=tournament_output_dir,
        update=update,
    )
    return attempt.promoted


def incumbent_manifest_exists(campaign_dir: Path) -> bool:
    return promoted_manifest_path(campaign_dir).is_file()
