"""Tournament-gated campaign promotion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.artifacts.promotion import PromotionAttempt
from src.artifacts.promotion_manifest import (
    commit_promotion,
    promoted_manifest_path,
)
from src.artifacts.run_paths import RunContext, _git_identity
from src.artifacts.tournament.resolve import load_train_config_from_checkpoint
from src.artifacts.tournament.types import AgentEntry, LeaderboardRow, TournamentResult
from src.config import TrainConfig

if TYPE_CHECKING:
    from src.artifacts.tournament.unified.reporting import UnifiedLadderVerdict


def _incumbent_tournament_metrics(campaign_dir: Path) -> dict[str, float | None]:
    manifest_path = promoted_manifest_path(campaign_dir)
    if not manifest_path.exists():
        return {
            "win_rate_vs_sniper": None,
            "win_rate_vs_incumbent": None,
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"win_rate_vs_sniper": None, "win_rate_vs_incumbent": None}

    def _float(key: str) -> float | None:
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
        return None

    sniper = _float("tournament_win_rate_vs_sniper")
    if sniper is None:
        legacy = _float("metric_value")
        metric_name = str(payload.get("metric_name", ""))
        if metric_name == "tournament_win_rate_vs_sniper":
            sniper = legacy
    return {
        "win_rate_vs_sniper": sniper,
        "win_rate_vs_incumbent": _float("tournament_win_rate_vs_incumbent"),
    }


def tournament_improves_incumbent(
    row: LeaderboardRow,
    *,
    campaign_dir: Path,
) -> tuple[bool, str]:
    """Return whether candidate tournament stats beat the promoted incumbent."""

    incumbent = _incumbent_tournament_metrics(campaign_dir)
    incumbent_sniper = incumbent["win_rate_vs_sniper"]
    if incumbent_sniper is not None:
        if row.win_rate_vs_sniper is None:
            return False, "missing_candidate_win_rate_vs_sniper"
        if row.win_rate_vs_sniper <= incumbent_sniper:
            return False, "incumbent_win_rate_vs_sniper_unchanged"

    incumbent_h2h = incumbent["win_rate_vs_incumbent"]
    if incumbent_h2h is not None and row.win_rate_vs_incumbent is not None:
        if row.win_rate_vs_incumbent <= incumbent_h2h:
            return False, "incumbent_head_to_head_unchanged"
    return True, "improved"


def promote_from_tournament(
    cfg: TrainConfig,
    context: RunContext,
    *,
    row: LeaderboardRow,
    tournament: TournamentResult,
    update: int | None = None,
) -> PromotionAttempt:
    """Write promoted manifest when tournament gates pass for a candidate."""

    promotion = cfg.artifacts.promotion
    if not promotion.enabled:
        return PromotionAttempt(promoted=False, reason="disabled", metric_name="")
    if not row.gates_passed:
        return PromotionAttempt(
            promoted=False,
            reason="tournament_gates_failed",
            metric_name="tournament_win_rate_vs_sniper",
            metric_value=row.win_rate_vs_sniper,
        )

    improves, improve_reason = tournament_improves_incumbent(
        row, campaign_dir=context.campaign_dir
    )
    if not improves:
        return PromotionAttempt(
            promoted=False,
            reason=improve_reason,
            metric_name="tournament_win_rate_vs_sniper",
            metric_value=row.win_rate_vs_sniper,
        )

    metric_name = "tournament_win_rate_vs_sniper"
    metric_value = row.win_rate_vs_sniper
    now = datetime.now(timezone.utc).isoformat()
    checkpoint_path = Path(row.checkpoint_path)
    overrides_path = context.run_dir / ".hydra" / "overrides.yaml"
    from src.artifacts.checkpoint_compat import feature_metadata

    agent_cfg = load_train_config_from_checkpoint(checkpoint_path)

    promoted_payload: dict[str, object] = {
        "campaign": context.campaign_slug,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_mode": "max",
        "promotion_strategy": promotion.strategy,
        "source_run_id": context.run_id,
        "source_update": int(update) if update is not None else None,
        "hydra_overrides_path": str(overrides_path),
        "git": _git_identity(),
        "feature_metadata": feature_metadata(
            agent_cfg.task, model_cfg=agent_cfg.model
        ),
        "updated_at": now,
        "tournament_id": tournament.tournament_id,
        "tournament_output_dir": str(tournament.output_dir),
        "tournament_win_rate_vs_sniper": row.win_rate_vs_sniper,
        "tournament_win_rate_vs_incumbent": row.win_rate_vs_incumbent,
        "tournament_first_place_rate_4p": row.first_place_rate_4p,
        "tournament_gates_passed": True,
    }

    manifest_out = commit_promotion(
        campaign_dir=context.campaign_dir,
        campaign_manifest_path=context.campaign_manifest_path,
        indexes_dir=context.indexes_dir,
        promoted_payload=promoted_payload,
        campaign_updates={
            "campaign": context.campaign_slug,
            "campaign_dir": str(context.campaign_dir),
            "promotion_metric_name": metric_name,
            "promotion_metric_mode": "max",
            "promotion_strategy": promotion.strategy,
            "current_best_value": metric_value,
            "current_best_run_id": context.run_id,
            "updated_at": now,
        },
        index_record={
            "campaign": context.campaign_slug,
            "run_id": context.run_id,
            "update": update,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "promotion_strategy": promotion.strategy,
            "tournament_id": tournament.tournament_id,
            "updated_at": now,
        },
    )

    return PromotionAttempt(
        promoted=True,
        reason="tournament_promoted",
        metric_name=metric_name,
        metric_value=metric_value,
        metric_mode="max",
        promoted_manifest_path=manifest_out,
    )


def top_passing_row(result: TournamentResult) -> LeaderboardRow | None:
    for row in result.leaderboard:
        if row.gates_passed:
            return row
    return None


def promote_from_unified_ladder(
    cfg: TrainConfig,
    context: RunContext,
    *,
    challenger: AgentEntry,
    verdict: UnifiedLadderVerdict,
    tournament_output_dir: Path,
    update: int | None = None,
) -> PromotionAttempt:
    """Write promoted manifest when unified Stage 2 R9 passes."""

    promotion = cfg.artifacts.promotion
    if not promotion.enabled:
        return PromotionAttempt(promoted=False, reason="disabled", metric_name="")
    if not verdict.passed or not verdict.incumbent_swap:
        return PromotionAttempt(
            promoted=False,
            reason=verdict.reason or "unified_ladder_not_passed",
            metric_name="unified_combined",
        )

    stage1 = verdict.stages[0] if verdict.stages else None
    noop_score = None
    random_score = None
    if stage1 is not None:
        for row in stage1.opponents:
            if row.opponent == "noop":
                noop_score = row.combined
            elif row.opponent == "random":
                random_score = row.combined

    metric_name = "unified_combined_noop"
    metric_value = noop_score
    now = datetime.now(timezone.utc).isoformat()
    checkpoint_path = challenger.checkpoint_path.resolve()
    overrides_path = context.run_dir / ".hydra" / "overrides.yaml"
    from src.artifacts.checkpoint_compat import feature_metadata

    agent_cfg = load_train_config_from_checkpoint(checkpoint_path)

    promoted_payload: dict[str, object] = {
        "campaign": context.campaign_slug,
        "checkpoint_path": str(checkpoint_path),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_mode": "max",
        "promotion_strategy": promotion.strategy,
        "source_run_id": context.run_id,
        "source_update": int(update) if update is not None else None,
        "hydra_overrides_path": str(overrides_path),
        "git": _git_identity(),
        "feature_metadata": feature_metadata(
            agent_cfg.task, model_cfg=agent_cfg.model
        ),
        "updated_at": now,
        "tournament_output_dir": str(tournament_output_dir),
        "tournament_gates_passed": True,
        "unified_ladder_passed": True,
        "unified_combined_noop": noop_score,
        "unified_combined_random": random_score,
        "unified_verdict_reason": verdict.reason,
    }

    manifest_out = commit_promotion(
        campaign_dir=context.campaign_dir,
        campaign_manifest_path=context.campaign_manifest_path,
        indexes_dir=context.indexes_dir,
        promoted_payload=promoted_payload,
        campaign_updates={
            "campaign": context.campaign_slug,
            "campaign_dir": str(context.campaign_dir),
            "promotion_metric_name": metric_name,
            "promotion_metric_mode": "max",
            "promotion_strategy": promotion.strategy,
            "current_best_value": metric_value,
            "current_best_run_id": context.run_id,
            "updated_at": now,
        },
        index_record={
            "campaign": context.campaign_slug,
            "run_id": context.run_id,
            "update": update,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "checkpoint_path": str(checkpoint_path),
            "promotion_strategy": promotion.strategy,
            "updated_at": now,
        },
    )

    return PromotionAttempt(
        promoted=True,
        reason="unified_ladder_promoted",
        metric_name=metric_name,
        metric_value=metric_value,
        metric_mode="max",
        promoted_manifest_path=manifest_out,
    )
