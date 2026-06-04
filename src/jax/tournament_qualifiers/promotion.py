"""Stage promotion floors and rollout opponent mix for SSOT qualifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp

from src.jax.qualifier_calibration import (
    QualifierCalibration,
    legs_for_stage,
    load_qualifier_calibration,
)
from src.jax.tournament_qualifiers.metrics import win_fraction
from src.opponents.constants import (
    OPPONENT_FAMILY_NAMES,
    OPPONENT_NEAREST_SNIPER,
    OPPONENT_NOOP,
    OPPONENT_RANDOM,
)
from src.opponents.pool import OPPONENT_FAMILY_IDS
from src.training.curriculum import StageView

QualifierStage = Literal[1, 2, 3, 4]

STAGE_NAMES: dict[int, str] = {
    1: "random",
    2: "noop_heavy",
    3: "sniper_heavy",
    4: "main_bracket",
}


@dataclass(frozen=True, slots=True)
class LegWinSummary:
    opponent: str
    wins: int
    games: int
    win_rate: float | None


@dataclass(frozen=True, slots=True)
class StagePromotionVerdict:
    stage: int
    promoted: bool
    next_stage: int
    leg_summaries: tuple[LegWinSummary, ...]
    fail_reason: str | None = None
    enter_main_bracket: bool = False


def _family_probability_vector(
    *,
    random: float,
    noop: float,
    nearest_sniper: float,
) -> tuple[float, ...]:
    probs = [0.0] * len(OPPONENT_FAMILY_NAMES)
    probs[OPPONENT_RANDOM] = random
    probs[OPPONENT_NOOP] = noop
    probs[OPPONENT_NEAREST_SNIPER] = nearest_sniper
    total = sum(probs)
    if total <= 0.0:
        probs[OPPONENT_RANDOM] = 1.0
        total = 1.0
    return tuple(p / total for p in probs)


def ssot_rollout_stage_view(
    stage: int,
    update: int,
    *,
    snapshot_ids: Any,
    snapshot_valid_mask: Any,
    snapshot_updates: Any,
) -> StageView:
    """Rollout opponent mixture from persisted SSOT qualifier stage (R15–R17)."""

    probs = opponent_family_probs_for_stage(stage)
    family_probs = jnp.asarray(probs, dtype=jnp.float32)
    valid_mask = jnp.asarray(snapshot_valid_mask, dtype=bool)
    ids = jnp.asarray(snapshot_ids, dtype=jnp.int32)
    updates = jnp.asarray(snapshot_updates, dtype=jnp.int32)
    ages = jnp.where(valid_mask, jnp.asarray(update, dtype=jnp.int32) - updates, 0)
    historical_probs = valid_mask.astype(jnp.float32)
    total = historical_probs.sum()
    historical_probs = jnp.where(
        total > 0.0,
        historical_probs / jnp.maximum(total, 1.0),
        jnp.zeros_like(historical_probs),
    )
    return StageView(
        stage_index=jnp.asarray(max(stage - 1, 0), dtype=jnp.int32),
        family_ids=OPPONENT_FAMILY_IDS,
        family_probs=family_probs,
        family_mask=family_probs > 0.0,
        snapshot_pool_ids=ids,
        snapshot_valid_mask=valid_mask,
        snapshot_age_updates=ages,
        historical_selection_probs=historical_probs,
        fallback_family_id=jnp.asarray(OPPONENT_RANDOM, dtype=jnp.int32),
    )


def opponent_family_probs_for_stage(stage: int) -> tuple[float, ...]:
    """Rollout opponent mixture by qualifier stage (R15–R17)."""

    if stage <= 1:
        return _family_probability_vector(random=0.85, noop=0.10, nearest_sniper=0.05)
    if stage == 2:
        return _family_probability_vector(random=0.20, noop=0.65, nearest_sniper=0.15)
    if stage == 3:
        return _family_probability_vector(random=0.15, noop=0.20, nearest_sniper=0.65)
    return _family_probability_vector(random=0.10, noop=0.10, nearest_sniper=0.80)


def evaluate_stage_promotion(
    *,
    stage: int,
    leg_wins: dict[str, tuple[int, int]],
    calibration: QualifierCalibration | None = None,
) -> StagePromotionVerdict:
    """Compare per-leg win rates against calibration or interim conservative floors."""

    cal = calibration or load_qualifier_calibration()
    required_legs = legs_for_stage(stage)
    if not required_legs:
        return StagePromotionVerdict(
            stage=stage,
            promoted=False,
            next_stage=stage,
            leg_summaries=(),
            fail_reason="unknown_stage",
        )
    summaries: list[LegWinSummary] = []
    for leg in required_legs:
        wins, games = leg_wins.get(leg, (0, 0))
        summaries.append(
            LegWinSummary(
                opponent=leg,
                wins=int(wins),
                games=int(games),
                win_rate=win_fraction(int(wins), int(games)),
            )
        )
    for summary in summaries:
        if summary.win_rate is None:
            return StagePromotionVerdict(
                stage=stage,
                promoted=False,
                next_stage=stage,
                leg_summaries=tuple(summaries),
                fail_reason=f"no_games_{summary.opponent}",
            )
        floor = cal.min_win_rate_for(stage, summary.opponent)
        if summary.win_rate + 1e-9 < floor:
            return StagePromotionVerdict(
                stage=stage,
                promoted=False,
                next_stage=stage,
                leg_summaries=tuple(summaries),
                fail_reason=f"below_floor_{summary.opponent}",
            )
    next_stage = stage + 1
    enter_main = stage >= 3
    if enter_main:
        next_stage = 4
    return StagePromotionVerdict(
        stage=stage,
        promoted=True,
        next_stage=next_stage,
        leg_summaries=tuple(summaries),
        enter_main_bracket=enter_main,
    )
