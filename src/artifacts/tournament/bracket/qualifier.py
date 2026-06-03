"""Qualifier ladder evaluation at combined 1.0 floors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUALIFIER_OPPONENTS = ("noop", "random", "nearest_sniper")
QUALIFIER_FLOOR = 1.0


@dataclass(frozen=True, slots=True)
class QualifierOpponentScore:
    opponent: str
    win_rate_2p: float | None
    win_rate_4p: float | None
    combined: float | None
    passed: bool
    fail_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QualifierVerdict:
    cleared: bool
    opponents: tuple[QualifierOpponentScore, ...]
    fail_reason: str | None = None
    crown_incumbent: bool = False


def qualifier_floors() -> dict[str, float]:
    return {opponent: QUALIFIER_FLOOR for opponent in QUALIFIER_OPPONENTS}


def _score_from_row(row: Any) -> QualifierOpponentScore:
    combined = getattr(row, "combined", None)
    passed = combined is not None and combined >= QUALIFIER_FLOOR
    fail_reason = None
    if combined is None:
        fail_reason = getattr(row, "fail_reason", None) or "missing_combined"
    elif not passed:
        fail_reason = f"below_floor_{QUALIFIER_FLOOR}"
    return QualifierOpponentScore(
        opponent=str(getattr(row, "opponent", "")),
        win_rate_2p=getattr(row, "win_rate_2p", None),
        win_rate_4p=getattr(row, "win_rate_4p", None),
        combined=combined,
        passed=passed,
        fail_reason=fail_reason,
    )


def evaluate_qualifier_scores(
    opponent_rows: tuple[Any, ...],
    *,
    incumbent_crowned: bool = False,
) -> QualifierVerdict:
    """Evaluate qualifier clearance from unified scoring rows.

    Expects rows for noop, random, and (unless incumbent crowned) nearest_sniper
    with ``combined`` scores on the unified 2p+4p metric.
    """

    scores = tuple(_score_from_row(row) for row in opponent_rows)
    required = QUALIFIER_OPPONENTS if not incumbent_crowned else ("noop", "random")
    by_opponent = {score.opponent: score for score in scores}

    for opponent in required:
        row = by_opponent.get(opponent)
        if row is None:
            return QualifierVerdict(
                cleared=False,
                opponents=scores,
                fail_reason=f"missing_opponent_{opponent}",
            )
        if not row.passed:
            return QualifierVerdict(
                cleared=False,
                opponents=scores,
                fail_reason=f"failed_qualifier_{opponent}",
            )

    crown = not incumbent_crowned and all(
        by_opponent.get("nearest_sniper") is not None
        and by_opponent["nearest_sniper"].passed
        for _ in [0]
    )
    if not incumbent_crowned:
        sniper_row = by_opponent.get("nearest_sniper")
        if sniper_row is None:
            return QualifierVerdict(
                cleared=False,
                opponents=scores,
                fail_reason="missing_opponent_nearest_sniper",
            )
        if not sniper_row.passed:
            return QualifierVerdict(
                cleared=False,
                opponents=scores,
                fail_reason="failed_qualifier_nearest_sniper",
            )
        crown = True

    return QualifierVerdict(
        cleared=True,
        opponents=scores,
        crown_incumbent=crown,
    )
