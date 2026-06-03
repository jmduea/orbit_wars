"""Tests for qualifier ladder at combined 1.0 floors."""

from __future__ import annotations

from dataclasses import dataclass

from src.artifacts.tournament.bracket.qualifier import (
    QUALIFIER_FLOOR,
    evaluate_qualifier_scores,
    qualifier_floors,
)


@dataclass
class _Row:
    opponent: str
    win_rate_2p: float | None
    win_rate_4p: float | None
    combined: float | None
    passed: bool = False
    fail_reason: str | None = None


def test_qualifier_floors_are_one() -> None:
    floors = qualifier_floors()
    assert floors["noop"] == QUALIFIER_FLOOR == 1.0
    assert floors["nearest_sniper"] == 1.0


def test_all_opponents_at_one_clears() -> None:
    rows = (
        _Row("noop", 1.0, 1.0, 1.0, True),
        _Row("random", 1.0, 1.0, 1.0, True),
        _Row("nearest_sniper", 1.0, 1.0, 1.0, True),
    )
    verdict = evaluate_qualifier_scores(rows)
    assert verdict.cleared is True
    assert verdict.crown_incumbent is True


def test_any_opponent_below_one_fails() -> None:
    rows = (
        _Row("noop", 1.0, 1.0, 1.0, True),
        _Row("random", 0.9, 1.0, 0.95, False),
        _Row("nearest_sniper", 1.0, 1.0, 1.0, True),
    )
    verdict = evaluate_qualifier_scores(rows)
    assert verdict.cleared is False
    assert verdict.fail_reason == "failed_qualifier_random"


def test_incumbent_crowned_skips_nearest_sniper_requirement() -> None:
    rows = (
        _Row("noop", 1.0, 1.0, 1.0, True),
        _Row("random", 1.0, 1.0, 1.0, True),
    )
    verdict = evaluate_qualifier_scores(rows, incumbent_crowned=True)
    assert verdict.cleared is True
    assert verdict.crown_incumbent is False
