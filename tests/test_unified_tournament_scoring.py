from __future__ import annotations

import pytest

from src.artifacts.tournament.types import MatchOutcome
from src.artifacts.tournament.unified.scoring import (
    all_seeds_perfect,
    combined_score,
    per_seed_combined_scores,
    score_opponent,
)


def _outcome(
    *,
    format_name: str,
    seed: int,
    challenger: str = "cand",
    opponent: str | None = None,
    win: bool = True,
    first_place: bool = True,
) -> MatchOutcome:
    if format_name == "2p_vs_baseline":
        baseline_id = f"baseline:{opponent or 'noop'}"
        return MatchOutcome(
            match_id=f"{format_name}_{seed}",
            format_name=format_name,
            seed=seed,
            agent_ids=(challenger, baseline_id),
            rewards={
                challenger: 1.0 if win else -1.0,
                baseline_id: -1.0 if win else 1.0,
            },
            results={
                challenger: "win" if win else "loss",
                baseline_id: "loss" if win else "win",
            },
        )
    if format_name == "2p_head_to_head":
        return MatchOutcome(
            match_id=f"{format_name}_{seed}",
            format_name=format_name,
            seed=seed,
            agent_ids=(challenger, "incumbent"),
            rewards={
                challenger: 1.0 if win else -1.0,
                "incumbent": -1.0 if win else 1.0,
            },
            results={
                challenger: "win" if win else "loss",
                "incumbent": "loss" if win else "win",
            },
        )
    return MatchOutcome(
        match_id=f"{format_name}_{seed}",
        format_name=format_name,
        seed=seed,
        agent_ids=(challenger, "baseline:noop", "baseline:random", "baseline:random"),
        rewards={challenger: 1.0 if first_place else 0.0},
        results={challenger: "win" if first_place else "loss"},
        placements={challenger: 1 if first_place else 2},
    )


def test_combined_score_happy_path() -> None:
    combined, reason = combined_score(0.80, 0.60)
    assert combined == pytest.approx(0.70)
    assert reason is None


def test_combined_score_missing_4p() -> None:
    combined, reason = combined_score(0.80, None)
    assert combined is None
    assert reason == "missing_4p_games"


def test_combined_score_missing_2p() -> None:
    combined, reason = combined_score(None, 0.60)
    assert combined is None
    assert reason == "missing_2p_games"


def test_score_opponent_combined_floor() -> None:
    outcomes = (
        _outcome(format_name="2p_vs_baseline", seed=0, opponent="noop", win=True),
        _outcome(format_name="2p_vs_baseline", seed=1, opponent="noop", win=True),
        _outcome(format_name="4p_challenger_vs_baselines", seed=0, first_place=True),
        _outcome(format_name="4p_challenger_vs_baselines", seed=1, first_place=False),
    )
    row = score_opponent(outcomes, challenger_id="cand", opponent="noop", floor=0.7)
    assert row.win_rate_2p == 1.0
    assert row.win_rate_4p == 0.5
    assert row.combined == pytest.approx(0.75)
    assert row.passed


def test_score_opponent_counts_mutual_win_as_not_win() -> None:
    tie = MatchOutcome(
        match_id="tie",
        format_name="2p_vs_baseline",
        seed=0,
        agent_ids=("cand", "baseline:noop"),
        rewards={"cand": 1.0, "baseline:noop": 1.0},
        results={"cand": "win", "baseline:noop": "win"},
    )
    row = score_opponent((tie,), challenger_id="cand", opponent="noop", floor=0.0)
    assert row.win_rate_2p == 0.0


def test_all_seeds_perfect_requires_every_seed() -> None:
    assert all_seeds_perfect([1.0] * 29 + [0.967]) is False
    assert all_seeds_perfect([1.0] * 30) is True


def test_per_seed_combined_stage2() -> None:
    outcomes = (
        _outcome(format_name="2p_head_to_head", seed=0, win=True),
        _outcome(format_name="4p_challenger_vs_baselines", seed=0, first_place=True),
        _outcome(format_name="2p_head_to_head", seed=1, win=True),
        _outcome(format_name="4p_challenger_vs_baselines", seed=1, first_place=False),
    )
    per_seed = per_seed_combined_scores(outcomes, challenger_id="cand", seeds=(0, 1))
    assert per_seed[0] == pytest.approx(1.0)
    assert per_seed[1] == pytest.approx(0.5)
