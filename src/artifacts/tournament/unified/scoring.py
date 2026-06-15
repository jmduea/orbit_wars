"""Combined 2p+4p scoring for unified tournament stages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.artifacts.tournament.runner import challenger_won_2p
from src.artifacts.tournament.types import MatchOutcome

FOUR_P_FORMATS = frozenset({"4p_free_for_all", "4p_challenger_vs_baselines"})


@dataclass(frozen=True, slots=True)
class UnifiedOpponentScore:
    """Per-opponent unified tournament rates."""

    opponent: str
    win_rate_2p: float | None
    win_rate_4p: float | None
    combined: float | None
    passed: bool = False
    fail_reason: str | None = None


def _win_rate(wins: int, games: int) -> float | None:
    if games <= 0:
        return None
    return wins / games


def combined_score(
    win_rate_2p: float | None,
    win_rate_4p: float | None,
) -> tuple[float | None, str | None]:
    """Return combined score or None with fail-closed reason when a leg is missing."""

    if win_rate_2p is None:
        return None, "missing_2p_games"
    if win_rate_4p is None:
        return None, "missing_4p_games"
    return 0.5 * win_rate_2p + 0.5 * win_rate_4p, None


def aggregate_format_win_rate(
    outcomes: tuple[MatchOutcome, ...],
    *,
    challenger_id: str,
    format_name: str,
    opponent_filter: str | None = None,
) -> float | None:
    """Aggregate win or first-place rate for one format leg."""

    wins = 0
    games = 0
    for outcome in outcomes:
        if outcome.format_name != format_name:
            continue
        if challenger_id not in outcome.agent_ids:
            continue
        if opponent_filter is not None:
            if format_name == "2p_vs_baseline":
                baseline_id = f"baseline:{opponent_filter}"
                if baseline_id not in outcome.agent_ids:
                    continue
            elif format_name == "2p_head_to_head":
                if opponent_filter not in outcome.agent_ids:
                    continue
        games += 1
        if format_name in FOUR_P_FORMATS:
            if outcome.placements.get(challenger_id) == 1:
                wins += 1
        elif challenger_won_2p(outcome, challenger_id):
            wins += 1
    return _win_rate(wins, games)


def score_opponent(
    outcomes: tuple[MatchOutcome, ...],
    *,
    challenger_id: str,
    opponent: str,
    floor: float | None = None,
) -> UnifiedOpponentScore:
    """Compute combined score for one Stage 1 baseline opponent."""

    win_rate_2p = aggregate_format_win_rate(
        outcomes,
        challenger_id=challenger_id,
        format_name="2p_vs_baseline",
        opponent_filter=opponent,
    )
    win_rate_4p = aggregate_format_win_rate(
        outcomes,
        challenger_id=challenger_id,
        format_name="4p_challenger_vs_baselines",
    )
    combined, reason = combined_score(win_rate_2p, win_rate_4p)
    passed = False
    if combined is not None and floor is not None:
        passed = combined >= floor
    elif combined is not None and floor is None:
        passed = True
    return UnifiedOpponentScore(
        opponent=opponent,
        win_rate_2p=win_rate_2p,
        win_rate_4p=win_rate_4p,
        combined=combined,
        passed=passed,
        fail_reason=reason if combined is None else None,
    )


def per_seed_combined_scores(
    outcomes: tuple[MatchOutcome, ...],
    *,
    challenger_id: str,
    seeds: tuple[int, ...],
) -> list[float | None]:
    """Per-seed combined scores for Stage 2 incumbent evaluation."""

    scores: list[float | None] = []
    for seed in seeds:
        seed_outcomes = tuple(item for item in outcomes if item.seed == seed)
        win_rate_2p = aggregate_format_win_rate(
            seed_outcomes,
            challenger_id=challenger_id,
            format_name="2p_head_to_head",
            opponent_filter="incumbent",
        )
        win_rate_4p = aggregate_format_win_rate(
            seed_outcomes,
            challenger_id=challenger_id,
            format_name="4p_challenger_vs_baselines",
        )
        combined, _ = combined_score(win_rate_2p, win_rate_4p)
        scores.append(combined)
    return scores


def all_seeds_perfect(per_seed: list[float | None]) -> bool:
    """Stage 2 pass requires every seed combined score == 1.0."""

    if not per_seed:
        return False
    return all(score is not None and score >= 1.0 - 1e-9 for score in per_seed)


def stage2_per_seed_summary(
    outcomes: tuple[MatchOutcome, ...],
    *,
    challenger_id: str,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    per_seed = per_seed_combined_scores(
        outcomes, challenger_id=challenger_id, seeds=seeds
    )
    return {
        "per_seed_combined": per_seed,
        "all_seeds_perfect": all_seeds_perfect(per_seed),
    }


def group_outcomes_by_opponent(
    outcomes: tuple[MatchOutcome, ...],
    *,
    challenger_id: str,
    opponents: tuple[str, ...],
) -> dict[str, tuple[MatchOutcome, ...]]:
    grouped: dict[str, list[MatchOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if challenger_id not in outcome.agent_ids:
            continue
        if outcome.format_name == "2p_vs_baseline":
            for opponent in opponents:
                if f"baseline:{opponent}" in outcome.agent_ids:
                    grouped[opponent].append(outcome)
                    break
        elif outcome.format_name == "4p_challenger_vs_baselines":
            for opponent in opponents:
                grouped[opponent].append(outcome)
    return {key: tuple(values) for key, values in grouped.items()}
