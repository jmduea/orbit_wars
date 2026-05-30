"""Aggregate match outcomes into leaderboard rows and promotion gates."""

from __future__ import annotations

from collections import defaultdict

from src.config.schema import PromotionTournamentConfig

from .types import AgentEntry, LeaderboardRow, MatchOutcome


def _win_rate(wins: int, games: int) -> float | None:
    if games <= 0:
        return None
    return wins / games


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def aggregate_pairwise_win_rates(
    outcomes: tuple[MatchOutcome, ...],
) -> dict[str, dict[str, float]]:
    wins: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    games: dict[tuple[str, str], int] = defaultdict(int)

    for outcome in outcomes:
        if len(outcome.agent_ids) != 2:
            continue
        left, right = outcome.agent_ids
        key = _pair_key(left, right)
        games[key] += 1
        if outcome.results.get(left) == "win":
            wins[key][left] += 1
        elif outcome.results.get(right) == "win":
            wins[key][right] += 1

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for key, total in games.items():
        left, right = key
        left_rate = _win_rate(wins[key][left], total)
        right_rate = _win_rate(wins[key][right], total)
        if left_rate is not None:
            matrix[left][right] = left_rate
        if right_rate is not None:
            matrix[right][left] = right_rate
    return dict(matrix)


def build_leaderboard(
    candidates: tuple[AgentEntry, ...],
    outcomes: tuple[MatchOutcome, ...],
    *,
    incumbent_id: str | None,
    baseline_name: str = "sniper",
) -> tuple[LeaderboardRow, ...]:
    """Summarize per-candidate win rates from tournament outcomes."""

    baseline_id = f"baseline:{baseline_name}"
    rows: list[LeaderboardRow] = []
    for candidate in candidates:
        vs_sniper_wins = 0
        vs_sniper_games = 0
        vs_incumbent_wins = 0
        vs_incumbent_games = 0
        first_places = 0
        four_player_games = 0
        games_played = 0

        for outcome in outcomes:
            if candidate.agent_id not in outcome.agent_ids:
                continue
            games_played += 1
            if outcome.format_name == "2p_vs_baseline":
                if baseline_id in outcome.agent_ids:
                    vs_sniper_games += 1
                    if outcome.results.get(candidate.agent_id) == "win":
                        vs_sniper_wins += 1
            elif outcome.format_name == "2p_head_to_head" and incumbent_id is not None:
                if incumbent_id in outcome.agent_ids:
                    vs_incumbent_games += 1
                    if outcome.results.get(candidate.agent_id) == "win":
                        vs_incumbent_wins += 1
            elif outcome.format_name == "4p_free_for_all":
                four_player_games += 1
                if outcome.placements.get(candidate.agent_id) == 1:
                    first_places += 1

        rows.append(
            LeaderboardRow(
                agent_id=candidate.agent_id,
                checkpoint_path=str(candidate.checkpoint_path),
                games_played=games_played,
                win_rate_vs_sniper=_win_rate(vs_sniper_wins, vs_sniper_games),
                win_rate_vs_incumbent=_win_rate(vs_incumbent_wins, vs_incumbent_games),
                first_place_rate_4p=_win_rate(first_places, four_player_games),
            )
        )

    rows.sort(
        key=lambda row: (
            row.win_rate_vs_sniper is not None,
            row.win_rate_vs_sniper or -1.0,
            row.win_rate_vs_incumbent or -1.0,
            row.first_place_rate_4p or -1.0,
            row.agent_id,
        ),
        reverse=True,
    )
    return tuple(rows)


def evaluate_gates(
    row: LeaderboardRow,
    gates: PromotionTournamentConfig,
    *,
    incumbent_present: bool,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a leaderboard row passes tournament promotion gates."""

    reasons: list[str] = []
    if row.win_rate_vs_sniper is None:
        reasons.append("missing_vs_sniper")
    elif row.win_rate_vs_sniper < gates.min_win_rate_vs_sniper:
        reasons.append("below_min_win_rate_vs_sniper")

    if gates.require_head_to_head and incumbent_present:
        if row.win_rate_vs_incumbent is None:
            reasons.append("missing_vs_incumbent")
        elif row.win_rate_vs_incumbent < gates.min_win_rate_vs_incumbent:
            reasons.append("below_min_win_rate_vs_incumbent")

    if gates.min_first_place_rate_4p is not None:
        if row.first_place_rate_4p is None:
            reasons.append("missing_4p_first_place")
        elif row.first_place_rate_4p < gates.min_first_place_rate_4p:
            reasons.append("below_min_first_place_rate_4p")

    passed = not reasons
    return passed, tuple(reasons)
