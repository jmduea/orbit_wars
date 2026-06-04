"""Final-score win metrics for SSOT tournament qualifiers (R18)."""

from __future__ import annotations

import numpy as np


def learner_won_from_final_scores(
    scores: np.ndarray,
    learner_player: int,
) -> bool:
    """True when learner has strictly highest final score (planet + fleet ships).

    Ties at the maximum do not count as wins (R19 interim conservative rule).
    """

    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return False
    learner = int(learner_player)
    if learner < 0 or learner >= arr.size:
        return False
    learner_score = float(arr[learner])
    if learner_score <= 0.0:
        return False
    best = float(np.max(arr))
    if learner_score < best:
        return False
    winners = np.flatnonzero(arr == best)
    return winners.size == 1 and int(winners[0]) == learner


def win_fraction(wins: int, games: int) -> float | None:
    if games <= 0:
        return None
    return float(wins) / float(games)
