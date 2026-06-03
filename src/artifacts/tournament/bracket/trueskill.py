"""Kaggle-style TrueSkill μ/σ rating updates (margin-independent)."""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25.0 / 3.0
DEFAULT_BETA = 25.0 / 6.0


@dataclass(frozen=True, slots=True)
class Rating:
    """Skill belief for one bracket entry."""

    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _v_win(t: float, epsilon: float = 0.0) -> float:
    """TrueSkill v function for win (with draw margin epsilon=0)."""

    denom = _cdf(t - epsilon)
    if denom < 1e-12:
        return -t
    return _pdf(t - epsilon) / denom


def _w_win(t: float, epsilon: float = 0.0) -> float:
    v = _v_win(t, epsilon)
    return v * (v + t - epsilon)


def _v_draw(t: float, epsilon: float) -> float:
    denom = _cdf(epsilon - t) - _cdf(-epsilon - t)
    if abs(denom) < 1e-12:
        return 0.0
    return (_pdf(-epsilon - t) - _pdf(epsilon - t)) / denom


def _w_draw(t: float, epsilon: float) -> float:
    v = _v_draw(t, epsilon)
    return v * v + (
        (epsilon - t) * _pdf(epsilon - t) + (epsilon + t) * _pdf(-epsilon - t)
    ) / (_cdf(epsilon - t) - _cdf(-epsilon - t) + 1e-12)


def _pair_update(
    winner: Rating,
    loser: Rating,
    *,
    outcome: str,
    beta: float = DEFAULT_BETA,
    draw_epsilon: float = 0.0,
) -> tuple[Rating, Rating]:
    """Update a pair of ratings from a single match outcome.

    ``outcome`` is ``win`` (first player wins), ``loss`` (second wins), or
    ``draw``. Episode score margin is intentionally ignored.
    """

    if outcome == "loss":
        updated_loser, updated_winner = _pair_update(
            loser, winner, outcome="win", beta=beta, draw_epsilon=draw_epsilon
        )
        return updated_winner, updated_loser

    sigma1_sq = winner.sigma * winner.sigma
    sigma2_sq = loser.sigma * loser.sigma
    c = math.sqrt(2.0 * beta * beta + sigma1_sq + sigma2_sq)
    t = (winner.mu - loser.mu) / c

    if outcome == "draw":
        v = _v_draw(t, draw_epsilon)
        w = _w_draw(t, draw_epsilon)
    elif outcome == "win":
        v = _v_win(t, draw_epsilon)
        w = _w_win(t, draw_epsilon)
    else:
        raise ValueError(f"Unknown outcome: {outcome!r}")

    winner_sigma_sq = sigma1_sq
    loser_sigma_sq = sigma2_sq
    winner_mu = winner.mu + (winner_sigma_sq / c) * v
    loser_mu = loser.mu - (loser_sigma_sq / c) * v
    winner_sigma = math.sqrt(
        max(winner_sigma_sq * (1.0 - (winner_sigma_sq / (c * c)) * w), 1e-9)
    )
    loser_sigma = math.sqrt(
        max(loser_sigma_sq * (1.0 - (loser_sigma_sq / (c * c)) * w), 1e-9)
    )
    return Rating(mu=winner_mu, sigma=winner_sigma), Rating(
        mu=loser_mu, sigma=loser_sigma
    )


def update_win(winner: Rating, loser: Rating, *, beta: float = DEFAULT_BETA) -> tuple[Rating, Rating]:
    """Apply a win/loss update; margin does not affect magnitude."""

    return _pair_update(winner, loser, outcome="win", beta=beta)


def update_draw(
    left: Rating,
    right: Rating,
    *,
    beta: float = DEFAULT_BETA,
    draw_epsilon: float = 0.0,
) -> tuple[Rating, Rating]:
    """Apply a draw update; both μ values move toward each other."""

    return _pair_update(left, right, outcome="draw", beta=beta, draw_epsilon=draw_epsilon)
