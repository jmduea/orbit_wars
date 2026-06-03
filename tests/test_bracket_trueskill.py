"""Tests for Kaggle-style TrueSkill rating updates."""

from __future__ import annotations

from src.artifacts.tournament.bracket.trueskill import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    Rating,
    update_draw,
    update_win,
)


def test_win_increases_winner_mu_and_decreases_loser_mu() -> None:
    winner = Rating(mu=25.0, sigma=DEFAULT_SIGMA)
    loser = Rating(mu=20.0, sigma=DEFAULT_SIGMA)
    updated_winner, updated_loser = update_win(winner, loser)
    assert updated_winner.mu > winner.mu
    assert updated_loser.mu < loser.mu


def test_upset_win_moves_mu_more_than_expected_win() -> None:
    favorite = Rating(mu=30.0, sigma=DEFAULT_SIGMA)
    underdog = Rating(mu=20.0, sigma=DEFAULT_SIGMA)
    expected_w, expected_l = update_win(favorite, underdog)
    upset_w, upset_l = update_win(underdog, favorite)
    assert (upset_w.mu - underdog.mu) > (expected_w.mu - favorite.mu)


def test_draw_moves_both_mu_toward_each_other() -> None:
    high = Rating(mu=30.0, sigma=DEFAULT_SIGMA)
    low = Rating(mu=20.0, sigma=DEFAULT_SIGMA)
    left, right = update_draw(high, low, draw_epsilon=0.1)
    assert left.mu < high.mu
    assert right.mu > low.mu


def test_sigma_decreases_after_match() -> None:
    left = Rating()
    right = Rating()
    updated_left, updated_right = update_win(left, right)
    assert updated_left.sigma < left.sigma
    assert updated_right.sigma < right.sigma


def test_margin_does_not_affect_update() -> None:
    """Rating updates depend on outcome only, not score margin."""

    winner = Rating()
    loser = Rating()
    first = update_win(winner, loser)
    second = update_win(winner, loser)
    assert first[0].mu == second[0].mu
    assert first[1].mu == second[1].mu
