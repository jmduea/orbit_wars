"""Shared win-rate helpers for tournament aggregation modules."""

from __future__ import annotations


def win_rate(wins: int, games: int) -> float | None:
    """Return ``wins / games`` when ``games > 0``."""

    if games <= 0:
        return None
    return float(wins) / float(games)
