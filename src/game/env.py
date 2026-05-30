"""Shared game reward helpers (Python env removed; JAX env is canonical)."""

from __future__ import annotations

from src.game.rewards import (
    apply_early_terminal_reward_shaping,
    normalized_ship_differential_reward,
    terminal_reward_from_scores,
)

__all__ = [
    "apply_early_terminal_reward_shaping",
    "normalized_ship_differential_reward",
    "terminal_reward_from_scores",
]
