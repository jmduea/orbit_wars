"""Main-bracket round-robin scheduling and rating updates."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Literal

from src.artifacts.pipeline import write_optional_job
from src.artifacts.tournament.bracket.state import BracketEntry, BracketState
from src.artifacts.tournament.bracket.trueskill import Rating, update_draw, update_win

MatchOutcomeLabel = Literal["win", "loss", "draw"]


def iter_round_robin_pairs(
    entries: tuple[BracketEntry, ...],
) -> tuple[tuple[str, str], ...]:
    """Return unordered agent_id pairs for a full round-robin among entries."""

    agent_ids = tuple(entry.agent_id for entry in entries)
    return tuple(
        (left, right)
        for left, right in itertools.combinations(sorted(agent_ids), 2)
    )


def apply_head_to_head_outcome(
    state: BracketState,
    *,
    agent_a: str,
    agent_b: str,
    outcome: MatchOutcomeLabel,
) -> BracketState:
    """Apply a 2p match outcome to bracket entry μ/σ (margin-independent)."""

    entry_a = state.entries.get(agent_a)
    entry_b = state.entries.get(agent_b)
    if entry_a is None or entry_b is None:
        return state

    rating_a = entry_a.rating()
    rating_b = entry_b.rating()
    if outcome == "draw":
        new_a, new_b = update_draw(rating_a, rating_b)
    elif outcome == "win":
        new_a, new_b = update_win(rating_a, rating_b)
    else:
        new_b, new_a = update_win(rating_b, rating_a)

    entry_a.mu, entry_a.sigma = new_a.mu, new_a.sigma
    entry_b.mu, entry_b.sigma = new_b.mu, new_b.sigma
    return state


def outcome_from_rewards(
    agent_a: str,
    agent_b: str,
    rewards: dict[str, float],
) -> MatchOutcomeLabel:
    """Map Kaggle env rewards to win/loss/draw for bracket rating updates."""

    reward_a = rewards.get(agent_a, 0.0)
    reward_b = rewards.get(agent_b, 0.0)
    if reward_a > reward_b:
        return "win"
    if reward_b > reward_a:
        return "loss"
    return "draw"


def queue_round_robin_matches(
    queue_dir: Path,
    *,
    state: BracketState,
    update: int,
    result_root: Path | None = None,
    campaign: str,
    output_root: Path,
) -> list[Path]:
    """Enqueue ``bracket_match`` jobs for all main-phase entry pairs (once per campaign)."""

    if state.round_robin_queued:
        return []

    entries = state.main_phase_entries()
    if len(entries) < 2:
        return []

    queued: list[Path] = []
    for agent_a, agent_b in iter_round_robin_pairs(entries):
        entry_a = state.entries[agent_a]
        entry_b = state.entries[agent_b]
        job_path = write_optional_job(
            queue_dir,
            kind="bracket_match",
            update=update,
            checkpoint_path=Path(entry_a.checkpoint_path),
            payload={
                "campaign": campaign,
                "output_root": str(output_root.resolve()),
                "agent_a": agent_a,
                "agent_b": agent_b,
                "checkpoint_path_a": entry_a.checkpoint_path,
                "checkpoint_path_b": entry_b.checkpoint_path,
            },
            result_root=result_root,
        )
        queued.append(job_path)
    if queued:
        state.round_robin_queued = True
    return queued
