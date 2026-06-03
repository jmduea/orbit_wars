"""Tests for bracket self-play opponent sampling."""

from __future__ import annotations

import random

from src.artifacts.tournament.bracket.self_play import sample_bracket_checkpoints
from src.artifacts.tournament.bracket.state import BracketEntry, BracketState


def test_empty_bracket_returns_empty() -> None:
    state = BracketState(phase="main")
    assert sample_bracket_checkpoints(state, count=2) == ()


def test_qualifier_phase_returns_empty() -> None:
    state = BracketState(
        phase="qualifier",
        entries={"a": BracketEntry(agent_id="a", checkpoint_path="/a.pkl", qualifier_cleared=True)},
    )
    assert sample_bracket_checkpoints(state, count=1) == ()


def test_main_phase_samples_qualified_paths() -> None:
    state = BracketState(
        phase="main",
        entries={
            "a": BracketEntry(
                agent_id="a",
                checkpoint_path="/a.pkl",
                qualifier_cleared=True,
            ),
            "b": BracketEntry(
                agent_id="b",
                checkpoint_path="/b.pkl",
                lineage_skip=True,
            ),
        },
    )
    paths = sample_bracket_checkpoints(state, count=2, rng=random.Random(0))
    assert set(paths) == {"/a.pkl", "/b.pkl"}
