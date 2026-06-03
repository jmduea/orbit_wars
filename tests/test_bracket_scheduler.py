"""Tests for main-bracket round-robin scheduling and rating updates."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.tournament.bracket.scheduler import (
    apply_head_to_head_outcome,
    iter_round_robin_pairs,
    queue_round_robin_matches,
)
from src.artifacts.tournament.bracket.state import BracketEntry, BracketState


def test_iter_round_robin_pairs_three_entries() -> None:
    entries = (
        BracketEntry(agent_id="a", checkpoint_path="/a.pkl", qualifier_cleared=True),
        BracketEntry(agent_id="b", checkpoint_path="/b.pkl", qualifier_cleared=True),
        BracketEntry(agent_id="c", checkpoint_path="/c.pkl", qualifier_cleared=True),
    )
    pairs = iter_round_robin_pairs(entries)
    assert pairs == (("a", "b"), ("a", "c"), ("b", "c"))


def test_apply_head_to_head_outcome_win_increases_winner_mu() -> None:
    state = BracketState(
        phase="main",
        entries={
            "a": BracketEntry(agent_id="a", checkpoint_path="/a.pkl", mu=25.0, sigma=8.0),
            "b": BracketEntry(agent_id="b", checkpoint_path="/b.pkl", mu=20.0, sigma=8.0),
        },
    )
    apply_head_to_head_outcome(state, agent_a="a", agent_b="b", outcome="win")
    assert state.entries["a"].mu > 25.0
    assert state.entries["b"].mu < 20.0


def test_queue_round_robin_matches_writes_jobs(tmp_path: Path) -> None:
    state = BracketState(
        phase="main",
        entries={
            "a": BracketEntry(
                agent_id="a",
                checkpoint_path=str(tmp_path / "a.pkl"),
                qualifier_cleared=True,
            ),
            "b": BracketEntry(
                agent_id="b",
                checkpoint_path=str(tmp_path / "b.pkl"),
                qualifier_cleared=True,
            ),
        },
    )
    (tmp_path / "a.pkl").write_bytes(b"a")
    (tmp_path / "b.pkl").write_bytes(b"b")
    queue_dir = tmp_path / "queue"
    jobs = queue_round_robin_matches(
        queue_dir,
        state=state,
        update=10,
        result_root=tmp_path / "evaluations",
        campaign="demo",
        output_root=tmp_path,
    )
    assert len(jobs) == 1
    assert jobs[0].name.startswith("bracket_match_u000010_")
    assert state.round_robin_queued is True
    assert queue_round_robin_matches(
        queue_dir,
        state=state,
        update=20,
        campaign="demo",
        output_root=tmp_path,
    ) == []
