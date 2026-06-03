"""Tests for bracket state persistence."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.tournament.bracket.state import (
    BracketEntry,
    BracketState,
    load_bracket_state,
    mark_qualifier_cleared,
    mark_weak_config,
    save_bracket_state,
    upsert_entry,
)


def test_round_trip_save_load(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = BracketState(
        entries={
            "a1": BracketEntry(
                agent_id="a1",
                checkpoint_path="/tmp/ckpt.pkl",
                mu=26.0,
                sigma=7.0,
            )
        }
    )
    save_bracket_state(path, state)
    loaded = load_bracket_state(path)
    assert loaded.entries["a1"].mu == 26.0
    assert loaded.entries["a1"].sigma == 7.0


def test_empty_path_returns_default_state(tmp_path: Path) -> None:
    state = load_bracket_state(tmp_path / "missing.json")
    assert state.phase == "qualifier"
    assert state.entries == {}


def test_mark_qualifier_cleared_transitions_to_main() -> None:
    state = BracketState(
        entries={"a1": BracketEntry(agent_id="a1", checkpoint_path="/c.pkl")}
    )
    mark_qualifier_cleared(state, agent_id="a1", crown_incumbent=True)
    assert state.phase == "main"
    assert state.incumbent_crowned is True
    assert state.entries["a1"].qualifier_cleared is True


def test_mark_weak_config() -> None:
    state = BracketState()
    mark_weak_config(state)
    assert state.phase == "weak_config"


def test_upsert_entry() -> None:
    state = BracketState()
    upsert_entry(
        state,
        BracketEntry(agent_id="x", checkpoint_path="/x.pkl", qualifier_cleared=True),
    )
    assert state.main_phase_entries()[0].checkpoint_path == "/x.pkl"
