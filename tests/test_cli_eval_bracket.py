"""Tests for ``ow eval bracket`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

from src.artifacts.tournament.bracket.state import BracketEntry, BracketState, save_bracket_state
from src.cli import eval as eval_cli


def test_bracket_status_missing_state(tmp_path: Path) -> None:
    args = eval_cli.build_parser().parse_args(
        ["bracket", "status", "--campaign", "c1", "--output-root", str(tmp_path)]
    )
    assert eval_cli.run_bracket_cli(args) == 0


def test_bracket_show_round_trip(tmp_path: Path, capsys) -> None:
    state_path = tmp_path / "campaigns" / "c1" / "bracket" / "state.json"
    state = BracketState(
        phase="qualifier",
        entries={"u1": BracketEntry(agent_id="u1", checkpoint_path="/x.pkl")},
    )
    save_bracket_state(state_path, state)
    args = eval_cli.build_parser().parse_args(
        ["bracket", "show", "--campaign", "c1", "--output-root", str(tmp_path)]
    )
    assert eval_cli.run_bracket_cli(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"]["phase"] == "qualifier"
    assert "u1" in payload["state"]["entries"]
