from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifacts.tournament.types import AgentEntry, MatchOutcome
from src.artifacts.tournament.unified.ladder import run_unified_ladder
from src.artifacts.tournament.unified.spec import parse_unified_tournament_section
from src.config import TrainConfig


def _spec(**overrides: object):
    base = {
        "enforcement": False,
        "games_per_pair": 1,
        "prerequisite_seeds": [0],
        "incumbent_seeds": [0, 1],
        "noop_min_combined": 0.7,
        "random_min_combined": 0.58,
        "four_p_baseline_fillers": ["noop", "random", "random"],
        "incumbent_checkpoint_path": str(Path("/tmp/incumbent.pkl")),
    }
    base.update(overrides)
    return parse_unified_tournament_section(base)


def _mock_outcome(
    *,
    format_name: str,
    seed: int,
    opponent: str = "noop",
    win: bool = True,
    first_place: bool = True,
) -> MatchOutcome:
    if format_name == "2p_vs_baseline":
        baseline_id = f"baseline:{opponent}"
        return MatchOutcome(
            match_id="m",
            format_name=format_name,
            seed=seed,
            agent_ids=("cand", baseline_id),
            rewards={"cand": 1.0 if win else -1.0},
            results={"cand": "win" if win else "loss", baseline_id: "loss" if win else "win"},
        )
    if format_name == "2p_head_to_head":
        return MatchOutcome(
            match_id="m",
            format_name=format_name,
            seed=seed,
            agent_ids=("cand", "incumbent"),
            rewards={"cand": 1.0 if win else -1.0},
            results={"cand": "win" if win else "loss", "incumbent": "loss" if win else "win"},
        )
    return MatchOutcome(
        match_id="m",
        format_name=format_name,
        seed=seed,
        agent_ids=("cand", "baseline:noop", "baseline:random", "baseline:random"),
        rewards={"cand": 1.0},
        results={"cand": "win" if first_place else "loss"},
        placements={"cand": 1 if first_place else 2},
    )


def _run_match_side_effect(**kwargs: object) -> tuple[MatchOutcome, object, dict[str, object]]:
    format_name = str(kwargs["format_name"])
    seed = int(kwargs["seed"])
    agent_ids = kwargs["agent_ids"]
    opponent = "noop"
    if format_name == "2p_vs_baseline":
        for agent_id in agent_ids:  # type: ignore[union-attr]
            if str(agent_id).startswith("baseline:"):
                opponent = str(agent_id).split(":", 1)[1]
    win = opponent != "noop" or seed % 2 == 0
    if format_name == "2p_vs_baseline" and opponent == "noop":
        win = False  # force noop fail for AE1 test
    return (
        _mock_outcome(format_name=format_name, seed=seed, opponent=opponent, win=win),
        object(),
        {"match_seconds": 0.1, "env_steps": 1, "max_action_seconds": 0.01},
    )


@patch("src.artifacts.tournament.unified.ladder.agent_from_checkpoint")
def test_prerequisite_noop_fail_skips_stage2(mock_agent, tmp_path: Path) -> None:
    mock_agent.return_value = AgentEntry(
        agent_id="cand",
        checkpoint_path=tmp_path / "ckpt.pkl",
        cfg=TrainConfig(),
        act_fn=lambda _obs: [],
    )
    (tmp_path / "ckpt.pkl").write_bytes(b"x")

    def fail_noop(**kwargs: object):
        format_name = str(kwargs["format_name"])
        seed = int(kwargs["seed"])
        agent_ids = kwargs["agent_ids"]
        opponent = "noop"
        for agent_id in agent_ids:  # type: ignore[union-attr]
            if str(agent_id).startswith("baseline:"):
                opponent = str(agent_id).split(":", 1)[1]
        win = not (format_name == "2p_vs_baseline" and opponent == "noop")
        return (
            _mock_outcome(format_name=format_name, seed=seed, opponent=opponent, win=win),
            object(),
            {"match_seconds": 0.1, "env_steps": 1, "max_action_seconds": 0.01},
        )

    spec = _spec()
    out = tmp_path / "eval"
    verdict = run_unified_ladder(
        tmp_path / "ckpt.pkl",
        spec,
        out,
        run_match_fn=fail_noop,
    )
    assert not verdict.passed
    assert verdict.reason == "failed_prerequisite_noop"
    assert len(verdict.stages) == 1
    assert not (out / "stage2_incumbent").exists()


@patch("src.artifacts.tournament.unified.ladder.agent_from_checkpoint")
@patch("src.artifacts.tournament.unified.ladder.resolve_incumbent")
def test_no_incumbent_skips_stage2(mock_incumbent, mock_agent, tmp_path: Path) -> None:
    mock_incumbent.return_value = None
    mock_agent.return_value = AgentEntry(
        agent_id="cand",
        checkpoint_path=tmp_path / "ckpt.pkl",
        cfg=TrainConfig(),
        act_fn=lambda _obs: [],
    )
    (tmp_path / "ckpt.pkl").write_bytes(b"x")

    def always_win(**kwargs: object):
        format_name = str(kwargs["format_name"])
        seed = int(kwargs["seed"])
        agent_ids = kwargs["agent_ids"]
        opponent = "noop"
        for agent_id in agent_ids:  # type: ignore[union-attr]
            if str(agent_id).startswith("baseline:"):
                opponent = str(agent_id).split(":", 1)[1]
        return (
            _mock_outcome(format_name=format_name, seed=seed, opponent=opponent, win=True),
            object(),
            {"match_seconds": 0.1, "env_steps": 1, "max_action_seconds": 0.01},
        )

    spec = _spec(incumbent_checkpoint_path=None)
    verdict = run_unified_ladder(
        tmp_path / "ckpt.pkl",
        spec,
        tmp_path / "eval",
        run_match_fn=always_win,
    )
    assert verdict.reason == "no_incumbent"
    assert len(verdict.stages) == 2
    assert verdict.stages[1].skip_reason == "no_incumbent"
