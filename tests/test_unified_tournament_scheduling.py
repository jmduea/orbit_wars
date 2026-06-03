from __future__ import annotations

from pathlib import Path

import pytest

from src.artifacts.tournament.types import AgentEntry
from src.artifacts.tournament.unified.scheduling import schedule_stage1_matches
from src.artifacts.tournament.unified.spec import parse_unified_tournament_section
from src.config import TrainConfig


def _challenger() -> AgentEntry:
    return AgentEntry(
        agent_id="cand",
        checkpoint_path=Path("/tmp/ckpt.pkl"),
        cfg=TrainConfig(),
        act_fn=lambda _obs: [],
    )


def _spec(**overrides: object):
    base = {
        "enforcement": False,
        "games_per_pair": 2,
        "prerequisite_seeds": [0, 1],
        "four_p_baseline_fillers": ["noop", "random", "random"],
    }
    base.update(overrides)
    return parse_unified_tournament_section(base)


def test_stage1_schedule_counts() -> None:
    spec = _spec()
    schedules = schedule_stage1_matches(_challenger(), spec)
    # 2 opponents * 2 seeds * 2 games (2p) + 2 seeds * 2 games (4p)
    assert len(schedules) == (2 * 2 * 2) + (2 * 2)


def test_validate_four_p_fillers_errors() -> None:
    with pytest.raises(ValueError, match="three baseline"):
        _spec(four_p_baseline_fillers=["noop"])


def test_stage1_includes_both_formats() -> None:
    schedules = schedule_stage1_matches(_challenger(), _spec())
    formats = {entry[0] for entry in schedules}
    assert formats == {"2p_vs_baseline", "4p_challenger_vs_baselines"}
