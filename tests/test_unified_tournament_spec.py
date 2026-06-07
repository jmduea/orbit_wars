from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.tournament.unified.spec import (
    load_unified_tournament_spec,
    parse_unified_tournament_section,
)
from src.config.schema import UnifiedTournamentConfig


def test_load_spec_from_fixture_json(tmp_path: Path) -> None:
    payload = {
        "unified_tournament": {
            "enforcement": False,
            "noop_min_combined": 0.7,
            "random_min_combined": 0.58,
            "games_per_pair": 4,
            "prerequisite_seeds": [0, 1, 2, 3, 4],
            "incumbent_seeds": list(range(30)),
            "four_p_baseline_fillers": ["noop", "random", "random"],
        }
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    spec = load_unified_tournament_spec(path)

    assert spec.stage1.seeds == (0, 1, 2, 3, 4)
    assert spec.stage2.seeds == tuple(range(30))
    assert spec.stage1.floors["noop"] == 0.7
    assert spec.stage1.floors["random"] == 0.58
    assert not spec.needs_calibration


def test_missing_unified_section_returns_needs_calibration() -> None:
    spec = parse_unified_tournament_section(None)
    assert spec.needs_calibration
    assert spec.blocking_reason == "needs_calibration"


def test_stage2_enforcement_without_incumbent_blocks() -> None:
    section = {
        "enforcement": True,
        "four_p_baseline_fillers": ["noop", "random", "random"],
        "incumbent_checkpoint_path": None,
    }
    spec = parse_unified_tournament_section(section)
    assert spec.blocking_reason == "no_incumbent"


def test_invalid_four_p_fillers_raises() -> None:
    with pytest.raises(ValueError, match="three baseline"):
        parse_unified_tournament_section(
            {"four_p_baseline_fillers": ["noop"], "enforcement": False}
        )


def test_hydra_unified_tournament_profile_composes() -> None:
    from src.config import compose_hydra_train_config

    cfg = compose_hydra_train_config(["artifacts=unified_tournament"])
    assert cfg.artifacts.unified_tournament.enabled
    assert cfg.artifacts.tournament.enabled
