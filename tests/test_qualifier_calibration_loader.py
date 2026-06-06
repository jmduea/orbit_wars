"""Qualifier seed calibration JSON loader."""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.calibration.qualifier_floors import (
    default_calibration_json_path,
    load_qualifier_calibration,
)


def test_default_calibration_path_exists() -> None:
    path = default_calibration_json_path()
    assert path.is_file()


def test_loader_enforcement_flag_from_json(tmp_path: Path) -> None:
    path = tmp_path / "qualifier.json"
    path.write_text(
        json.dumps(
            {
                "enforcement": True,
                "stages": {"1": {"random": 0.6}},
            }
        ),
        encoding="utf-8",
    )
    cal = load_qualifier_calibration(path)
    assert cal.enforcement is True
    assert cal.min_win_rate_for(1, "random") == 0.6


def test_missing_file_uses_interim_floors(tmp_path: Path) -> None:
    cal = load_qualifier_calibration(tmp_path / "missing.json")
    assert cal.enforcement is False
    assert cal.min_win_rate_for(3, "nearest_sniper") == 0.45
