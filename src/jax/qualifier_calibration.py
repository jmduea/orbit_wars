"""Load SSOT tournament qualifier floors (R26–R28)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "docs" / "benchmarks" / "qualifier-seed-calibration.json"

STAGE_LEGS: dict[int, tuple[str, ...]] = {
    1: ("random",),
    2: ("noop", "random"),
    3: ("noop", "random", "nearest_sniper"),
}

INTERIM_MIN_WIN_RATE: dict[int, dict[str, float]] = {
    1: {"random": 0.55},
    2: {"noop": 0.55, "random": 0.50},
    3: {"noop": 0.55, "random": 0.50, "nearest_sniper": 0.45},
}


@dataclass(frozen=True, slots=True)
class QualifierCalibration:
    enforcement: bool
    stages: dict[int, dict[str, float]]

    def min_win_rate_for(self, stage: int, opponent: str) -> float:
        stage_floors = self.stages.get(stage) or INTERIM_MIN_WIN_RATE.get(stage, {})
        if opponent in stage_floors:
            return float(stage_floors[opponent])
        interim = INTERIM_MIN_WIN_RATE.get(stage, {})
        return float(interim.get(opponent, 1.0))


def default_calibration_json_path() -> Path:
    return DEFAULT_CALIBRATION_JSON


def legs_for_stage(stage: int) -> tuple[str, ...]:
    return STAGE_LEGS.get(stage, ())


def load_qualifier_calibration(
    path: Path | None = None,
) -> QualifierCalibration:
    """Load committed calibration or return conservative interim floors (R19)."""

    resolved = path or default_calibration_json_path()
    if not resolved.is_file():
        return QualifierCalibration(
            enforcement=False,
            stages={k: dict(v) for k, v in INTERIM_MIN_WIN_RATE.items()},
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"qualifier calibration must be a JSON object: {resolved}")
    enforcement = bool(payload.get("enforcement", False))
    raw_stages = payload.get("stages", {})
    stages: dict[int, dict[str, float]] = {}
    if isinstance(raw_stages, dict):
        for key, legs in raw_stages.items():
            if not isinstance(legs, dict):
                continue
            try:
                stage_id = int(key)
            except (TypeError, ValueError):
                continue
            stages[stage_id] = {
                str(opponent): float(rate)
                for opponent, rate in legs.items()
            }
    if not stages:
        stages = {k: dict(v) for k, v in INTERIM_MIN_WIN_RATE.items()}
    return QualifierCalibration(enforcement=enforcement, stages=stages)


def default_qualifier_calibration_stub(*, enforcement: bool = False) -> dict[str, object]:
    return {
        "enforcement": enforcement,
        "stages": {
            str(stage): {opponent: rate for opponent, rate in legs.items()}
            for stage, legs in INTERIM_MIN_WIN_RATE.items()
        },
        "notes": "Stub until ow benchmark calibrate-qualifier-seeds campaign commits floors.",
    }
