"""Calibration workflow scaffolding for unified tournament thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.jax.preflight_calibration import default_calibration_json_path, git_head_sha
from src.artifacts.tournament.unified.spec import parse_unified_tournament_section


@dataclass(frozen=True, slots=True)
class UnifiedCalibrationPlan:
    checkpoint_paths: tuple[Path, ...]
    games_per_pair_candidates: tuple[int, ...]
    dry_run: bool


def default_unified_tournament_stub(*, enforcement: bool = False) -> dict[str, object]:
    """Non-enforcing unified tournament section for committed calibration JSON."""

    return {
        "enforcement": enforcement,
        "noop_min_combined": 0.7,
        "random_min_combined": 0.58,
        "games_per_pair": 4,
        "prerequisite_seeds": [0, 1, 2, 3, 4],
        "incumbent_seeds": list(range(30)),
        "four_p_baseline_fillers": ["noop", "random", "random"],
        "incumbent_checkpoint_path": None,
        "notes": [
            "Floors are initial combined-metric placeholders until U8 calibration campaigns complete.",
            "Set enforcement=true only after measured pass rates justify thresholds.",
        ],
    }


def build_unified_calibration_report(
    *,
    repo_root: Path,
    plan: UnifiedCalibrationPlan,
    analyze_only: bool,
    seconds_total: float,
) -> dict[str, object]:
    """Build calibration report JSON; GPU campaigns deferred when dry_run."""

    stub = default_unified_tournament_stub(enforcement=False)
    parsed = parse_unified_tournament_section(stub)
    return {
        "gate": "unified_tournament_calibration",
        "commit_sha": git_head_sha(repo_root),
        "seconds_total": seconds_total,
        "analyze_only": analyze_only,
        "dry_run": plan.dry_run,
        "checkpoint_paths": [str(path) for path in plan.checkpoint_paths],
        "games_per_pair_candidates": list(plan.games_per_pair_candidates),
        "unified_tournament": stub,
        "spec_validation": {
            "needs_calibration": parsed.needs_calibration,
            "stage1_seeds": len(parsed.stage1.seeds),
            "stage2_seeds": len(parsed.stage2.seeds),
        },
        "notes": [
            "Run unified ladder campaigns on representative checkpoints before enabling enforcement.",
            "Recalibrate noop/random floors on combined 2p+4p metric (Q2).",
            "Operator publishes bootstrap incumbent_checkpoint_path before Stage 2 enforcement (Q4).",
        ],
    }


def merge_unified_section_into_calibration(
    calibration_path: Path,
    unified_section: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, Any]
    if calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        payload = {"thresholds": {}}
    payload["unified_tournament"] = unified_section
    return payload


def write_unified_calibration_artifact(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def load_unified_section_from_calibration(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    section = payload.get("unified_tournament")
    return section if isinstance(section, dict) else None
