"""Unified ladder verdict serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.artifacts.run_paths import atomic_write_json
from src.artifacts.tournament.unified.scoring import UnifiedOpponentScore


@dataclass(slots=True)
class UnifiedStageResult:
    name: str
    passed: bool
    opponents: tuple[UnifiedOpponentScore, ...] = ()
    per_seed_combined: list[float | None] = field(default_factory=list)
    all_seeds_perfect: bool = False
    skip_reason: str | None = None
    output_dir: str | None = None


@dataclass(slots=True)
class UnifiedLadderVerdict:
    passed: bool
    reason: str
    stages: tuple[UnifiedStageResult, ...]
    challenger_checkpoint: str
    incumbent_swap: bool = False
    enforcement: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "challenger_checkpoint": self.challenger_checkpoint,
            "incumbent_swap": self.incumbent_swap,
            "enforcement": self.enforcement,
            "stages": [
                {
                    "name": stage.name,
                    "passed": stage.passed,
                    "skip_reason": stage.skip_reason,
                    "output_dir": stage.output_dir,
                    "opponents": [
                        {
                            "opponent": row.opponent,
                            "win_rate_2p": row.win_rate_2p,
                            "win_rate_4p": row.win_rate_4p,
                            "combined": row.combined,
                            "passed": row.passed,
                            "fail_reason": row.fail_reason,
                        }
                        for row in stage.opponents
                    ],
                    "per_seed_combined": stage.per_seed_combined,
                    "all_seeds_perfect": stage.all_seeds_perfect,
                }
                for stage in self.stages
            ],
        }


def write_unified_verdict(output_dir: Path, verdict: UnifiedLadderVerdict) -> Path:
    path = output_dir / "unified_verdict.json"
    atomic_write_json(path, verdict.to_dict())
    return path
