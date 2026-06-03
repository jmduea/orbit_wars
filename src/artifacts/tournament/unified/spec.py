"""Canonical unified tournament ladder specification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.artifacts.tournament.resolve import DEFAULT_BOOTSTRAP_INCUMBENT
from src.config.schema import UnifiedTournamentConfig

STAGE1_OPPONENTS = ("noop", "random")
STAGE1_FORMATS = ("2p_vs_baseline", "4p_challenger_vs_baselines")
STAGE2_FORMATS = ("2p_head_to_head", "4p_challenger_vs_baselines")


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One ladder stage: opponents, seeds, formats, and optional score floors."""

    name: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    games_per_pair: int
    formats: tuple[str, ...] = STAGE1_FORMATS
    floors: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UnifiedTournamentSpec:
    """Full held-out tournament ladder definition."""

    stage1: StageSpec
    stage2: StageSpec
    four_p_baseline_fillers: tuple[str, ...]
    incumbent_bootstrap_opponent: str | None = DEFAULT_BOOTSTRAP_INCUMBENT
    enforcement: bool = False
    needs_calibration: bool = False
    blocking_reason: str | None = None
    max_steps: int = 500
    per_step_seconds: float = 1.0
    overage_budget_seconds: float = 60.0
    write_replays: bool = False

    @property
    def stage2_blocked(self) -> bool:
        return self.blocking_reason in {"no_incumbent", "needs_calibration"}


def _parse_seed_list(raw: Any, *, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(raw, list):
        return tuple(int(value) for value in raw)
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if parts:
            return tuple(int(part) for part in parts)
    return fallback


def _parse_fillers(raw: Any, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, list) and raw:
        return tuple(str(value).strip() for value in raw if str(value).strip())
    return fallback


def _bootstrap_opponent(raw: Any, *, fallback: str | None) -> str | None:
    if raw is None:
        return fallback
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    return fallback


def _stage1_from_payload(
    section: dict[str, object],
    *,
    hydra: UnifiedTournamentConfig | None,
) -> StageSpec:
    hydra = hydra or UnifiedTournamentConfig()
    seeds = _parse_seed_list(
        section.get("prerequisite_seeds"),
        fallback=tuple(hydra.prerequisite_seeds),
    )
    games = int(section.get("games_per_pair", hydra.games_per_pair))
    floors: dict[str, float] = {}
    if "noop_min_combined" in section:
        floors["noop"] = float(section["noop_min_combined"])
    elif hydra.noop_min_combined is not None:
        floors["noop"] = float(hydra.noop_min_combined)
    if "random_min_combined" in section:
        floors["random"] = float(section["random_min_combined"])
    elif hydra.random_min_combined is not None:
        floors["random"] = float(hydra.random_min_combined)
    return StageSpec(
        name="stage1_prerequisites",
        opponents=STAGE1_OPPONENTS,
        seeds=seeds,
        games_per_pair=games,
        formats=STAGE1_FORMATS,
        floors=floors,
    )


def _stage2_from_payload(
    section: dict[str, object],
    *,
    hydra: UnifiedTournamentConfig | None,
) -> StageSpec:
    hydra = hydra or UnifiedTournamentConfig()
    seeds = _parse_seed_list(
        section.get("incumbent_seeds"),
        fallback=tuple(hydra.incumbent_seeds),
    )
    games = int(section.get("games_per_pair", hydra.games_per_pair))
    return StageSpec(
        name="stage2_incumbent",
        opponents=("incumbent",),
        seeds=seeds,
        games_per_pair=games,
        formats=STAGE2_FORMATS,
        floors={},
    )


def parse_unified_tournament_section(
    section: dict[str, object] | None,
    *,
    hydra: UnifiedTournamentConfig | None = None,
) -> UnifiedTournamentSpec:
    """Build a spec from a calibration JSON ``unified_tournament`` section."""

    hydra = hydra or UnifiedTournamentConfig()
    if section is None:
        return UnifiedTournamentSpec(
            stage1=_stage1_from_payload({}, hydra=hydra),
            stage2=_stage2_from_payload({}, hydra=hydra),
            four_p_baseline_fillers=_parse_fillers(
                None, fallback=tuple(hydra.four_p_baseline_fillers)
            ),
            incumbent_bootstrap_opponent=_bootstrap_opponent(
                hydra.incumbent_bootstrap_opponent,
                fallback=DEFAULT_BOOTSTRAP_INCUMBENT,
            ),
            enforcement=False,
            needs_calibration=True,
            blocking_reason="needs_calibration",
            max_steps=hydra.max_steps,
            per_step_seconds=hydra.per_step_seconds,
            overage_budget_seconds=hydra.overage_budget_seconds,
            write_replays=hydra.write_replays,
        )

    fillers = _parse_fillers(
        section.get("four_p_baseline_fillers"),
        fallback=tuple(hydra.four_p_baseline_fillers),
    )
    bootstrap_opponent = _bootstrap_opponent(
        section.get("incumbent_bootstrap_opponent", hydra.incumbent_bootstrap_opponent),
        fallback=DEFAULT_BOOTSTRAP_INCUMBENT,
    )
    enforcement = bool(section.get("enforcement", hydra.enforcement))
    blocking: str | None = None
    if len(fillers) < 3:
        raise ValueError(
            "four_p_baseline_fillers requires three baseline slots for 4p leg"
        )
    if enforcement and bootstrap_opponent is None:
        blocking = "no_incumbent"

    return UnifiedTournamentSpec(
        stage1=_stage1_from_payload(section, hydra=hydra),
        stage2=_stage2_from_payload(section, hydra=hydra),
        four_p_baseline_fillers=fillers,
        incumbent_bootstrap_opponent=bootstrap_opponent,
        enforcement=enforcement,
        needs_calibration=False,
        blocking_reason=blocking,
        max_steps=int(section.get("max_steps", hydra.max_steps)),
        per_step_seconds=float(
            section.get("per_step_seconds", hydra.per_step_seconds)
        ),
        overage_budget_seconds=float(
            section.get("overage_budget_seconds", hydra.overage_budget_seconds)
        ),
        write_replays=bool(section.get("write_replays", hydra.write_replays)),
    )


def load_unified_tournament_spec(
    calibration_path: Path | None = None,
    *,
    hydra: UnifiedTournamentConfig | None = None,
    require_enforcement: bool = False,
) -> UnifiedTournamentSpec:
    """Load unified ladder spec from calibration JSON with Hydra fallbacks."""

    hydra = hydra or UnifiedTournamentConfig()
    section: dict[str, object] | None = None
    if calibration_path is not None and calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        raw = payload.get("unified_tournament")
        if isinstance(raw, dict):
            section = raw
        elif "thresholds" in payload and isinstance(payload["thresholds"], dict):
            nested = payload["thresholds"].get("unified_tournament")
            if isinstance(nested, dict):
                section = nested

    spec = parse_unified_tournament_section(section, hydra=hydra)
    if require_enforcement and not spec.enforcement:
        return UnifiedTournamentSpec(
            stage1=spec.stage1,
            stage2=spec.stage2,
            four_p_baseline_fillers=spec.four_p_baseline_fillers,
            incumbent_bootstrap_opponent=spec.incumbent_bootstrap_opponent,
            enforcement=False,
            needs_calibration=True,
            blocking_reason="needs_calibration",
            max_steps=spec.max_steps,
            per_step_seconds=spec.per_step_seconds,
            overage_budget_seconds=spec.overage_budget_seconds,
            write_replays=spec.write_replays,
        )
    return spec


def validate_spec_for_stage2(
    spec: UnifiedTournamentSpec,
    *,
    incumbent_resolved: bool,
) -> str | None:
    """Return blocking reason when Stage 2 cannot run."""

    if spec.needs_calibration:
        return "needs_calibration"
    if not incumbent_resolved and spec.incumbent_bootstrap_opponent is None:
        return "no_incumbent"
    return None
