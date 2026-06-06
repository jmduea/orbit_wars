"""GPU calibration campaigns for unified tournament Stage-1 floors."""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts.tournament.unified.ladder import run_unified_ladder
from src.artifacts.tournament.unified.reporting import UnifiedLadderVerdict
from src.artifacts.tournament.unified.scoring import UnifiedOpponentScore
from src.artifacts.tournament.unified.spec import (
    UnifiedTournamentSpec,
    parse_unified_tournament_section,
)
from src.jax.preflight_calibration import default_calibration_json_path, git_head_sha

DEFAULT_CALIBRATION_CHECKPOINT = Path(
    "outputs/campaigns/preflight_beat_random/runs/"
    "20260602T193448Z-s42-0422c38a/checkpoints/jax_ckpt_last.pkl"
)
DEFAULT_GAMES_PER_PAIR_CANDIDATES: tuple[int, ...] = (2, 4, 8)
UNIFIED_CAL_CAMPAIGN_RE = re.compile(r"^unified_tournament_cal_g(\d+)$")


@dataclass(frozen=True, slots=True)
class UnifiedCalibrationPlan:
    checkpoint_paths: tuple[Path, ...]
    games_per_pair_candidates: tuple[int, ...]
    dry_run: bool
    output_root: Path = Path("outputs")


@dataclass(frozen=True, slots=True)
class UnifiedCalSnapshot:
    checkpoint_path: str
    games_per_pair: int
    campaign: str
    output_dir: str | None
    seconds_total: float | None
    noop_combined: float | None
    random_combined: float | None
    noop_win_rate_2p: float | None
    noop_win_rate_4p: float | None
    random_win_rate_2p: float | None
    random_win_rate_4p: float | None
    stage1_passed: bool
    reason: str


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
        "incumbent_bootstrap_opponent": "nearest_sniper",
        "notes": [
            "Floors are initial combined-metric placeholders until U8 calibration campaigns complete.",
            "Set enforcement=true only after measured pass rates justify thresholds.",
        ],
    }


def unified_cal_campaign(games_per_pair: int) -> str:
    return f"unified_tournament_cal_g{games_per_pair}"


def unified_cal_output_dir(
    *,
    output_root: Path,
    games_per_pair: int,
    checkpoint_stem: str,
) -> Path:
    return (
        output_root
        / "campaigns"
        / unified_cal_campaign(games_per_pair)
        / "evaluations"
        / f"cal_{checkpoint_stem}"
    )


def _spec_for_measurement(
    *,
    games_per_pair: int,
    base_section: dict[str, object] | None,
) -> UnifiedTournamentSpec:
    section: dict[str, object] = {
        "enforcement": False,
        "games_per_pair": games_per_pair,
        "prerequisite_seeds": [0, 1, 2, 3, 4],
        "incumbent_seeds": list(range(30)),
        "four_p_baseline_fillers": ["noop", "random", "random"],
    }
    if base_section:
        for key in (
            "prerequisite_seeds",
            "incumbent_seeds",
            "four_p_baseline_fillers",
            "incumbent_bootstrap_opponent",
            "max_steps",
            "per_step_seconds",
            "overage_budget_seconds",
        ):
            if key in base_section:
                section[key] = base_section[key]
    return parse_unified_tournament_section(section)


def _opponent_score(
    verdict: UnifiedLadderVerdict, opponent: str
) -> UnifiedOpponentScore | None:
    if not verdict.stages:
        return None
    stage1 = verdict.stages[0]
    for row in stage1.opponents:
        if row.opponent == opponent:
            return row
    return None


def analyze_unified_cal_verdict(
    *,
    checkpoint_path: Path,
    games_per_pair: int,
    output_dir: Path,
    verdict: UnifiedLadderVerdict,
    seconds_total: float | None,
) -> UnifiedCalSnapshot:
    noop = _opponent_score(verdict, "noop")
    random = _opponent_score(verdict, "random")
    return UnifiedCalSnapshot(
        checkpoint_path=str(checkpoint_path.resolve()),
        games_per_pair=games_per_pair,
        campaign=unified_cal_campaign(games_per_pair),
        output_dir=str(output_dir),
        seconds_total=seconds_total,
        noop_combined=noop.combined if noop else None,
        random_combined=random.combined if random else None,
        noop_win_rate_2p=noop.win_rate_2p if noop else None,
        noop_win_rate_4p=noop.win_rate_4p if noop else None,
        random_win_rate_2p=random.win_rate_2p if random else None,
        random_win_rate_4p=random.win_rate_4p if random else None,
        stage1_passed=bool(verdict.stages and verdict.stages[0].passed),
        reason=verdict.reason,
    )


def run_unified_calibration_arm(
    *,
    checkpoint_path: Path,
    games_per_pair: int,
    output_root: Path,
    repo_root: Path,
    base_section: dict[str, object] | None,
    dry_run: bool,
) -> UnifiedCalSnapshot:
    spec = _spec_for_measurement(games_per_pair=games_per_pair, base_section=base_section)
    output_dir = unified_cal_output_dir(
        output_root=output_root,
        games_per_pair=games_per_pair,
        checkpoint_stem=checkpoint_path.stem,
    )
    if dry_run:
        print(
            f"unified calibration: checkpoint={checkpoint_path} games_per_pair={games_per_pair} "
            f"output_dir={output_dir}",
            flush=True,
        )
        verdict = run_unified_ladder(
            checkpoint_path,
            spec,
            output_dir,
            output_root=output_root,
            dry_run=True,
        )
        return analyze_unified_cal_verdict(
            checkpoint_path=checkpoint_path,
            games_per_pair=games_per_pair,
            output_dir=output_dir,
            verdict=verdict,
            seconds_total=None,
        )

    started = time.perf_counter()
    verdict = run_unified_ladder(
        checkpoint_path,
        spec,
        output_dir,
        campaign=unified_cal_campaign(games_per_pair),
        output_root=output_root,
        stop_after_stage1=True,
    )
    return analyze_unified_cal_verdict(
        checkpoint_path=checkpoint_path,
        games_per_pair=games_per_pair,
        output_dir=output_dir,
        verdict=verdict,
        seconds_total=time.perf_counter() - started,
    )


def run_unified_calibration_sweep(
    *,
    plan: UnifiedCalibrationPlan,
    repo_root: Path,
    base_section: dict[str, object] | None,
) -> list[UnifiedCalSnapshot]:
    checkpoints = plan.checkpoint_paths or (DEFAULT_CALIBRATION_CHECKPOINT,)
    arms = [
        (checkpoint, games)
        for checkpoint in checkpoints
        for games in plan.games_per_pair_candidates
    ]
    print(
        f"Unified tournament calibration: {len(arms)} Stage-1 arm(s), "
        f"output_root={plan.output_root}",
        flush=True,
    )
    snapshots: list[UnifiedCalSnapshot] = []
    for arm_index, (checkpoint, games) in enumerate(arms, start=1):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing calibration checkpoint: {checkpoint}")
        print(
            f"=== unified calibration arm {arm_index}/{len(arms)} "
            f"checkpoint={checkpoint.name} games_per_pair={games} ===",
            flush=True,
        )
        snapshots.append(
            run_unified_calibration_arm(
                checkpoint_path=checkpoint,
                games_per_pair=games,
                output_root=plan.output_root,
                repo_root=repo_root,
                base_section=base_section,
                dry_run=plan.dry_run,
            )
        )
    return snapshots


def discover_unified_cal_snapshots(
    output_root: Path,
    *,
    games_per_pair_candidates: tuple[int, ...],
    checkpoint_paths: tuple[Path, ...],
) -> list[UnifiedCalSnapshot]:
    snapshots: list[UnifiedCalSnapshot] = []
    checkpoints = checkpoint_paths or (DEFAULT_CALIBRATION_CHECKPOINT,)
    for games in games_per_pair_candidates:
        campaign = unified_cal_campaign(games)
        eval_root = output_root / "campaigns" / campaign / "evaluations"
        if not eval_root.is_dir():
            continue
        for checkpoint in checkpoints:
            output_dir = eval_root / f"cal_{checkpoint.stem}"
            verdict_path = output_dir / "unified_verdict.json"
            if not verdict_path.is_file():
                continue
            payload = json.loads(verdict_path.read_text(encoding="utf-8"))
            stages = payload.get("stages") or []
            stage1 = stages[0] if stages else {}
            opponents = {
                str(row.get("opponent")): row for row in stage1.get("opponents", [])
            }

            def _combined(name: str) -> float | None:
                raw = opponents.get(name, {}).get("combined")
                return float(raw) if raw is not None else None

            snapshots.append(
                UnifiedCalSnapshot(
                    checkpoint_path=str(checkpoint.resolve()),
                    games_per_pair=games,
                    campaign=campaign,
                    output_dir=str(output_dir),
                    seconds_total=None,
                    noop_combined=_combined("noop"),
                    random_combined=_combined("random"),
                    noop_win_rate_2p=_rate(opponents.get("noop"), "win_rate_2p"),
                    noop_win_rate_4p=_rate(opponents.get("noop"), "win_rate_4p"),
                    random_win_rate_2p=_rate(opponents.get("random"), "win_rate_2p"),
                    random_win_rate_4p=_rate(opponents.get("random"), "win_rate_4p"),
                    stage1_passed=bool(stage1.get("passed")),
                    reason=str(payload.get("reason", "unknown")),
                )
            )
    return snapshots


def _rate(opponent_row: dict[str, object] | None, key: str) -> float | None:
    if not opponent_row:
        return None
    raw = opponent_row.get(key)
    return float(raw) if raw is not None else None


def pick_games_per_pair(snapshots: list[UnifiedCalSnapshot]) -> dict[str, object]:
    """Choose games-per-pair with valid combined scores and lowest wall-time tie-break."""

    by_games: dict[int, list[UnifiedCalSnapshot]] = {}
    for snapshot in snapshots:
        if snapshot.noop_combined is None or snapshot.random_combined is None:
            continue
        by_games.setdefault(snapshot.games_per_pair, []).append(snapshot)

    candidates: list[tuple[int, float, float, float | None]] = []
    for games, group in sorted(by_games.items()):
        noop_vals = [float(item.noop_combined) for item in group]
        random_vals = [float(item.random_combined) for item in group]
        times = [item.seconds_total for item in group if item.seconds_total is not None]
        candidates.append(
            (
                games,
                min(noop_vals + random_vals),
                statistics.mean(noop_vals + random_vals),
                max(times) if times else None,
            )
        )

    if not candidates:
        return {
            "chosen_games_per_pair": None,
            "reason": "no arms with both noop and random combined scores",
        }

    candidates.sort(key=lambda item: (-item[1], -item[2], item[3] if item[3] is not None else 0.0))
    chosen = candidates[0][0]
    return {
        "chosen_games_per_pair": chosen,
        "min_combined_observed": candidates[0][1],
        "mean_combined_observed": candidates[0][2],
        "max_seconds": candidates[0][3],
        "candidate_count": len(candidates),
    }


def derive_unified_floors(
    snapshots: list[UnifiedCalSnapshot],
    *,
    games_per_pair: int,
    margin_fraction: float = 0.05,
) -> dict[str, object]:
    """Derive noop/random combined floors from measured Stage-1 scores."""

    selected = [item for item in snapshots if item.games_per_pair == games_per_pair]
    noop_vals = [
        float(item.noop_combined)
        for item in selected
        if item.noop_combined is not None
    ]
    random_vals = [
        float(item.random_combined)
        for item in selected
        if item.random_combined is not None
    ]
    if not noop_vals or not random_vals:
        return {
            "noop_min_combined": None,
            "random_min_combined": None,
            "reason": "missing measured combined scores at chosen games_per_pair",
        }

    noop_obs = min(noop_vals)
    random_obs = min(random_vals)
    noop_floor = round(max(0.0, noop_obs * (1.0 - margin_fraction)), 3)
    random_floor = round(max(0.0, random_obs * (1.0 - margin_fraction)), 3)
    if random_floor > noop_floor:
        random_floor = noop_floor

    return {
        "noop_min_combined": noop_floor,
        "random_min_combined": random_floor,
        "noop_combined_observed_min": noop_obs,
        "random_combined_observed_min": random_obs,
        "margin_fraction": margin_fraction,
        "legacy_2p_floors": {"noop": 0.7, "random": 0.58},
    }


def verification_passes_at_derived_floors(
    snapshots: list[UnifiedCalSnapshot],
    *,
    games_per_pair: int,
    noop_floor: float,
    random_floor: float,
) -> bool:
    selected = [item for item in snapshots if item.games_per_pair == games_per_pair]
    if not selected:
        return False
    return all(
        item.noop_combined is not None
        and item.random_combined is not None
        and float(item.noop_combined) >= noop_floor
        and float(item.random_combined) >= random_floor
        for item in selected
    )


def build_calibrated_unified_section(
    snapshots: list[UnifiedCalSnapshot],
    *,
    base_section: dict[str, object] | None,
    enable_enforcement: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    games_decision = pick_games_per_pair(snapshots)
    chosen = games_decision.get("chosen_games_per_pair")
    if chosen is None:
        stub = default_unified_tournament_stub(enforcement=False)
        if base_section:
            stub.update(
                {
                    key: base_section[key]
                    for key in (
                        "incumbent_bootstrap_opponent",
                        "prerequisite_seeds",
                        "incumbent_seeds",
                        "four_p_baseline_fillers",
                    )
                    if key in base_section
                }
            )
        return stub, {"games_decision": games_decision, "floors": {}, "enforcement": False}

    floors = derive_unified_floors(snapshots, games_per_pair=int(chosen))
    noop_floor = floors.get("noop_min_combined")
    random_floor = floors.get("random_min_combined")
    enforcement = False
    if (
        enable_enforcement
        and isinstance(noop_floor, (int, float))
        and isinstance(random_floor, (int, float))
        and verification_passes_at_derived_floors(
            snapshots,
            games_per_pair=int(chosen),
            noop_floor=float(noop_floor),
            random_floor=float(random_floor),
        )
    ):
        enforcement = True

    section: dict[str, object] = {
        "enforcement": enforcement,
        "noop_min_combined": noop_floor,
        "random_min_combined": random_floor,
        "games_per_pair": int(chosen),
        "prerequisite_seeds": [0, 1, 2, 3, 4],
        "incumbent_seeds": list(range(30)),
        "four_p_baseline_fillers": ["noop", "random", "random"],
        "incumbent_bootstrap_opponent": "nearest_sniper",
        "notes": [
            "Floors derived from unified Stage-1 calibration (combined 2p+4p).",
            f"Measured at games_per_pair={chosen} with {floors.get('margin_fraction', 0.05):.0%} margin below observed min.",
        ],
    }
    if base_section:
        for key in (
            "incumbent_bootstrap_opponent",
            "prerequisite_seeds",
            "incumbent_seeds",
            "four_p_baseline_fillers",
            "max_steps",
            "per_step_seconds",
            "overage_budget_seconds",
        ):
            if key in base_section and base_section[key] is not None:
                section[key] = base_section[key]
    legacy = floors.get("legacy_2p_floors")
    if isinstance(legacy, dict):
        noop_legacy = legacy.get("noop")
        random_legacy = legacy.get("random")
        if isinstance(noop_floor, (int, float)) and isinstance(noop_legacy, (int, float)):
            if float(noop_floor) > float(noop_legacy):
                section["notes"].append(
                    f"noop combined floor {noop_floor} exceeds legacy 2p-only {noop_legacy}."
                )
        if isinstance(random_floor, (int, float)) and isinstance(
            random_legacy, (int, float)
        ):
            if float(random_floor) > float(random_legacy):
                section["notes"].append(
                    f"random combined floor {random_floor} exceeds legacy 2p-only {random_legacy}."
                )
    return section, {
        "games_decision": games_decision,
        "floors": floors,
        "enforcement": enforcement,
    }


def snapshot_to_dict(snapshot: UnifiedCalSnapshot) -> dict[str, object]:
    return {
        "checkpoint_path": snapshot.checkpoint_path,
        "games_per_pair": snapshot.games_per_pair,
        "campaign": snapshot.campaign,
        "output_dir": snapshot.output_dir,
        "seconds_total": snapshot.seconds_total,
        "noop_combined": snapshot.noop_combined,
        "random_combined": snapshot.random_combined,
        "noop_win_rate_2p": snapshot.noop_win_rate_2p,
        "noop_win_rate_4p": snapshot.noop_win_rate_4p,
        "random_win_rate_2p": snapshot.random_win_rate_2p,
        "random_win_rate_4p": snapshot.random_win_rate_4p,
        "stage1_passed": snapshot.stage1_passed,
        "reason": snapshot.reason,
    }


def build_unified_calibration_report(
    *,
    repo_root: Path,
    plan: UnifiedCalibrationPlan,
    snapshots: list[UnifiedCalSnapshot],
    analyze_only: bool,
    seconds_total: float,
    base_section: dict[str, object] | None,
    enable_enforcement: bool,
) -> dict[str, object]:
    calibrated_section, decision = build_calibrated_unified_section(
        snapshots,
        base_section=base_section,
        enable_enforcement=enable_enforcement and not plan.dry_run and bool(snapshots),
    )
    parsed = parse_unified_tournament_section(calibrated_section)
    return {
        "gate": "unified_tournament_calibration",
        "commit_sha": git_head_sha(repo_root),
        "seconds_total": seconds_total,
        "analyze_only": analyze_only,
        "dry_run": plan.dry_run,
        "checkpoint_paths": [str(path) for path in plan.checkpoint_paths],
        "games_per_pair_candidates": list(plan.games_per_pair_candidates),
        "unified_tournament": calibrated_section,
        "decision": decision,
        "spec_validation": {
            "needs_calibration": parsed.needs_calibration,
            "stage1_seeds": len(parsed.stage1.seeds),
            "stage2_seeds": len(parsed.stage2.seeds),
        },
        "runs": [snapshot_to_dict(item) for item in snapshots],
        "notes": [
            "Stage-1-only calibration sweep; Stage-2 incumbent bar unchanged.",
            "Set enforcement=true only after measured combined floors verify on calibration checkpoints.",
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


def default_unified_calibration_artifact_path(repo_root: Path) -> Path:
    return repo_root / "docs/benchmarks/unified-tournament-calibration.json"


def default_preflight_calibration_path(repo_root: Path) -> Path:
    return default_calibration_json_path(repo_root)
