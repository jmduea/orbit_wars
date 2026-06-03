"""Staged unified tournament ladder runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.artifacts.run_paths import atomic_write_json
from src.artifacts.timing import TournamentTimingError
from src.artifacts.tournament.resolve import agent_from_checkpoint
from src.artifacts.tournament.runner import run_match
from src.artifacts.tournament.types import AgentEntry, MatchOutcome
from src.artifacts.tournament.unified.incumbent import resolve_incumbent
from src.artifacts.tournament.unified.reporting import (
    UnifiedLadderVerdict,
    UnifiedStageResult,
    write_unified_verdict,
)
from src.artifacts.tournament.unified.scheduling import (
    ScheduleEntry,
    schedule_stage1_matches,
    schedule_stage2_matches,
)
from src.artifacts.tournament.unified.scoring import (
    score_opponent,
    stage2_per_seed_summary,
)
from src.artifacts.tournament.unified.spec import (
    UnifiedTournamentSpec,
    qualifier_sniper_stage,
    validate_spec_for_stage2,
    with_qualifier_floors,
)

MatchRunner = Callable[..., tuple[MatchOutcome, Any, dict[str, object]]]


def _default_run_match(**kwargs: Any) -> tuple[MatchOutcome, Any, dict[str, object]]:
    return run_match(**kwargs)


def _execute_schedules(
    schedules: Sequence[ScheduleEntry],
    *,
    output_dir: Path,
    spec: UnifiedTournamentSpec,
    run_match_fn: MatchRunner,
    match_index_start: int = 0,
) -> tuple[tuple[MatchOutcome, ...], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[MatchOutcome] = []
    match_index = match_index_start
    for format_name, baseline_name, agent_ids, agents, logical_seed in schedules:
        match_id = f"{format_name}_{match_index:04d}"
        outcome, _env, timing_summary = run_match_fn(
            match_id=match_id,
            format_name=format_name,
            seed=logical_seed,
            agent_ids=agent_ids,
            agents=agents,
            max_steps=spec.max_steps,
            per_step_seconds=spec.per_step_seconds,
            overage_budget_seconds=spec.overage_budget_seconds,
        )
        outcomes.append(outcome)
        atomic_write_json(
            output_dir / "matches" / f"{match_id}.json",
            {
                "match_id": match_id,
                "format_name": format_name,
                "baseline_name": baseline_name,
                "seed": logical_seed,
                "agent_ids": list(outcome.agent_ids),
                "results": outcome.results,
                "rewards": outcome.rewards,
                "placements": outcome.placements,
                "timing": timing_summary,
            },
        )
        match_index += 1
    return tuple(outcomes), match_index


def _evaluate_stage1(
    outcomes: tuple[MatchOutcome, ...],
    *,
    challenger_id: str,
    spec: UnifiedTournamentSpec,
) -> tuple[bool, tuple[Any, ...], str | None]:
    opponent_scores = []
    for opponent in spec.stage1.opponents:
        floor = spec.stage1.floors.get(opponent)
        row = score_opponent(
            outcomes,
            challenger_id=challenger_id,
            opponent=opponent,
            floor=floor,
        )
        opponent_scores.append(row)
        if row.combined is None:
            reason = row.fail_reason or f"missing_combined_{opponent}"
            return False, tuple(opponent_scores), f"failed_prerequisite_{opponent}_{reason}"
        if floor is not None and not row.passed:
            return (
                False,
                tuple(opponent_scores),
                f"failed_prerequisite_{opponent}",
            )
    return True, tuple(opponent_scores), None


def run_unified_ladder(
    checkpoint_path: Path,
    spec: UnifiedTournamentSpec,
    output_dir: Path,
    *,
    campaign: str | None = None,
    output_root: Path | None = None,
    run_match_fn: MatchRunner | None = None,
    dry_run: bool = False,
    stop_after_stage1: bool = False,
    qualifier_mode: bool = False,
    skip_stage2: bool = False,
) -> UnifiedLadderVerdict:
    """Run prerequisite-first unified ladder with early exit."""

    if qualifier_mode:
        spec = with_qualifier_floors(spec)

    run_match_fn = run_match_fn or _default_run_match
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_root or Path("outputs")

    if dry_run:
        return UnifiedLadderVerdict(
            passed=False,
            reason="dry_run",
            stages=(
                UnifiedStageResult(name=spec.stage1.name, passed=False),
                UnifiedStageResult(name=spec.stage2.name, passed=False),
            ),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
        )

    challenger = agent_from_checkpoint(checkpoint_path.resolve())
    challenger_id = challenger.agent_id

    stage1_dir = output_dir / "stage1_prerequisites"
    stage1_schedules = schedule_stage1_matches(challenger, spec)
    try:
        stage1_outcomes, _ = _execute_schedules(
            stage1_schedules,
            output_dir=stage1_dir,
            spec=spec,
            run_match_fn=run_match_fn,
        )
    except TournamentTimingError as exc:
        verdict = UnifiedLadderVerdict(
            passed=False,
            reason=f"stage1_timing_error:{exc}",
            stages=(
                UnifiedStageResult(
                    name=spec.stage1.name,
                    passed=False,
                    output_dir=str(stage1_dir),
                ),
            ),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
        )
        write_unified_verdict(output_dir, verdict)
        return verdict

    stage1_passed, opponent_scores, fail_reason = _evaluate_stage1(
        stage1_outcomes, challenger_id=challenger_id, spec=spec
    )
    stage1_result = UnifiedStageResult(
        name=spec.stage1.name,
        passed=stage1_passed,
        opponents=opponent_scores,
        output_dir=str(stage1_dir),
        skip_reason=fail_reason,
    )
    if not stage1_passed:
        verdict = UnifiedLadderVerdict(
            passed=False,
            reason=fail_reason or "failed_prerequisite",
            stages=(stage1_result,),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
        )
        write_unified_verdict(output_dir, verdict)
        return verdict

    if stop_after_stage1:
        verdict = UnifiedLadderVerdict(
            passed=True,
            reason="stage1_calibration_only",
            stages=(stage1_result,),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
        )
        write_unified_verdict(output_dir, verdict)
        return verdict

    stages: list[UnifiedStageResult] = [stage1_result]
    if qualifier_mode:
        sniper_stage = qualifier_sniper_stage(spec)
        sniper_dir = output_dir / "stage1b_qualifier_sniper"
        sniper_schedules = schedule_stage1_matches(
            challenger, spec, stage=sniper_stage
        )
        try:
            sniper_outcomes, _ = _execute_schedules(
                sniper_schedules,
                output_dir=sniper_dir,
                spec=spec,
                run_match_fn=run_match_fn,
                match_index_start=len(stage1_schedules),
            )
        except TournamentTimingError as exc:
            sniper_result = UnifiedStageResult(
                name=sniper_stage.name,
                passed=False,
                output_dir=str(sniper_dir),
                skip_reason=f"timing_error:{exc}",
            )
            verdict = UnifiedLadderVerdict(
                passed=False,
                reason=f"qualifier_sniper_timing_error:{exc}",
                stages=(*stages, sniper_result),
                challenger_checkpoint=str(checkpoint_path.resolve()),
                enforcement=spec.enforcement,
            )
            write_unified_verdict(output_dir, verdict)
            return verdict

        sniper_spec = UnifiedTournamentSpec(
            stage1=sniper_stage,
            stage2=spec.stage2,
            four_p_baseline_fillers=spec.four_p_baseline_fillers,
            incumbent_bootstrap_opponent=spec.incumbent_bootstrap_opponent,
            enforcement=spec.enforcement,
            needs_calibration=spec.needs_calibration,
            blocking_reason=spec.blocking_reason,
            max_steps=spec.max_steps,
            per_step_seconds=spec.per_step_seconds,
            overage_budget_seconds=spec.overage_budget_seconds,
            write_replays=spec.write_replays,
        )
        sniper_passed, sniper_scores, sniper_fail = _evaluate_stage1(
            sniper_outcomes,
            challenger_id=challenger_id,
            spec=sniper_spec,
        )
        sniper_result = UnifiedStageResult(
            name=sniper_stage.name,
            passed=sniper_passed,
            opponents=sniper_scores,
            output_dir=str(sniper_dir),
            skip_reason=sniper_fail,
        )
        stages.append(sniper_result)
        if not sniper_passed:
            verdict = UnifiedLadderVerdict(
                passed=False,
                reason=sniper_fail or "failed_qualifier_nearest_sniper",
                stages=tuple(stages),
                challenger_checkpoint=str(checkpoint_path.resolve()),
                enforcement=spec.enforcement,
            )
            write_unified_verdict(output_dir, verdict)
            return verdict

    if skip_stage2 or qualifier_mode:
        verdict = UnifiedLadderVerdict(
            passed=True,
            reason="qualifier_cleared" if qualifier_mode else "stage1_only",
            stages=tuple(stages),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
            incumbent_swap=qualifier_mode,
        )
        write_unified_verdict(output_dir, verdict)
        return verdict

    incumbent = resolve_incumbent(spec, campaign=campaign, output_root=root)
    block_reason = validate_spec_for_stage2(
        spec, incumbent_resolved=incumbent is not None
    )
    if block_reason is not None:
        stage2_result = UnifiedStageResult(
            name=spec.stage2.name,
            passed=False,
            skip_reason=block_reason,
        )
        verdict = UnifiedLadderVerdict(
            passed=not spec.enforcement,
            reason=block_reason,
            stages=(stage1_result, stage2_result),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
        )
        write_unified_verdict(output_dir, verdict)
        return verdict

    assert incumbent is not None
    stage2_dir = output_dir / "stage2_incumbent"
    stage2_schedules = schedule_stage2_matches(challenger, incumbent, spec)
    try:
        stage2_outcomes, _ = _execute_schedules(
            stage2_schedules,
            output_dir=stage2_dir,
            spec=spec,
            run_match_fn=run_match_fn,
            match_index_start=len(stage1_schedules),
        )
    except TournamentTimingError as exc:
        verdict = UnifiedLadderVerdict(
            passed=False,
            reason=f"stage2_timing_error:{exc}",
            stages=(
                stage1_result,
                UnifiedStageResult(
                    name=spec.stage2.name,
                    passed=False,
                    output_dir=str(stage2_dir),
                ),
            ),
            challenger_checkpoint=str(checkpoint_path.resolve()),
            enforcement=spec.enforcement,
        )
        write_unified_verdict(output_dir, verdict)
        return verdict

    stage2_summary = stage2_per_seed_summary(
        stage2_outcomes,
        challenger_id=challenger_id,
        seeds=spec.stage2.seeds,
    )
    all_perfect = bool(stage2_summary["all_seeds_perfect"])
    stage2_result = UnifiedStageResult(
        name=spec.stage2.name,
        passed=all_perfect,
        per_seed_combined=list(stage2_summary["per_seed_combined"]),
        all_seeds_perfect=all_perfect,
        output_dir=str(stage2_dir),
        skip_reason=None if all_perfect else "incumbent_not_defeated",
    )
    passed = all_perfect
    reason = "pass" if passed else "incumbent_not_defeated"
    verdict = UnifiedLadderVerdict(
        passed=passed,
        reason=reason,
        stages=(stage1_result, stage2_result),
        challenger_checkpoint=str(checkpoint_path.resolve()),
        incumbent_swap=passed,
        enforcement=spec.enforcement,
    )
    write_unified_verdict(output_dir, verdict)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "tournament_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"_{uuid4().hex[:8]}",
            "unified": True,
            "verdict_path": str(output_dir / "unified_verdict.json"),
            "passed": passed,
            "reason": reason,
        },
    )
    return verdict
