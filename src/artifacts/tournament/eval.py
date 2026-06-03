"""Orchestrate local tournament evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from src.artifacts.run_paths import atomic_write_json
from src.artifacts.timing import TournamentTimingError
from src.config.schema import PromotionTournamentConfig, TournamentConfig

from .ranking import aggregate_pairwise_win_rates, build_leaderboard, evaluate_gates
from .runner import build_baseline_agent, run_match
from .types import AgentEntry, MatchOutcome, TournamentResult


def _tournament_timing(cfg: TournamentConfig) -> tuple[float, float]:
    return (
        float(getattr(cfg, "per_step_seconds", 1.0)),
        float(getattr(cfg, "overage_budget_seconds", 60.0)),
    )


def _match_seeds(cfg: TournamentConfig) -> list[int]:
    seeds = [int(seed) for seed in cfg.seeds]
    if not seeds:
        seeds = [0]
    return seeds


def _schedule_matches(
    candidates: Sequence[AgentEntry],
    *,
    incumbent: AgentEntry | None,
    cfg: TournamentConfig,
) -> list[tuple[str, str, tuple[str, ...], list[object]]]:
    """Return (format_name, baseline_name, agent_ids, agents) schedules."""

    schedules: list[tuple[str, str, tuple[str, ...], list[object]]] = []
    baseline_name = cfg.baselines[0] if cfg.baselines else "sniper"
    baseline_id = f"baseline:{baseline_name}"
    baseline_agent = build_baseline_agent(baseline_name)

    formats = {value.strip() for value in cfg.formats}
    if "2p_vs_baseline" in formats:
        for candidate in candidates:
            schedules.append(
                (
                    "2p_vs_baseline",
                    baseline_name,
                    (candidate.agent_id, baseline_id),
                    [candidate.act_fn, baseline_agent],
                )
            )
    if "2p_head_to_head" in formats and incumbent is not None:
        for candidate in candidates:
            if candidate.agent_id == incumbent.agent_id:
                continue
            schedules.append(
                (
                    "2p_head_to_head",
                    "incumbent",
                    (candidate.agent_id, incumbent.agent_id),
                    [candidate.act_fn, incumbent.act_fn],
                )
            )
    if "4p_free_for_all" in formats and len(candidates) >= 4:
        top = list(candidates[:4])
        schedules.append(
            (
                "4p_free_for_all",
                "mixed",
                tuple(agent.agent_id for agent in top),
                [agent.act_fn for agent in top],
            )
        )
    if "4p_challenger_vs_baselines" in formats and len(candidates) == 1:
        candidate = candidates[0]
        fillers = cfg.baselines[:3] if len(cfg.baselines) >= 3 else ["noop", "random", "random"]
        filler_agents = [build_baseline_agent(name) for name in fillers]
        filler_ids = tuple(f"baseline:{name}" for name in fillers)
        schedules.append(
            (
                "4p_challenger_vs_baselines",
                "mixed",
                (candidate.agent_id, *filler_ids),
                [candidate.act_fn, *filler_agents],
            )
        )
    return schedules


def _write_match_record(
    output_dir: Path,
    *,
    match_id: str,
    format_name: str,
    baseline_name: str,
    seed: int,
    outcome: MatchOutcome,
    timing_summary: dict[str, object],
) -> None:
    match_json = output_dir / "matches" / f"{match_id}.json"
    match_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        match_json,
        {
            "match_id": match_id,
            "format_name": format_name,
            "baseline_name": baseline_name,
            "seed": seed,
            "agent_ids": list(outcome.agent_ids),
            "results": outcome.results,
            "rewards": outcome.rewards,
            "placements": outcome.placements,
            "timing": timing_summary,
        },
    )


def run_tournament(
    candidates: Sequence[AgentEntry],
    *,
    cfg: TournamentConfig,
    output_dir: Path,
    incumbent: AgentEntry | None = None,
    promotion_gates: PromotionTournamentConfig | None = None,
) -> TournamentResult:
    """Run configured tournament formats and write leaderboard artifacts."""

    if not candidates:
        raise ValueError("tournament requires at least one candidate agent.")

    for candidate in candidates:
        if candidate.act_fn is None:
            raise ValueError(f"candidate {candidate.agent_id!r} is missing act_fn.")

    output_dir.mkdir(parents=True, exist_ok=True)
    tournament_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{uuid4().hex[:8]}"
    per_step_seconds, overage_budget_seconds = _tournament_timing(cfg)
    seeds = _match_seeds(cfg)
    games_per_pair = max(int(cfg.games_per_pair), 1)
    schedules = _schedule_matches(candidates, incumbent=incumbent, cfg=cfg)
    total_matches = len(schedules) * len(seeds) * games_per_pair
    atomic_write_json(
        output_dir / "progress.json",
        {
            "tournament_id": tournament_id,
            "status": "running",
            "match_count_total": total_matches,
            "match_count_completed": 0,
        },
    )
    print(
        f"tournament {tournament_id}: {total_matches} matches scheduled "
        f"({len(candidates)} candidates, formats={list(cfg.formats)})",
        flush=True,
    )

    outcomes: list[MatchOutcome] = []
    match_index = 0
    for format_name, baseline_name, agent_ids, agents in schedules:
        for seed in seeds:
            for _ in range(games_per_pair):
                match_id = f"{format_name}_{match_index:04d}"
                match_seed = seed + match_index
                print(
                    f"tournament match {match_index + 1}/{total_matches}: "
                    f"{format_name} {' vs '.join(agent_ids)} seed={match_seed}",
                    flush=True,
                )
                try:
                    outcome, env, timing_summary = run_match(
                        match_id=match_id,
                        format_name=format_name,
                        seed=match_seed,
                        agent_ids=agent_ids,
                        agents=agents,
                        max_steps=int(cfg.max_steps),
                        per_step_seconds=per_step_seconds,
                        overage_budget_seconds=overage_budget_seconds,
                    )
                except TournamentTimingError as exc:
                    atomic_write_json(
                        output_dir / "progress.json",
                        {
                            "tournament_id": tournament_id,
                            "status": "failed",
                            "failed_match_id": match_id,
                            "error": str(exc),
                            "match_count_total": total_matches,
                            "match_count_completed": match_index,
                        },
                    )
                    raise
                outcomes.append(outcome)
                _write_match_record(
                    output_dir,
                    match_id=match_id,
                    format_name=format_name,
                    baseline_name=baseline_name,
                    seed=match_seed,
                    outcome=outcome,
                    timing_summary=timing_summary,
                )
                if cfg.write_replays:
                    replay_path = output_dir / "matches" / f"{match_id}.html"
                    replay_path.write_text(env.render(mode="html"), encoding="utf-8")
                primary = agent_ids[0]
                print(
                    f"tournament match {match_index + 1}/{total_matches} done: "
                    f"{primary} -> {outcome.results.get(primary, '?')} "
                    f"({float(timing_summary['match_seconds']):.1f}s, "
                    f"steps={timing_summary['env_steps']}, "
                    f"max_action={float(timing_summary['max_action_seconds']):.3f}s)",
                    flush=True,
                )
                atomic_write_json(
                    output_dir / "progress.json",
                    {
                        "tournament_id": tournament_id,
                        "status": "running",
                        "match_count_total": total_matches,
                        "match_count_completed": match_index + 1,
                        "last_match_id": match_id,
                    },
                )
                match_index += 1

    outcome_tuple = tuple(outcomes)
    leaderboard = build_leaderboard(
        tuple(candidates),
        outcome_tuple,
        incumbent_id=incumbent.agent_id if incumbent is not None else None,
        baseline_name=cfg.baselines[0] if cfg.baselines else "sniper",
    )
    pairwise = aggregate_pairwise_win_rates(outcome_tuple)

    if promotion_gates is not None:
        incumbent_present = incumbent is not None
        updated_rows: list = []
        for row in leaderboard:
            passed, reasons = evaluate_gates(
                row,
                promotion_gates,
                incumbent_present=incumbent_present,
            )
            updated_rows.append(
                type(row)(
                    agent_id=row.agent_id,
                    checkpoint_path=row.checkpoint_path,
                    games_played=row.games_played,
                    win_rate_vs_sniper=row.win_rate_vs_sniper,
                    win_rate_vs_incumbent=row.win_rate_vs_incumbent,
                    first_place_rate_4p=row.first_place_rate_4p,
                    gates_passed=passed,
                    gate_reasons=reasons,
                )
            )
        leaderboard = tuple(updated_rows)

    leaderboard_payload = [
        {
            "agent_id": row.agent_id,
            "checkpoint_path": row.checkpoint_path,
            "games_played": row.games_played,
            "win_rate_vs_sniper": row.win_rate_vs_sniper,
            "win_rate_vs_incumbent": row.win_rate_vs_incumbent,
            "first_place_rate_4p": row.first_place_rate_4p,
            "gates_passed": row.gates_passed,
            "gate_reasons": list(row.gate_reasons),
        }
        for row in leaderboard
    ]
    atomic_write_json(output_dir / "leaderboard.json", {"rows": leaderboard_payload})
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "tournament_id": tournament_id,
            "candidate_count": len(candidates),
            "incumbent_id": incumbent.agent_id if incumbent is not None else None,
            "match_count": len(outcomes),
            "formats": list(cfg.formats),
            "seeds": seeds,
            "games_per_pair": games_per_pair,
            "per_step_seconds": per_step_seconds,
            "overage_budget_seconds": overage_budget_seconds,
        },
    )
    if pairwise:
        atomic_write_json(output_dir / "pairwise.json", pairwise)
    atomic_write_json(
        output_dir / "progress.json",
        {
            "tournament_id": tournament_id,
            "status": "completed",
            "match_count_total": total_matches,
            "match_count_completed": total_matches,
        },
    )
    print(f"tournament {tournament_id}: completed {total_matches} matches", flush=True)

    return TournamentResult(
        tournament_id=tournament_id,
        output_dir=output_dir,
        outcomes=outcome_tuple,
        leaderboard=leaderboard,
        pairwise_win_rates=pairwise,
    )
