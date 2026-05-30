"""Orchestrate local tournament evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from src.artifacts.run_paths import atomic_write_json
from src.config.schema import PromotionTournamentConfig, TournamentConfig

from .ranking import aggregate_pairwise_win_rates, build_leaderboard, evaluate_gates
from .runner import build_baseline_agent, run_match
from .types import AgentEntry, MatchOutcome, TournamentResult


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
    return schedules


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
    seeds = _match_seeds(cfg)
    games_per_pair = max(int(cfg.games_per_pair), 1)
    schedules = _schedule_matches(candidates, incumbent=incumbent, cfg=cfg)

    outcomes: list[MatchOutcome] = []
    match_index = 0
    for format_name, baseline_name, agent_ids, agents in schedules:
        for seed in seeds:
            for _ in range(games_per_pair):
                match_id = f"{format_name}_{match_index:04d}"
                outcome, env = run_match(
                    match_id=match_id,
                    format_name=format_name,
                    seed=seed + match_index,
                    agent_ids=agent_ids,
                    agents=agents,
                    max_steps=int(cfg.max_steps),
                )
                outcomes.append(outcome)
                if cfg.write_replays:
                    replay_path = output_dir / "matches" / f"{match_id}.html"
                    replay_path.parent.mkdir(parents=True, exist_ok=True)
                    replay_path.write_text(env.render(mode="html"), encoding="utf-8")
                    match_json = output_dir / "matches" / f"{match_id}.json"
                    atomic_write_json(
                        match_json,
                        {
                            "match_id": match_id,
                            "format_name": format_name,
                            "baseline_name": baseline_name,
                            "seed": seed + match_index,
                            "agent_ids": list(agent_ids),
                            "results": outcome.results,
                            "rewards": outcome.rewards,
                            "placements": outcome.placements,
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
    atomic_write_json(output_dir / "leaderboard.json", leaderboard_payload)
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
        },
    )
    if pairwise:
        atomic_write_json(output_dir / "pairwise.json", pairwise)

    return TournamentResult(
        tournament_id=tournament_id,
        output_dir=output_dir,
        outcomes=outcome_tuple,
        leaderboard=leaderboard,
        pairwise_win_rates=pairwise,
    )
