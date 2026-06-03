"""Match scheduling helpers for unified tournament stages."""

from __future__ import annotations

from typing import Sequence

from src.artifacts.tournament.runner import build_baseline_agent
from src.artifacts.tournament.types import AgentEntry
from src.artifacts.tournament.unified.spec import StageSpec, UnifiedTournamentSpec

ScheduleEntry = tuple[str, str, tuple[str, ...], list[object], int]


def validate_four_p_fillers(spec: UnifiedTournamentSpec) -> None:
    if len(spec.four_p_baseline_fillers) < 3:
        raise ValueError(
            "4p_challenger_vs_baselines requires three baseline filler slots"
        )


def schedule_stage1_matches(
    challenger: AgentEntry,
    spec: UnifiedTournamentSpec,
    *,
    stage: StageSpec | None = None,
) -> list[ScheduleEntry]:
    """Schedule Stage 1 prerequisite matches (noop then random 2p + shared 4p leg)."""

    validate_four_p_fillers(spec)
    stage = stage or spec.stage1
    schedules: list[ScheduleEntry] = []
    challenger_id = challenger.agent_id

    for opponent in stage.opponents:
        baseline_id = f"baseline:{opponent}"
        baseline_agent = build_baseline_agent(opponent)
        if "2p_vs_baseline" in stage.formats:
            for seed in stage.seeds:
                for _ in range(stage.games_per_pair):
                    schedules.append(
                        (
                            "2p_vs_baseline",
                            opponent,
                            (challenger_id, baseline_id),
                            [challenger.act_fn, baseline_agent],
                            seed,
                        )
                    )
    if "4p_challenger_vs_baselines" in stage.formats:
        fillers = spec.four_p_baseline_fillers[:3]
        filler_agents = [build_baseline_agent(name) for name in fillers]
        filler_ids = tuple(f"baseline:{name}" for name in fillers)
        for seed in stage.seeds:
            for _ in range(stage.games_per_pair):
                schedules.append(
                    (
                        "4p_challenger_vs_baselines",
                        "mixed",
                        (challenger_id, *filler_ids),
                        [challenger.act_fn, *filler_agents],
                        seed,
                    )
                )
    return schedules


def schedule_stage2_matches(
    challenger: AgentEntry,
    incumbent: AgentEntry,
    spec: UnifiedTournamentSpec,
    *,
    stage: StageSpec | None = None,
) -> list[ScheduleEntry]:
    """Schedule Stage 2 incumbent ladder matches."""

    validate_four_p_fillers(spec)
    stage = stage or spec.stage2
    schedules: list[ScheduleEntry] = []
    challenger_id = challenger.agent_id
    incumbent_id = incumbent.agent_id

    if "2p_head_to_head" in stage.formats:
        for seed in stage.seeds:
            for _ in range(stage.games_per_pair):
                schedules.append(
                    (
                        "2p_head_to_head",
                        "incumbent",
                        (challenger_id, incumbent_id),
                        [challenger.act_fn, incumbent.act_fn],
                        seed,
                    )
                )
    if "4p_challenger_vs_baselines" in stage.formats:
        fillers = spec.four_p_baseline_fillers[:3]
        filler_agents = [build_baseline_agent(name) for name in fillers]
        filler_ids = tuple(f"baseline:{name}" for name in fillers)
        for seed in stage.seeds:
            for _ in range(stage.games_per_pair):
                schedules.append(
                    (
                        "4p_challenger_vs_baselines",
                        "mixed",
                        (challenger_id, *filler_ids),
                        [challenger.act_fn, *filler_agents],
                        seed,
                    )
                )
    return schedules


def count_scheduled_matches(
    schedules: Sequence[ScheduleEntry],
    *,
    seeds: tuple[int, ...],
    games_per_pair: int,
    opponents: tuple[str, ...],
    include_4p: bool,
) -> int:
    """Return expected schedule size for dry-run validation."""

    per_opponent_2p = len(seeds) * games_per_pair if opponents else 0
    per_seed_4p = len(seeds) * games_per_pair if include_4p else 0
    return len(opponents) * per_opponent_2p + per_seed_4p
