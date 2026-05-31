"""Tournament evaluation value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import TrainConfig
from src.jax.submission_runtime import SubmissionReadyAgent

AgentActFn = Callable[[object], list[list[float | int]]]
AgentHandle = AgentActFn | SubmissionReadyAgent


@dataclass(slots=True)
class AgentEntry:
    """Checkpoint-backed agent participating in a tournament."""

    agent_id: str
    checkpoint_path: Path
    cfg: TrainConfig
    act_fn: AgentHandle | None = None


@dataclass(slots=True)
class MatchOutcome:
    """Single Kaggle-env episode result."""

    match_id: str
    format_name: str
    seed: int
    agent_ids: tuple[str, ...]
    rewards: dict[str, float]
    results: dict[str, str]
    placements: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class LeaderboardRow:
    """Aggregated tournament stats for one agent."""

    agent_id: str
    checkpoint_path: str
    games_played: int
    win_rate_vs_sniper: float | None = None
    win_rate_vs_incumbent: float | None = None
    first_place_rate_4p: float | None = None
    gates_passed: bool = False
    gate_reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class TournamentResult:
    """Full tournament output bundle."""

    tournament_id: str
    output_dir: Path
    outcomes: tuple[MatchOutcome, ...]
    leaderboard: tuple[LeaderboardRow, ...]
    pairwise_win_rates: dict[str, dict[str, float]] = field(default_factory=dict)
