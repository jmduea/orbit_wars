"""Local Kaggle-env tournament evaluation and ranking."""

from .eval import run_tournament
from .ranking import build_leaderboard, evaluate_gates
from .resolve import (
    ShortlistResolveResult,
    run_context_for_agent,
    validate_agents_feature_compatible,
)
from .types import AgentEntry, LeaderboardRow, MatchOutcome, TournamentResult

__all__ = [
    "AgentEntry",
    "LeaderboardRow",
    "MatchOutcome",
    "ShortlistResolveResult",
    "TournamentResult",
    "build_leaderboard",
    "evaluate_gates",
    "run_context_for_agent",
    "run_tournament",
    "validate_agents_feature_compatible",
]
