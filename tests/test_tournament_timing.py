from __future__ import annotations

import pytest

from src.artifacts.tournament.timing import StepTimingBudget, TournamentTimingError


def test_step_timing_budget_allows_cumulative_overage_up_to_cap() -> None:
    budget = StepTimingBudget(per_step_seconds=1.0, overage_budget_seconds=0.5)
    budget.record(1.2)
    budget.record(1.3)
    with pytest.raises(TournamentTimingError):
        budget.record(1.1)


def test_step_timing_budget_accepts_actions_within_limit() -> None:
    budget = StepTimingBudget(per_step_seconds=1.0, overage_budget_seconds=60.0)
    for _ in range(10):
        budget.record(0.05)
    summary = budget.summary()
    assert summary["agent_calls"] == 10
    assert summary["total_overage_seconds"] == 0.0
