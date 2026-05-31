"""Kaggle per-action timing limits for tournament and submission validation."""

from __future__ import annotations


class TournamentTimingError(RuntimeError):
    """Raised when cumulative agent-action latency exceeds the allowed overage budget."""

    def __init__(
        self,
        message: str,
        *,
        total_overage_seconds: float,
        overage_budget_seconds: float,
        per_step_seconds: float,
        last_elapsed_seconds: float,
    ) -> None:
        super().__init__(message)
        self.total_overage_seconds = total_overage_seconds
        self.overage_budget_seconds = overage_budget_seconds
        self.per_step_seconds = per_step_seconds
        self.last_elapsed_seconds = last_elapsed_seconds


class StepTimingBudget:
    """Track per-action latency against a per-step limit and cumulative overage cap."""

    def __init__(self, per_step_seconds: float, overage_budget_seconds: float) -> None:
        self.per_step_seconds = float(per_step_seconds)
        self.overage_budget_seconds = float(overage_budget_seconds)
        self.total_overage_seconds = 0.0
        self.latencies: list[float] = []

    def record(self, elapsed: float) -> None:
        elapsed = float(elapsed)
        self.latencies.append(elapsed)
        self.total_overage_seconds += max(0.0, elapsed - self.per_step_seconds)
        if self.total_overage_seconds > self.overage_budget_seconds:
            raise TournamentTimingError(
                "cumulative agent-action overage "
                f"{self.total_overage_seconds:.3f}s exceeded budget "
                f"{self.overage_budget_seconds:.3f}s "
                f"(last action {elapsed:.3f}s, per-action limit {self.per_step_seconds:.3f}s)",
                total_overage_seconds=self.total_overage_seconds,
                overage_budget_seconds=self.overage_budget_seconds,
                per_step_seconds=self.per_step_seconds,
                last_elapsed_seconds=elapsed,
            )

    def summary(self) -> dict[str, float | int]:
        latencies = list(self.latencies)
        mean_action = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "agent_calls": len(latencies),
            "max_action_seconds": max(latencies) if latencies else 0.0,
            "mean_action_seconds": mean_action,
            "total_overage_seconds": self.total_overage_seconds,
            "per_step_seconds": self.per_step_seconds,
            "overage_budget_seconds": self.overage_budget_seconds,
        }
