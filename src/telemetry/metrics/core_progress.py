"""Telemetry metric definitions for the core_progress group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_CORE_PROGRESS_BY_NAME: dict[str, MetricDefinition] = {
    "update": metric(
        "update",
        "core_progress",
        "Completed PPO update index.",
        record_kinds=("update", "event"),
        protected=True,
    ),
    "total_env_steps": metric(
        "total_env_steps",
        "core_progress",
        "Cumulative environment steps processed so far.",
        protected=True,
    ),
    "completed_episodes": metric(
        "completed_episodes",
        "core_progress",
        "Completed episodes across all rollout groups.",
        protected=True,
    ),
    "samples": metric(
        "samples",
        "core_progress",
        "Learner decision samples consumed by the update.",
        protected=True,
        rollout_scalar_role="base_sum",
    ),
    "win_rate_2p": metric(
        "win_rate_2p",
        "core_progress",
        "First-place win rate in 2-player episodes.",
        protected=True,
        rollout_scalar_role="finalized_rate",
    ),
    "first_place_rate_4p": metric(
        "first_place_rate_4p",
        "core_progress",
        "First-place rate in 4-player episodes.",
        protected=True,
        rollout_scalar_role="finalized_rate",
    ),
    "average_placement_4p": metric(
        "average_placement_4p",
        "core_progress",
        "Average final placement in completed 4-player episodes.",
        rollout_scalar_role="finalized_rate",
    ),
    "overall_win_rate": metric(
        "overall_win_rate",
        "core_progress",
        "Overall first-place rate across completed episodes.",
        protected=True,
        rollout_scalar_role="finalized_rate",
    ),
    "average_reward": metric(
        "average_reward",
        "core_progress",
        "Mean per-step reward over the rollout.",
        rollout_scalar_role="base_sum",
    ),
    "episode_reward_mean": metric(
        "episode_reward_mean",
        "core_progress",
        "Mean episodic reward across completed episodes.",
        protected=True,
        rollout_scalar_role="base_sum",
    ),
}


def core_progress_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_CORE_PROGRESS_BY_NAME[name] for name in _CORE_PROGRESS_BY_NAME)
