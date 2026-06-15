"""Telemetry metric definitions for the action_decision group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric

_ACTION_DECISION_BY_NAME: dict[str, MetricDefinition] = {
    "stop_rate": metric(
        "stop_rate",
        "action_decision",
        "Fraction of active factorized launch steps where the stop head fired.",
        rollout_scalar_role="base_sum",
    ),
    "mean_active_launches_per_turn": metric(
        "mean_active_launches_per_turn",
        "action_decision",
        "Mean non-stop launches with a positive ship bucket per env-turn.",
        rollout_scalar_role="base_sum",
    ),
    "win_rate_delta_10": metric(
        "win_rate_delta_10",
        "action_decision",
        "Last-window minus first-window overall_win_rate over 10 updates.",
    ),
    "win_rate_recovery_delta_10": metric(
        "win_rate_recovery_delta_10",
        "action_decision",
        "Last 10-update overall_win_rate mean minus the weakest prior 10-update mean.",
    ),
    "win_rate_window_mean_10": metric(
        "win_rate_window_mean_10",
        "action_decision",
        "Mean overall_win_rate over the latest 10 training updates.",
    ),
    "win_rate_best_window_mean_10": metric(
        "win_rate_best_window_mean_10",
        "action_decision",
        "Best rolling 10-update mean overall_win_rate observed so far.",
    ),
    "approx_kl_window_mean": metric(
        "approx_kl_window_mean",
        "action_decision",
        "Mean approx_kl over the last 10 training updates (preflight-aligned).",
    ),
    "entropy_window_mean": metric(
        "entropy_window_mean",
        "action_decision",
        "Mean policy entropy over the last 10 training updates (preflight-aligned).",
    ),
    "entropy_delta_10": metric(
        "entropy_delta_10",
        "action_decision",
        "Latest 10-update entropy mean minus the first tracked 10-update entropy mean.",
    ),
    "entropy_retention_ratio_10": metric(
        "entropy_retention_ratio_10",
        "action_decision",
        "Latest 10-update entropy mean divided by the first tracked 10-update entropy mean.",
    ),
    "preflight_sweep_score": metric(
        "preflight_sweep_score",
        "action_decision",
        "W&B preflight sweep objective: running best eligible score logged each update.",
    ),
    "preflight_sweep_score_update": metric(
        "preflight_sweep_score_update",
        "action_decision",
        "Per-update preflight sweep score before running-best aggregation.",
    ),
    "stop_utilization_ratio": metric(
        "stop_utilization_ratio",
        "action_decision",
        "mean_active_launches_per_turn divided by model.max_moves_k (L1 gate).",
    ),
}


def action_decision_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_ACTION_DECISION_BY_NAME[name] for name in _ACTION_DECISION_BY_NAME)
