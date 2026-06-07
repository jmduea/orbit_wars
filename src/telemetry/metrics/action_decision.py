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
    "planet_flow_sweep_score": metric(
        "planet_flow_sweep_score",
        "action_decision",
        "W&B sweep objective: win_rate_delta_10 when window-mean KL/entropy floors pass, else -1.",
    ),
    "preflight_sweep_score": metric(
        "preflight_sweep_score",
        "action_decision",
        "W&B preflight sweep objective: win_rate_delta_10 when Gates 2–3 floors pass, else -1.",
    ),
    "stop_utilization_ratio": metric(
        "stop_utilization_ratio",
        "action_decision",
        "mean_active_launches_per_turn divided by model.max_moves_k (L1 gate).",
    ),
}


def action_decision_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_ACTION_DECISION_BY_NAME[name] for name in _ACTION_DECISION_BY_NAME)
