"""SSOT pipeline update metrics (qualifier stage, rollout mix, held-out win rates)."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric

_SSOT_BY_NAME: dict[str, MetricDefinition] = {
    "ssot_qualifier_phase": metric(
        "ssot_qualifier_phase",
        "events",
        "SSOT qualifier phase (qualifier stage name, main, or weak_config).",
    ),
    "ssot_qualifier_stage": metric(
        "ssot_qualifier_stage",
        "curriculum",
        "Active SSOT tournament qualifier stage (1–4).",
    ),
}

for _family in (
    "latest",
    "historical",
    "random",
    "noop",
    "nearest_sniper",
    "turtle",
    "opportunistic",
):
    _SSOT_BY_NAME[f"ssot_rollout_family_prob_{_family}"] = metric(
        f"ssot_rollout_family_prob_{_family}",
        "opponent_composition",
        f"SSOT rollout sampling probability for {_family} at the active qualifier stage.",
    )

for _opponent in ("random", "noop", "nearest_sniper"):
    _SSOT_BY_NAME[f"ssot_qualifier_win_rate_{_opponent}"] = metric(
        f"ssot_qualifier_win_rate_{_opponent}",
        "core_progress",
        f"Held-out tournament qualifier win rate vs {_opponent} on eval seeds.",
    )


def ssot_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_SSOT_BY_NAME[name] for name in _SSOT_BY_NAME)
