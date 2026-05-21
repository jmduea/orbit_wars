from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.conf_schema import TrainConfig
from src.metric_registry import (
    DEFAULT_ENABLED_GROUPS,
    KNOWN_SWEEP_OBJECTIVE_METRIC_NAMES,
    METRIC_DEFINITIONS,
    METRIC_GROUPS,
    enabled_metric_names,
    filter_metric_record,
    metric_definition,
    protected_metric_names,
)


def _metric_groups(**overrides: bool) -> SimpleNamespace:
    values = {
        group_name: group_name in DEFAULT_ENABLED_GROUPS for group_name in METRIC_GROUPS
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_metric_registry_names_are_unique_and_grouped():
    names = [definition.name for definition in METRIC_DEFINITIONS]
    assert len(names) == len(set(names))
    assert all(metric_definition(name).group in METRIC_GROUPS for name in names)


def test_filter_metric_record_keeps_protected_metrics_when_groups_are_disabled():
    groups = _metric_groups(timing=False, action_decision=False)
    record = {
        "update": 5,
        "total_env_steps": 1024,
        "completed_episodes": 12,
        "samples": 256,
        "episode_reward_mean": 0.75,
        "overall_win_rate": 0.6,
        "env_steps_per_sec": 1100.0,
        "noop_percent": 25.0,
        "policy_loss": 0.1,
    }

    filtered = filter_metric_record(
        record,
        metric_groups_cfg=groups,
        record_kind="update",
        extra_protected_names=protected_metric_names(),
    )

    assert "noop_percent" not in filtered
    assert filtered["env_steps_per_sec"] == 1100.0
    assert filtered["episode_reward_mean"] == 0.75
    assert filtered["policy_loss"] == 0.1


def test_filter_metric_record_filters_event_fields_but_keeps_checkpoint_operational_fields():
    groups = _metric_groups(events=False)
    record = {
        "event": "checkpoint_result",
        "update": 7,
        "checkpoint_status": "committed",
        "checkpoint_final": False,
        "checkpoint_reason": "periodic",
        "checkpoint_error": None,
        "metric": "overall_win_rate",
        "metric_value": 0.8,
    }

    filtered = filter_metric_record(
        record,
        metric_groups_cfg=groups,
        record_kind="event",
    )

    assert filtered == {
        "event": "checkpoint_result",
        "update": 7,
        "checkpoint_status": "committed",
        "checkpoint_final": False,
        "checkpoint_reason": "periodic",
        "checkpoint_error": None,
    }


def test_enabled_metric_names_include_dynamic_retention_and_plateau_metrics():
    cfg = TrainConfig()
    cfg.checkpoint_retention.best_metric_name = "total_loss"
    cfg.plateau_metric = "policy_loss"
    groups = _metric_groups(losses=False)

    names = enabled_metric_names(
        groups,
        record_kind="update",
        extra_protected_names=protected_metric_names(cfg),
    )

    assert "policy_loss" in names
    assert "total_loss" in names


def test_repo_sweep_metrics_are_registered_and_enabled_by_default():
    sweep_dir = Path(__file__).resolve().parents[1] / "conf" / "sweeps"
    metric_names = set(KNOWN_SWEEP_OBJECTIVE_METRIC_NAMES)
    for path in sweep_dir.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        metric_names.add(str(payload["metric"]["name"]))

    names = enabled_metric_names(
        _metric_groups(),
        record_kind="update",
        extra_protected_names=protected_metric_names(),
    )
    for metric_name in metric_names:
        assert metric_definition(metric_name).record_kinds == frozenset({"update"})
        assert metric_name in names
