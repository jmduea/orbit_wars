from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from src.config.schema import TrainConfig
from src.config import compose_hydra_train_config, train_config_from_omegaconf
from src.jax.rollout.metric_contract import (
    BASE_ROLLOUT_SCALAR_KEYS,
    FINALIZED_ROLLOUT_RATE_KEYS,
    LOGGED_ROLLOUT_SCALAR_KEYS,
    ROLLOUT_ALLOWED_SCALAR_KEYS,
    ROLLOUT_INTERNAL_SCALAR_KEYS,
)
from src.telemetry.metric_registry import (
    DEFAULT_ENABLED_GROUPS,
    KNOWN_SWEEP_OBJECTIVE_METRIC_NAMES,
    METRIC_DEFINITIONS,
    METRIC_DEFINITIONS_BY_NAME,
    METRIC_GROUPS,
    ROLLOUT_OUTPUT_METRIC_NAMES,
    ROLLOUT_SCALAR_ORDER,
    enabled_metric_names,
    filter_metric_record,
    filter_update_record,
    metric_definition,
    protected_metric_names,
    prune_scalar_metrics,
    rollout_compute_scalar_keys,
)


def _metric_groups(**overrides: bool) -> SimpleNamespace:
    values = {
        group_name: group_name in DEFAULT_ENABLED_GROUPS for group_name in METRIC_GROUPS
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hydra_config_supports_metric_group_overrides():
    cfg = compose_hydra_train_config(
        [
            "training.total_updates=1",
            "telemetry.metric_groups.trajectory_shield_debug=true",
            "telemetry.metric_groups.losses=false",
        ]
    )

    assert cfg.telemetry.metric_groups.trajectory_shield_debug is True
    assert cfg.telemetry.metric_groups.losses is False
    assert cfg.telemetry.metric_groups.core_progress is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training.plateau_metric", "opponent_composition"),
        ("training.plateau_metric", "curriculum_stage_id"),
        ("artifacts.checkpoint_retention.best_metric_name", "seed_scheduler_policy"),
    ],
)
def test_non_scalar_or_unknown_retention_metrics_are_rejected(field: str, value: str):
    cfg = OmegaConf.structured(TrainConfig)
    OmegaConf.update(cfg, field, value)

    with pytest.raises(
        ValueError, match="registered canonical scalar telemetry metric"
    ):
        train_config_from_omegaconf(cfg)


def test_filter_update_record_preserves_configured_retention_metric():
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.losses = False
    cfg.telemetry.metric_groups.opponent_composition = False
    cfg.artifacts.checkpoint_retention.best_metric_name = "total_loss"

    record = {
        "update": 3,
        "total_env_steps": 300,
        "completed_episodes": 7,
        "samples": 128,
        "overall_win_rate": 0.5,
        "win_rate_2p": 0.5,
        "first_place_rate_4p": 0.0,
        "episode_reward_mean": 0.25,
        "env_steps_per_sec": 900.0,
        "total_loss": 1.75,
        "opponent_slots_total": 8.0,
    }

    filtered = filter_update_record(record, cfg)

    assert filtered["total_loss"] == 1.75
    assert "opponent_slots_total" not in filtered


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
        "policy_loss": 0.1,
        "stop_rate": 0.25,
    }

    filtered = filter_metric_record(
        record,
        metric_groups_cfg=groups,
        record_kind="update",
        extra_protected_names=protected_metric_names(),
    )

    assert "stop_rate" not in filtered
    assert filtered["env_steps_per_sec"] == 1100.0
    assert filtered["episode_reward_mean"] == 0.75
    assert filtered["policy_loss"] == 0.1


def test_filter_update_record_omits_disabled_group_keys():
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.action_decision = False

    record = {
        "update": 2,
        "total_env_steps": 200,
        "completed_episodes": 4,
        "samples": 64,
        "episode_reward_mean": 0.5,
        "overall_win_rate": 0.5,
        "env_steps_per_sec": 800.0,
        "stop_rate": 0.1,
        "mean_active_launches_per_turn": 1.2,
    }

    filtered = filter_update_record(record, cfg)

    assert "stop_rate" not in filtered
    assert "mean_active_launches_per_turn" not in filtered
    assert filtered["overall_win_rate"] == 0.5


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
    cfg.artifacts.checkpoint_retention.best_metric_name = "total_loss"
    cfg.training.plateau_metric = "policy_loss"
    groups = _metric_groups(losses=False)

    names = enabled_metric_names(
        groups,
        record_kind="update",
        extra_protected_names=protected_metric_names(cfg),
    )

    assert "policy_loss" in names
    assert "total_loss" in names


def test_per_format_loss_metrics_are_registered_as_losses():
    names = enabled_metric_names(
        _metric_groups(losses=True),
        record_kind="update",
        extra_protected_names=protected_metric_names(),
    )

    for suffix in ("2p", "4p"):
        assert f"policy_loss_{suffix}" in names
        assert f"value_loss_{suffix}" in names
        assert f"entropy_{suffix}" in names
        assert f"approx_kl_{suffix}" in names
        assert f"total_loss_{suffix}" in names
        assert f"loss_sample_count_{suffix}" in names


def test_per_format_timing_metrics_are_registered_as_timing():
    expected_names = {
        "rollout_seconds_2p",
        "rollout_seconds_4p",
        "env_steps_per_sec_2p",
        "env_steps_per_sec_4p",
        "rollout_env_steps_per_sec_2p",
        "rollout_env_steps_per_sec_4p",
        "samples_per_sec_2p",
        "samples_per_sec_4p",
        "rollout_samples_per_sec_2p",
        "rollout_samples_per_sec_4p",
        "update_time_rollout_fraction",
        "update_time_ppo_fraction",
    }
    names = enabled_metric_names(
        _metric_groups(timing=True),
        record_kind="update",
        extra_protected_names=protected_metric_names(),
    )

    for name in expected_names:
        assert metric_definition(name).group == "timing"
        assert name in names


def test_rollout_metric_contract_syncs_with_telemetry_registry() -> None:
    assert ROLLOUT_SCALAR_ORDER == LOGGED_ROLLOUT_SCALAR_KEYS

    registered_logged_keys = frozenset(
        name
        for name in LOGGED_ROLLOUT_SCALAR_KEYS
        if name in METRIC_DEFINITIONS_BY_NAME
    )
    assert ROLLOUT_OUTPUT_METRIC_NAMES == registered_logged_keys
    assert ROLLOUT_OUTPUT_METRIC_NAMES <= frozenset(LOGGED_ROLLOUT_SCALAR_KEYS)

    assert all(key in BASE_ROLLOUT_SCALAR_KEYS for key in ROLLOUT_INTERNAL_SCALAR_KEYS)
    assert not any(key in LOGGED_ROLLOUT_SCALAR_KEYS for key in ROLLOUT_INTERNAL_SCALAR_KEYS)
    assert all(
        key in LOGGED_ROLLOUT_SCALAR_KEYS for key in FINALIZED_ROLLOUT_RATE_KEYS
    )

    logged_and_internal = set(LOGGED_ROLLOUT_SCALAR_KEYS) | set(ROLLOUT_INTERNAL_SCALAR_KEYS)
    unexpected_base_keys = sorted(
        key for key in BASE_ROLLOUT_SCALAR_KEYS if key not in logged_and_internal
    )
    assert unexpected_base_keys == []
    assert ROLLOUT_ALLOWED_SCALAR_KEYS >= logged_and_internal


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


def test_rollout_compute_scalar_keys_omits_disabled_groups():
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.opponent_composition = False
    cfg.telemetry.metric_groups.action_decision = False
    cfg.telemetry.metric_groups.trajectory_shield_debug = False

    keys = rollout_compute_scalar_keys(cfg)

    assert "samples" in keys
    assert "wins_2p" in keys
    assert "opponent_slots_total" not in keys
    assert "stop_rate" not in keys
    assert "trajectory_shield_blocked_count" not in keys


def test_required_ppo_metric_names_keep_plateau_when_losses_disabled():
    from src.telemetry.metric_registry import required_ppo_metric_names

    cfg = TrainConfig()
    cfg.telemetry.metric_groups.losses = False
    cfg.training.plateau_metric = "policy_loss"
    cfg.artifacts.checkpoint_retention.best_metric_name = "total_loss"

    names = required_ppo_metric_names(
        cfg,
        ("policy_loss", "value_loss", "approx_kl", "total_loss", "minibatches"),
    )

    assert "policy_loss" in names
    assert "total_loss" in names
    assert "approx_kl" in names
    assert "value_loss" not in names
    assert "minibatches" not in names


def test_prune_scalar_metrics_drops_disabled_keys():
    pruned = prune_scalar_metrics(
        {"samples": 1.0, "stop_rate": 0.2, "policy_loss": 0.1},
        frozenset({"samples", "policy_loss"}),
    )

    assert pruned == {"samples": 1.0, "policy_loss": 0.1}


def test_merge_metric_dicts_omits_disabled_group_keys():
    import jax
    import jax.numpy as jnp

    from src.jax.train import _merge_metric_dicts

    lean_chunk = {
        "env_steps": jnp.asarray(8.0),
        "average_reward": jnp.asarray(0.5),
        "episode_done": jnp.asarray(0.0),
        "episode_reward_mean": jnp.asarray(0.0),
        "samples": jnp.asarray(16.0),
    }

    merged = _merge_metric_dicts([lean_chunk, lean_chunk])

    assert "stop_rate" not in merged
    assert "opponent_slots_total" not in merged
    assert float(merged["samples"]) == 32.0


def test_filter_update_record_jsonl_omits_disabled_rollout_groups():
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.opponent_composition = False
    cfg.telemetry.metric_groups.action_decision = False

    record = {
        "update": 1,
        "total_env_steps": 64,
        "completed_episodes": 2,
        "samples": 32,
        "episode_reward_mean": 0.4,
        "overall_win_rate": 0.5,
        "env_steps_per_sec": 100.0,
        "opponent_slots_total": 4.0,
        "stop_rate": 0.2,
    }

    filtered = filter_update_record(record, cfg)

    assert "opponent_slots_total" not in filtered
    assert "stop_rate" not in filtered
    assert filtered["overall_win_rate"] == 0.5
