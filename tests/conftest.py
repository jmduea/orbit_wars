"""Pytest hooks for tier markers and WSL2-safe serial execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAIN_BY_FILE: dict[str, str] = {
    "test_config_consolidation.py": "config",
    "test_telemetry.py": "config",
    "test_metric_registry.py": "config",
    "test_run_paths.py": "config",
    "test_features.py": "features",
    "test_feature_history.py": "features",
    "test_feature_registry.py": "features",
    "test_normalization.py": "features",
    "test_jax_env.py": "jax_env",
    "test_jax_env_v2_dispatch.py": "jax_env",
    "test_jax_env_parity.py": "jax_env",
    "test_jax_policy.py": "policy",
    "test_jax_policy_v2.py": "policy",
    "test_jax_ppo.py": "policy",
    "test_jax_rollout_v2.py": "policy",
    "test_jax_scripted_opponents_v2.py": "policy",
    "test_trajectory_shield.py": "policy",
    "test_artifact_pipeline.py": "artifacts",
    "test_replay.py": "artifacts",
    "test_kaggle_submission_packager.py": "artifacts",
    "test_curriculum.py": "curriculum",
    "test_jax_train_timing.py": "curriculum",
}

FULL_JAX_FILES = frozenset(
    {
        "test_jax_env.py",
        "test_jax_env_v2_dispatch.py",
        "test_jax_env_parity.py",
        "test_jax_policy.py",
        "test_jax_policy_v2.py",
        "test_jax_ppo.py",
        "test_jax_rollout_v2.py",
        "test_jax_scripted_opponents_v2.py",
    }
)

# Mixed modules: mark jax per test, not per file.
JAX_TEST_NAMES = frozenset(
    {
        "test_recent_biased_snapshot_selection_prefers_newer_updates",
        "test_jax_replay_actor_handles_sequence_policy_outputs",
        "test_python_and_jax_launch_reasons_match_for_sun_and_hit_modes",
    }
)

SLOW_FILES = frozenset({"test_jax_env_parity.py"})

SLOW_TESTS = frozenset(
    {
        "test_wandb_sweep_campaign_samples_compose_full",
        "test_checkpoint_payload_roundtrips_curriculum_and_historical_pool",
        "test_checkpoint_payload_builder_freezes_curriculum_state_for_async_jobs",
        "test_two_player_rollout_reports_sampled_random_family_slots",
        "test_four_player_rollout_reports_single_random_family_slots",
        "test_two_player_rollout_reports_single_latest_family_slots",
        "test_historical_family_falls_back_to_latest_when_pool_empty",
        "test_training_loop_logs_curriculum_events_on_same_update",
        "test_lean_rollout_metrics_skips_expensive_scan_payloads",
        "test_jax_batched_reset_and_step_shapes",
        "test_jax_owner_relative_feature_shapes_and_values_for_four_players",
        "test_jax_history_shapes_use_configured_window",
        "test_launch_and_production_match_core_orbit_wars_mechanics",
        "test_jax_candidates_are_sorted_by_distance_before_id_tiebreaker",
        "test_jax_candidate_history_aligns_by_source_and_target_id_after_reorder",
        "test_jax_candidate_history_zeros_missing_prior_targets",
    }
)

FAST_JAX_PPO_TESTS = frozenset(
    {
        "test_rollout_metric_aggregation_recomputes_rate_metrics",
        "test_rollout_microbatching_requires_even_environment_division",
        "test_jax_action_builder_allows_fewer_fleet_slots_than_planets",
        "test_jax_action_builder_emits_multiple_launch_slots_per_source",
        "test_jax_action_builder_invalid_step_does_not_consume_later_ships",
    }
)

SLOW_TRAJECTORY_SHIELD_TESTS = frozenset(
    {
        "test_jax_batch_shield_reports_blocked_metrics_for_sun_crossing",
        "test_jax_batch_shield_allows_static_launches_on_mixed_rotating_maps",
        "test_jax_batch_shield_keeps_target_when_some_ship_buckets_are_safe",
        "test_jax_batch_shield_recomputes_bucket_legality_from_remaining_ships",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses not in (None, "0", 0):
        raise pytest.UsageError(
            "pytest-xdist is disabled for Orbit Wars: parallel workers that import "
            "JAX/CUDA can exhaust GPU memory and crash WSL2. Use serial targets "
            "such as `make test-fast`, `make test-jax`, or `make test`."
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    markexpr = session.config.option.markexpr or ""
    if markexpr or os.environ.get("ORBIT_WARS_ALLOW_SLOW_TESTS") == "1":
        return
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line("")
        terminal.write_line(
            "WARNING: full suite selected (156 tests incl. 49 slow/JAX-heavy). "
            "Daily loop: make test-fast (CPU) or make test-jax (serial JAX). "
            "Set ORBIT_WARS_ALLOW_SLOW_TESTS=1 to silence.",
            yellow=True,
        )
        terminal.write_line("")


def _is_jax(item: pytest.Item) -> bool:
    filename = item.path.name
    test_name = item.name.split("[", 1)[0]
    if filename in FULL_JAX_FILES:
        return True
    return test_name in JAX_TEST_NAMES


def _is_slow(item: pytest.Item) -> bool:
    filename = item.path.name
    test_name = item.name.split("[", 1)[0]
    if filename in SLOW_FILES:
        return True
    if test_name in SLOW_TESTS:
        return True
    if filename == "test_jax_ppo.py" and test_name not in FAST_JAX_PPO_TESTS:
        return True
    if filename == "test_trajectory_shield.py" and test_name in SLOW_TRAJECTORY_SHIELD_TESTS:
        return True
    return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        filename = item.path.name
        domain = DOMAIN_BY_FILE.get(filename)
        if domain is not None:
            item.add_marker(getattr(pytest.mark, domain))
        if _is_jax(item):
            item.add_marker(pytest.mark.jax)
        if _is_slow(item):
            item.add_marker(pytest.mark.slow)
