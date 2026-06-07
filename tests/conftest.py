"""Pytest hooks for tier markers and WSL2-safe serial execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin JAX to CPU before any test module imports src.jax (loop.py sets cuda,cpu on NVIDIA).
if os.environ.get("ORBIT_WARS_PYTEST_USE_GPU") != "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA"] = "1"

if os.environ.get("ORBIT_WARS_PYTEST_JAX_CACHE", "1") == "1":
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        cache_dir = str(Path.home() / ".cache" / "orbit-wars-jax-compile")
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from jax_warmup import warmup_rollout_compile

# Modules that always import/execute JAX (excluded from make test-fast).
FULL_JAX_FILES = frozenset(
    {
        "test_jax_env.py",
        "test_jax_env_dispatch.py",
        "test_jax_env_parity.py",
        "test_jax_trace_hygiene.py",
        "test_jax_ppo.py",
        "test_jax_rollout.py",
        "test_jax_scripted_opponents.py",
        "test_opponent_jax_sampling.py",
        "test_opponents_pool.py",
        "test_decoder_carry.py",
        "test_factorized_launch_metrics.py",
    }
)

DOMAIN_BY_FILE: dict[str, str] = {
    "test_config_consolidation.py": "config",
    "test_telemetry.py": "config",
    "test_metric_registry.py": "config",
    "test_run_paths.py": "config",
    "test_feature_registry.py": "features",
    "test_feature_encoding_golden.py": "features",
    "test_jax_env.py": "jax_env",
    "test_jax_env_dispatch.py": "jax_env",
    "test_jax_env_parity.py": "jax_env",
    "test_jax_trace_hygiene.py": "jax_env",
    "test_jax_policy_encoder.py": "policy",
    "test_jax_policy_factorized_decoder.py": "policy",
    "test_trajectory_shield_factorized.py": "policy",
    "test_factored_action_builders.py": "policy",
    "test_factored_step_vmap.py": "policy",
    "test_jax_ppo.py": "policy",
    "test_ppo_update.py": "policy",
    "test_jax_rollout.py": "policy",
    "test_jax_scripted_opponents.py": "policy",
    "test_opponent_jax_sampling.py": "policy",
    "test_opponents_pool.py": "policy",
    "test_jax_curriculum.py": "curriculum",
    "test_trajectory_shield.py": "policy",
    "test_decoder_carry.py": "policy",
    "test_factorized_launch_metrics.py": "policy",
    "test_artifact_pipeline.py": "artifacts",
    "test_replay.py": "artifacts",
    "test_kaggle_submission_packager.py": "artifacts",
    "test_curriculum.py": "curriculum",
    "test_jax_train_timing.py": "curriculum",
}

# Mixed modules: mark jax per test, not per file.
JAX_TEST_NAMES = frozenset(
    {
        "test_recent_biased_snapshot_selection_prefers_newer_updates",
        "test_python_and_jax_launch_reasons_match_for_sun_and_hit_modes",
    }
)

# Kaggle/reference env parity stays in the jax (not slow) tier — see test-kaggle-parity.
SLOW_FILES = frozenset(
    {
        # JAX compile / rollout / training-loop smokes — pre-merge only.
        "test_jax_rollout.py",
        "test_jax_curriculum.py",
        "test_jax_scripted_opponents.py",
    }
)

SWEEP_TESTS = frozenset(
    {
        "test_wandb_sweep_campaign_samples_compose_full",
    }
)

BOUNDED_SWEEP_TESTS = frozenset(
    {
        "test_wandb_sweep_campaign_samples_compose_bounded",
    }
)

# One XLA warmup collect per module before rollout/PPO integration tests run.
JAX_WARMUP_MODULES = frozenset(
    {
        "test_jax_ppo.py",
        "test_jax_rollout.py",
        "test_jax_curriculum.py",
        "test_jax_scripted_opponents.py",
        "test_jax_seed_scheduler.py",
        "test_curriculum.py",
        "test_ppo_update.py",
        "test_factorized_launch_metrics.py",
    }
)

SLOW_TESTS = frozenset(
    {
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
        "test_encode_v2_jit_vmap_smoke",
        "test_end_to_end_jax_rollout_and_update_smoke",
        "test_jax_rollout_groups_collect_two_and_four_player_formats_under_jit",
        "test_collect_rollout_jax_supports_four_player_multi_player_step",
        "test_collect_rollout_jax_two_player_static_shapes",
        "test_collect_rollout_jax_rotates_learner_after_reset_done",
        "test_collect_rollout_jax_emits_training_scalar_metric_contract",
        "test_collect_rollout_jax_logs_trajectory_shield_metrics_and_keeps_k_step_masks",
        "test_collect_rollout_jax_rotation_covers_all_player_ids_across_envs",
        "test_ppo_update_jax_accepts_four_player_rollout_transitions",
        "test_rollout_microbatching_preserves_full_environment_axis",
        "test_rollout_initializes_env_state_decoder_hidden_for_scan_structure",
        "test_factorized_rollout_emits_launches_with_shield_disabled",
    }
)

FAST_JAX_PPO_TESTS = frozenset(
    {
        "test_discounted_returns_resets_at_terminal_steps",
        "test_masked_mean_respects_zero_mask_entries",
        "test_default_training_config_uses_canonical_gae_lambda",
        "test_gae_lambda_one_matches_monte_carlo_path",
        "test_gae_lambda_below_one_differs_from_monte_carlo",
        "test_invalid_gae_lambda_rejected_at_compose",
        "test_training_ppo_hyperparameters_compose_from_hydra",
        "test_ppo_vf_and_ent_coefs_scale_reported_total_loss",
        "test_ppo_update_factorized_path_matches_on_policy_kl",
        "test_ppo_update_changes_params_after_optimizer_step",
        "test_gradient_checkpointing_encoder_init_apply_smoke",
        "test_rollout_metric_aggregation_recomputes_rate_metrics",
        "test_rollout_microbatching_requires_even_environment_division",
        "test_jax_action_builder_allows_fewer_fleet_slots_than_planets",
        "test_jax_action_builder_emits_multiple_launch_slots_per_source",
        "test_jax_action_builder_invalid_step_does_not_consume_later_ships",
    }
)

SLOW_TRAJECTORY_SHIELD_TESTS = frozenset(
    {
        "test_v2_batch_shield_allows_static_launches_on_mixed_rotating_maps",
        "test_v2_batch_shield_keeps_target_when_some_ship_buckets_are_safe",
        "test_v2_batch_shield_recomputes_bucket_legality_from_remaining_ships",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses not in (None, "0", 0):
        if os.environ.get("ORBIT_WARS_PYTEST_XDIST") != "1":
            raise pytest.UsageError(
                "pytest-xdist is opt-in only: use `make test-fast-parallel` or "
                "`make test-jax-parallel` (CPU workers). Bare pytest -n is blocked "
                "after parallel CUDA workers crashed WSL2."
            )
        if os.environ.get("ORBIT_WARS_PYTEST_USE_GPU") == "1":
            raise pytest.UsageError(
                "pytest-xdist requires CPU pytest (do not set ORBIT_WARS_PYTEST_USE_GPU=1)."
            )


def pytest_sessionstart(session: pytest.Session) -> None:
    markexpr = session.config.option.markexpr or ""
    if markexpr or os.environ.get("ORBIT_WARS_ALLOW_SLOW_TESTS") == "1":
        return
    if session.testscollected <= 0:
        return
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line("")
        terminal.write_line(
            f"WARNING: unfiltered pytest ({session.testscollected} tests). "
            "Daily loop: make test / make test-daily. Pre-merge: make test-premerge "
            "(+ make test-sweep before release). GPU pytest: ORBIT_WARS_PYTEST_USE_GPU=1. "
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


def _is_sweep(item: pytest.Item) -> bool:
    return item.name.split("[", 1)[0] in SWEEP_TESTS


def _is_slow(item: pytest.Item) -> bool:
    filename = item.path.name
    test_name = item.name.split("[", 1)[0]
    if filename in SLOW_FILES:
        return True
    if test_name in SLOW_TESTS:
        return True
    if (
        filename in {"test_jax_ppo.py", "test_ppo_update.py"}
        and test_name not in FAST_JAX_PPO_TESTS
    ):
        return True
    if (
        filename == "test_trajectory_shield.py"
        and test_name in SLOW_TRAJECTORY_SHIELD_TESTS
    ):
        return True
    return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        filename = item.path.name
        domain = DOMAIN_BY_FILE.get(filename)
        if domain is not None:
            item.add_marker(getattr(pytest.mark, domain))
        if filename in {"test_jax_env_parity.py", "test_map_pool_reset.py"}:
            item.add_marker(pytest.mark.kaggle_parity)
        if _is_jax(item):
            item.add_marker(pytest.mark.jax)
        if _is_slow(item):
            item.add_marker(pytest.mark.slow)
        test_name = item.name.split("[", 1)[0]
        if test_name in SWEEP_TESTS:
            item.add_marker(pytest.mark.sweep)
        if test_name in BOUNDED_SWEEP_TESTS:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="module", autouse=True)
def _jax_rollout_module_warmup(request: pytest.FixtureRequest) -> None:
    module_path = Path(getattr(request.module, "__file__", ""))
    if module_path.name not in JAX_WARMUP_MODULES:
        return
    warmup_rollout_compile()
