"""Assemble non-planet-flow metrics in legacy registry order.

Order must match the pre-shard ``metric_registry`` tuple (including the
planet-flow splice and losses/debug interleaving). New metrics need an entry in
``_REGISTRY_ASSEMBLY_ORDER``.
"""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition
from src.telemetry.metrics.action_decision import _ACTION_DECISION_BY_NAME
from src.telemetry.metrics.core_progress import _CORE_PROGRESS_BY_NAME
from src.telemetry.metrics.curriculum import _CURRICULUM_BY_NAME
from src.telemetry.metrics.debug import _DEBUG_BY_NAME
from src.telemetry.metrics.events import _EVENTS_BY_NAME
from src.telemetry.metrics.game_state import _GAME_STATE_BY_NAME
from src.telemetry.metrics.historical_pool import _HISTORICAL_POOL_BY_NAME
from src.telemetry.metrics.losses import _LOSSES_BY_NAME
from src.telemetry.metrics.opponent_composition import _OPPONENT_COMPOSITION_BY_NAME
from src.telemetry.metrics.timing import _TIMING_BY_NAME
from src.telemetry.metrics.trajectory_shield_debug import (
    _TRAJECTORY_SHIELD_DEBUG_BY_NAME,
)

_PLANET_FLOW_MARKER = "__planet_flow__"

_REGISTRY_ASSEMBLY_ORDER: tuple[str, ...] = (
    "update",
    "total_env_steps",
    "completed_episodes",
    "samples",
    "win_rate_2p",
    "first_place_rate_4p",
    "average_placement_4p",
    "overall_win_rate",
    "average_reward",
    "episode_reward_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "entropy_stop",
    "entropy_move",
    "approx_kl",
    "approx_kl_v2",
    "approx_kl_first_minibatch",
    "approx_kl_last_minibatch",
    "approx_kl_v2_first_minibatch",
    "approx_kl_v2_last_minibatch",
    "log_ratio_abs_mean",
    "log_ratio_abs_max_last_minibatch",
    "importance_ratio_mean",
    "clip_fraction",
    "parity_logprob_delta_abs_mean",
    "parity_logprob_delta_abs_max",
    "parity_old_log_prob_finite",
    "parity_new_log_prob_finite",
    "total_loss",
    "policy_loss_2p",
    "value_loss_2p",
    "entropy_2p",
    "approx_kl_2p",
    "approx_kl_v2_2p",
    "total_loss_2p",
    "loss_sample_count_2p",
    "policy_loss_4p",
    "value_loss_4p",
    "entropy_4p",
    "approx_kl_4p",
    "approx_kl_v2_4p",
    "total_loss_4p",
    "loss_sample_count_4p",
    "minibatches",
    "update_seconds",
    "elapsed_seconds",
    "rollout_seconds",
    "ppo_seconds",
    "env_steps_per_sec",
    "rollout_env_steps_per_sec",
    "samples_per_sec",
    "ppo_samples_per_sec",
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
    "gpu_memory_used_gb",
    "gpu_memory_total_gb",
    "gpu_memory_peak_gb",
    "gpu_name",
    "seed_scheduler_policy",
    "seed_scheduler_plateau_metric",
    "curriculum_stage_id",
    "curriculum_stage_index",
    "curriculum_stage_update",
    "curriculum_stage_dwell_updates",
    "opponent_slots_total",
    "opponent_slots_latest",
    "opponent_slots_historical",
    "opponent_slots_random",
    "opponent_slots_noop",
    "opponent_slots_nearest_sniper",
    "opponent_slots_turtle",
    "opponent_slots_opportunistic",
    "opponent_historical_fallback_latest_slots",
    "stop_rate",
    "mean_active_launches_per_turn",
    _PLANET_FLOW_MARKER,
    "win_rate_delta_10",
    "approx_kl_window_mean",
    "entropy_window_mean",
    "planet_flow_sweep_score",
    "preflight_sweep_score",
    "ssot_preflight_sweep_score",
    "stop_utilization_ratio",
    "mean_ships_per_launch",
    "survival_time",
    "score_share",
    "debug_step_mask_sum",
    "debug_old_log_prob_finite",
    "debug_returns_finite",
    "debug_advantages_finite",
    "debug_ship_bucket_mask_any_min",
    "debug_ship_bucket_mask_all_false",
    "debug_source_mask_all_false",
    "debug_active_launch_all_false_bucket",
    "trajectory_shield_blocked_count",
    "trajectory_shield_blocked_sun_count",
    "trajectory_shield_blocked_bounds_count",
    "trajectory_shield_blocked_unintended_hit_count",
    "trajectory_shield_blocked_horizon_count",
    "trajectory_shield_fallback_noop_count",
    "trajectory_shield_legal_non_noop_rate",
    "historical_pool_size",
    "historical_pool_capacity",
    "historical_snapshot_ids",
    "historical_snapshot_ages_updates",
    "reseed_events",
    "curriculum_phase_events",
    "event",
    "checkpoint_status",
    "checkpoint_final",
    "checkpoint_reason",
    "checkpoint_error",
    "snapshot_id",
    "snapshot_slot",
    "historical_snapshot_evicted",
    "from_stage",
    "to_stage",
    "stage",
    "reason",
    "metric",
    "metric_value",
    "threshold",
    "bracket_training_phase",
    "weak_config",
)


def _planet_flow_metrics() -> tuple[MetricDefinition, ...]:
    from src.telemetry.planet_flow_registry import planet_flow_metric_definitions

    return planet_flow_metric_definitions()


def _definitions_by_name() -> dict[str, MetricDefinition]:
    return {
        **_CORE_PROGRESS_BY_NAME,
        **_LOSSES_BY_NAME,
        **_DEBUG_BY_NAME,
        **_TIMING_BY_NAME,
        **_EVENTS_BY_NAME,
        **_CURRICULUM_BY_NAME,
        **_OPPONENT_COMPOSITION_BY_NAME,
        **_ACTION_DECISION_BY_NAME,
        **_GAME_STATE_BY_NAME,
        **_TRAJECTORY_SHIELD_DEBUG_BY_NAME,
        **_HISTORICAL_POOL_BY_NAME,
    }


def assemble_non_planet_flow_metrics() -> tuple[MetricDefinition, ...]:
    by_name = _definitions_by_name()
    assembled: list[MetricDefinition] = []
    for item in _REGISTRY_ASSEMBLY_ORDER:
        if item == _PLANET_FLOW_MARKER:
            assembled.extend(_planet_flow_metrics())
            continue
        assembled.append(by_name[item])
    return tuple(assembled)
