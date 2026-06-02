from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from src.jax.rollout.metric_contract import (
    FINALIZED_ROLLOUT_RATE_KEYS,
    LOGGED_ROLLOUT_SCALAR_KEYS,
    ROLLOUT_CHUNK_ONLY_SCALAR_KEYS,
    ROLLOUT_INTERNAL_SCALAR_KEYS,
)

METRIC_GROUPS: tuple[str, ...] = (
    "core_progress",
    "losses",
    "timing",
    "curriculum",
    "opponent_composition",
    "action_decision",
    "game_state",
    "trajectory_shield_debug",
    "historical_pool",
    "events",
    "debug",
)

DEFAULT_ENABLED_GROUPS: frozenset[str] = frozenset(
    {
        "core_progress",
        "losses",
        "timing",
        "curriculum",
        "events",
    }
)

KNOWN_SWEEP_OBJECTIVE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "overall_win_rate",
        "env_steps_per_sec",
    }
)

KNOWN_PROTECTED_UPDATE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "update",
        "total_env_steps",
        "completed_episodes",
        "samples",
        "episode_reward_mean",
        "overall_win_rate",
        "env_steps_per_sec",
        "win_rate_2p",
        "first_place_rate_4p",
    }
)

CURRICULUM_PROMOTION_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "overall_win_rate",
        "win_rate_2p",
        "first_place_rate_4p",
        "average_reward",
        "episode_reward_mean",
        "survival_time",
        "score_share",
        "approx_kl",
    }
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    group: str
    description: str
    record_kinds: frozenset[str]
    protected: bool = False
    internal_only: bool = False


def _metric(
    name: str,
    group: str,
    description: str,
    *,
    record_kinds: tuple[str, ...] = ("update",),
    protected: bool = False,
    internal_only: bool = False,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        group=group,
        description=description,
        record_kinds=frozenset(record_kinds),
        protected=protected,
        internal_only=internal_only,
    )


_METRICS: tuple[MetricDefinition, ...] = (
    _metric(
        "update",
        "core_progress",
        "Completed PPO update index.",
        record_kinds=("update", "event"),
        protected=True,
    ),
    _metric(
        "total_env_steps",
        "core_progress",
        "Cumulative environment steps processed so far.",
        protected=True,
    ),
    _metric(
        "completed_episodes",
        "core_progress",
        "Completed episodes across all rollout groups.",
        protected=True,
    ),
    _metric(
        "samples",
        "core_progress",
        "Learner decision samples consumed by the update.",
        protected=True,
    ),
    _metric(
        "win_rate_2p",
        "core_progress",
        "First-place win rate in 2-player episodes.",
        protected=True,
    ),
    _metric(
        "first_place_rate_4p",
        "core_progress",
        "First-place rate in 4-player episodes.",
        protected=True,
    ),
    _metric(
        "average_placement_4p",
        "core_progress",
        "Average final placement in completed 4-player episodes.",
    ),
    _metric(
        "overall_win_rate",
        "core_progress",
        "Overall first-place rate across completed episodes.",
        protected=True,
    ),
    _metric(
        "average_reward",
        "core_progress",
        "Mean per-step reward over the rollout.",
    ),
    _metric(
        "episode_reward_mean",
        "core_progress",
        "Mean episodic reward across completed episodes.",
        protected=True,
    ),
    _metric(
        "policy_loss",
        "losses",
        "Mean PPO policy loss across minibatches.",
    ),
    _metric(
        "value_loss",
        "losses",
        "Mean PPO value loss across minibatches.",
    ),
    _metric("entropy", "losses", "Mean action entropy across minibatches."),
    _metric(
        "entropy_stop",
        "losses",
        "Mean stop-head entropy for factorized pointer decoders.",
    ),
    _metric(
        "entropy_move",
        "losses",
        "Mean source/target/ship entropy for factorized pointer decoders.",
    ),
    _metric(
        "approx_kl",
        "losses",
        "Unweighted mean over PPO minibatches of mean(old_log_prob - new_log_prob); "
        "can diverge with epochs>1 and many inner steps.",
    ),
    _metric(
        "approx_kl_v2",
        "losses",
        "Schulman-style approximate KL using clipped importance ratios.",
    ),
    _metric(
        "approx_kl_first_minibatch",
        "losses",
        "approx_kl on the first minibatch before any optimizer step (parity sentinel).",
        protected=True,
    ),
    _metric(
        "approx_kl_last_minibatch",
        "losses",
        "approx_kl on the final minibatch after all inner optimizer steps.",
    ),
    _metric(
        "approx_kl_v2_first_minibatch",
        "losses",
        "approx_kl_v2 on the first minibatch before any optimizer step.",
    ),
    _metric(
        "approx_kl_v2_last_minibatch",
        "losses",
        "approx_kl_v2 on the final minibatch after all inner optimizer steps.",
    ),
    _metric(
        "log_ratio_abs_mean",
        "losses",
        "Mean absolute log-probability delta between rollout and replay.",
    ),
    _metric(
        "log_ratio_abs_max_last_minibatch",
        "losses",
        "Max absolute log-probability delta on the final PPO minibatch.",
    ),
    _metric(
        "importance_ratio_mean",
        "losses",
        "Mean clipped importance ratio exp(clip(new_log_prob - old_log_prob)).",
    ),
    _metric(
        "clip_fraction",
        "losses",
        "Fraction of masked steps where the importance ratio exceeds clip_coef.",
    ),
    _metric(
        "parity_logprob_delta_abs_mean",
        "losses",
        "Pre-update mean |replay_log_prob - stored_old_log_prob| on the first minibatch.",
        protected=True,
    ),
    _metric(
        "parity_logprob_delta_abs_max",
        "losses",
        "Pre-update max |replay_log_prob - stored_old_log_prob| on the first minibatch.",
        protected=True,
    ),
    _metric(
        "parity_old_log_prob_finite",
        "losses",
        "Whether stored old log-probs are finite on the first minibatch parity slice.",
    ),
    _metric(
        "parity_new_log_prob_finite",
        "losses",
        "Whether replay log-probs are finite on the first minibatch parity slice.",
    ),
    _metric("total_loss", "losses", "Final weighted PPO loss used for optimization."),
    _metric("policy_loss_2p", "losses", "PPO policy loss for 2-player samples."),
    _metric("value_loss_2p", "losses", "PPO value loss for 2-player samples."),
    _metric("entropy_2p", "losses", "Action entropy for 2-player samples."),
    _metric("approx_kl_2p", "losses", "Approximate KL for 2-player samples."),
    _metric(
        "approx_kl_v2_2p",
        "losses",
        "Schulman-style approximate KL for 2-player samples.",
    ),
    _metric("total_loss_2p", "losses", "Weighted PPO loss for 2-player samples."),
    _metric(
        "loss_sample_count_2p",
        "losses",
        "Learner decision samples contributing to 2-player PPO loss diagnostics.",
    ),
    _metric("policy_loss_4p", "losses", "PPO policy loss for 4-player samples."),
    _metric("value_loss_4p", "losses", "PPO value loss for 4-player samples."),
    _metric("entropy_4p", "losses", "Action entropy for 4-player samples."),
    _metric("approx_kl_4p", "losses", "Approximate KL for 4-player samples."),
    _metric(
        "approx_kl_v2_4p",
        "losses",
        "Schulman-style approximate KL for 4-player samples.",
    ),
    _metric("total_loss_4p", "losses", "Weighted PPO loss for 4-player samples."),
    _metric(
        "loss_sample_count_4p",
        "losses",
        "Learner decision samples contributing to 4-player PPO loss diagnostics.",
    ),
    _metric("minibatches", "losses", "Minibatch count used in the PPO update."),
    _metric("update_seconds", "timing", "Wall-clock seconds for the full update loop."),
    _metric("elapsed_seconds", "timing", "Wall-clock seconds since training started."),
    _metric(
        "rollout_seconds", "timing", "Wall-clock seconds spent collecting rollouts."
    ),
    _metric("ppo_seconds", "timing", "Wall-clock seconds spent in PPO optimization."),
    _metric(
        "env_steps_per_sec",
        "timing",
        "Environment steps processed per second over the full update.",
        protected=True,
    ),
    _metric(
        "rollout_env_steps_per_sec",
        "timing",
        "Environment steps processed per second during rollout collection.",
    ),
    _metric("samples_per_sec", "timing", "Decision samples processed per second."),
    _metric(
        "ppo_samples_per_sec",
        "timing",
        "Decision samples processed per second during PPO optimization.",
    ),
    _metric(
        "rollout_seconds_2p",
        "debug",
        "Wall-clock seconds spent collecting 2-player rollout groups.",
    ),
    _metric(
        "rollout_seconds_4p",
        "debug",
        "Wall-clock seconds spent collecting 4-player rollout groups.",
    ),
    _metric(
        "env_steps_per_sec_2p",
        "debug",
        "2-player environment steps processed per second over the full update.",
    ),
    _metric(
        "env_steps_per_sec_4p",
        "debug",
        "4-player environment steps processed per second over the full update.",
    ),
    _metric(
        "rollout_env_steps_per_sec_2p",
        "debug",
        "2-player environment steps processed per second during 2-player rollout collection.",
    ),
    _metric(
        "rollout_env_steps_per_sec_4p",
        "debug",
        "4-player environment steps processed per second during 4-player rollout collection.",
    ),
    _metric(
        "samples_per_sec_2p",
        "debug",
        "2-player learner decision samples processed per second over the full update.",
    ),
    _metric(
        "samples_per_sec_4p",
        "debug",
        "4-player learner decision samples processed per second over the full update.",
    ),
    _metric(
        "rollout_samples_per_sec_2p",
        "debug",
        "2-player learner decision samples processed per second during 2-player rollout collection.",
    ),
    _metric(
        "rollout_samples_per_sec_4p",
        "debug",
        "4-player learner decision samples processed per second during 4-player rollout collection.",
    ),
    _metric(
        "update_time_rollout_fraction",
        "timing",
        "Fraction of update wall time spent collecting rollouts.",
    ),
    _metric(
        "update_time_ppo_fraction",
        "timing",
        "Fraction of update wall time spent in PPO optimization.",
    ),
    _metric(
        "gpu_memory_used_gb",
        "timing",
        "Device memory in use after the update (GiB, driver-reported when available).",
    ),
    _metric(
        "gpu_memory_total_gb",
        "timing",
        "Total device memory for the active GPU (GiB).",
    ),
    _metric(
        "gpu_memory_peak_gb",
        "timing",
        "Running peak device memory observed since run start (GiB).",
        protected=True,
    ),
    _metric(
        "gpu_name",
        "events",
        "Observed GPU product name for the training run.",
        record_kinds=("update", "event"),
    ),
    _metric(
        "seed_scheduler_policy",
        "curriculum",
        "Seed scheduling policy selected for the next update.",
    ),
    _metric(
        "seed_scheduler_plateau_metric",
        "curriculum",
        "Canonical plateau metric monitored by the seed scheduler.",
    ),
    _metric("curriculum_stage_id", "curriculum", "Active curriculum stage identifier."),
    _metric("curriculum_stage_index", "curriculum", "Active curriculum stage index."),
    _metric(
        "curriculum_stage_update",
        "curriculum",
        "Update index attached to the active curriculum stage snapshot.",
    ),
    _metric(
        "curriculum_stage_dwell_updates",
        "curriculum",
        "Updates spent in the current curriculum stage.",
    ),
    _metric(
        "opponent_slots_total",
        "opponent_composition",
        "Total opponent slots sampled across the rollout.",
    ),
    _metric(
        "opponent_slots_latest",
        "opponent_composition",
        "Opponent slots filled by the latest learner snapshot.",
    ),
    _metric(
        "opponent_slots_historical",
        "opponent_composition",
        "Opponent slots filled by historical learner snapshots.",
    ),
    _metric(
        "opponent_slots_random",
        "opponent_composition",
        "Opponent slots filled by random policy opponents.",
    ),
    _metric(
        "opponent_slots_noop",
        "opponent_composition",
        "Opponent slots filled by no-op opponents.",
    ),
    _metric(
        "opponent_slots_nearest_sniper",
        "opponent_composition",
        "Opponent slots filled by nearest-sniper opponents.",
    ),
    _metric(
        "opponent_slots_turtle",
        "opponent_composition",
        "Opponent slots filled by turtle opponents.",
    ),
    _metric(
        "opponent_slots_opportunistic",
        "opponent_composition",
        "Opponent slots filled by opportunistic opponents.",
    ),
    _metric(
        "opponent_historical_fallback_latest_slots",
        "opponent_composition",
        "Historical opponent slots that fell back to the latest policy.",
    ),
    _metric(
        "stop_rate",
        "action_decision",
        "Fraction of active factorized launch steps where the stop head fired.",
    ),
    _metric(
        "mean_active_launches_per_turn",
        "action_decision",
        "Mean non-stop launches with a positive ship bucket per env-turn.",
    ),
    _metric(
        "planet_flow_demanded_mass_sum",
        "action_decision",
        "Sum of active target pressure mass sampled by Planet Flow.",
    ),
    _metric(
        "planet_flow_unreachable_demand_mass_sum",
        "action_decision",
        "Planet Flow target pressure mass with no feasible candidate edge.",
    ),
    _metric(
        "planet_flow_held_demand_mass_sum",
        "action_decision",
        "Planet Flow target pressure mass not represented by emitted launches.",
    ),
    _metric(
        "planet_flow_requested_ship_mass_sum",
        "action_decision",
        "Ship mass requested by the Planet Flow compiler before capacity truncation.",
    ),
    _metric(
        "planet_flow_emitted_ship_mass_sum",
        "action_decision",
        "Ship mass emitted by the Planet Flow compiler.",
    ),
    _metric(
        "planet_flow_capacity_dropped_launch_count",
        "action_decision",
        "Planet Flow launch intents dropped by per-turn action capacity.",
    ),
    _metric(
        "planet_flow_emitted_launch_count",
        "action_decision",
        "Valid launches emitted by the Planet Flow compiler.",
    ),
    _metric(
        "planet_flow_small_launch_count",
        "action_decision",
        "Planet Flow emitted launches with at most one ship.",
    ),
    _metric(
        "planet_flow_duplicate_source_target_count",
        "action_decision",
        "Mergeable same-source/same-target Planet Flow duplicate launches.",
    ),
    _metric(
        "planet_flow_unreachable_demand_rate",
        "action_decision",
        "Unreachable Planet Flow demand mass divided by demanded mass.",
    ),
    _metric(
        "planet_flow_held_demand_rate",
        "action_decision",
        "Held Planet Flow demand mass divided by demanded mass.",
    ),
    _metric(
        "planet_flow_emitted_ship_mass_rate",
        "action_decision",
        "Emitted Planet Flow ship mass divided by requested ship mass.",
    ),
    _metric(
        "planet_flow_capacity_drop_rate",
        "action_decision",
        "Planet Flow capacity-dropped launch intents divided by attempted launches.",
    ),
    _metric(
        "planet_flow_small_launch_rate",
        "action_decision",
        "Planet Flow one-ship emitted launches divided by emitted launches.",
    ),
    _metric(
        "planet_flow_duplicate_source_target_rate",
        "action_decision",
        "Duplicate same-source/same-target Planet Flow launches divided by emitted launches.",
    ),
    _metric(
        "planet_flow_control_demanded_mass_sum",
        "action_decision",
        "Seeded-random control target pressure mass run through the Planet Flow compiler.",
    ),
    _metric(
        "planet_flow_control_unreachable_demand_mass_sum",
        "action_decision",
        "Seeded-random control demand mass with no feasible candidate edge.",
    ),
    _metric(
        "planet_flow_control_held_demand_mass_sum",
        "action_decision",
        "Seeded-random control demand mass not represented by emitted launches.",
    ),
    _metric(
        "planet_flow_control_requested_ship_mass_sum",
        "action_decision",
        "Ship mass requested by the seeded-random Planet Flow compiler control.",
    ),
    _metric(
        "planet_flow_control_emitted_ship_mass_sum",
        "action_decision",
        "Ship mass emitted by the seeded-random Planet Flow compiler control.",
    ),
    _metric(
        "planet_flow_control_capacity_dropped_launch_count",
        "action_decision",
        "Seeded-random control launch intents dropped by per-turn action capacity.",
    ),
    _metric(
        "planet_flow_control_emitted_launch_count",
        "action_decision",
        "Valid launches emitted by the seeded-random Planet Flow compiler control.",
    ),
    _metric(
        "planet_flow_control_small_launch_count",
        "action_decision",
        "Seeded-random control emitted launches with at most one ship.",
    ),
    _metric(
        "planet_flow_control_duplicate_source_target_count",
        "action_decision",
        "Seeded-random control duplicate same-source/same-target launch count.",
    ),
    _metric(
        "planet_flow_control_unreachable_demand_rate",
        "action_decision",
        "Seeded-random control unreachable demand mass divided by demanded mass.",
    ),
    _metric(
        "planet_flow_control_held_demand_rate",
        "action_decision",
        "Seeded-random control held demand mass divided by demanded mass.",
    ),
    _metric(
        "planet_flow_control_emitted_ship_mass_rate",
        "action_decision",
        "Seeded-random control emitted ship mass divided by requested ship mass.",
    ),
    _metric(
        "planet_flow_control_capacity_drop_rate",
        "action_decision",
        "Seeded-random control capacity-dropped intents divided by attempted launches.",
    ),
    _metric(
        "planet_flow_control_small_launch_rate",
        "action_decision",
        "Seeded-random control one-ship launches divided by emitted launches.",
    ),
    _metric(
        "planet_flow_control_duplicate_source_target_rate",
        "action_decision",
        "Seeded-random control duplicate launches divided by emitted launches.",
    ),
    _metric(
        "planet_flow_emitted_launch_count_delta_vs_control",
        "action_decision",
        "Learned Planet Flow emitted launch count minus seeded-random control count.",
    ),
    _metric(
        "planet_flow_emitted_ship_mass_delta_vs_control",
        "action_decision",
        "Learned Planet Flow emitted ship mass minus seeded-random control mass.",
    ),
    _metric(
        "planet_flow_unreachable_demand_rate_delta_vs_control",
        "action_decision",
        "Learned unreachable demand rate minus seeded-random control rate.",
    ),
    _metric(
        "planet_flow_held_demand_rate_delta_vs_control",
        "action_decision",
        "Learned held demand rate minus seeded-random control rate.",
    ),
    _metric(
        "planet_flow_emitted_ship_mass_rate_delta_vs_control",
        "action_decision",
        "Learned emitted ship mass rate minus seeded-random control rate.",
    ),
    _metric(
        "planet_flow_small_launch_rate_delta_vs_control",
        "action_decision",
        "Learned one-ship launch rate minus seeded-random control rate.",
    ),
    _metric(
        "planet_flow_duplicate_source_target_rate_delta_vs_control",
        "action_decision",
        "Learned duplicate launch rate minus seeded-random control rate.",
    ),
    _metric(
        "win_rate_delta_10",
        "action_decision",
        "Last-window minus first-window overall_win_rate over 10 updates.",
    ),
    _metric(
        "approx_kl_window_mean",
        "action_decision",
        "Mean approx_kl over the last 10 training updates (preflight-aligned).",
    ),
    _metric(
        "entropy_window_mean",
        "action_decision",
        "Mean policy entropy over the last 10 training updates (preflight-aligned).",
    ),
    _metric(
        "planet_flow_sweep_score",
        "action_decision",
        "W&B sweep objective: win_rate_delta_10 when window-mean KL/entropy floors pass, else -1.",
    ),
    _metric(
        "stop_utilization_ratio",
        "action_decision",
        "mean_active_launches_per_turn divided by model.max_moves_k (L1 gate).",
    ),
    _metric(
        "survival_time",
        "game_state",
        "Mean survival time for completed episodes.",
    ),
    _metric(
        "score_share",
        "game_state",
        "Mean score share for completed episodes.",
    ),
    _metric(
        "debug_step_mask_sum",
        "debug",
        "Sum of active PPO step masks in the update minibatch.",
    ),
    _metric(
        "debug_old_log_prob_finite",
        "debug",
        "Whether all stored old log-probabilities are finite.",
    ),
    _metric(
        "debug_returns_finite",
        "debug",
        "Whether all computed returns are finite.",
    ),
    _metric(
        "debug_advantages_finite",
        "debug",
        "Whether all computed advantages are finite.",
    ),
    _metric(
        "debug_ship_bucket_mask_any_min",
        "debug",
        "Minimum per-step ship-bucket mask count in the update batch.",
    ),
    _metric(
        "debug_ship_bucket_mask_all_false",
        "debug",
        "Count of rows whose ship-bucket mask is entirely false.",
    ),
    _metric(
        "debug_source_mask_all_false",
        "debug",
        "Count of rows whose source mask is entirely false.",
    ),
    _metric(
        "debug_active_launch_all_false_bucket",
        "debug",
        "Count of active launch rows with an all-false ship bucket.",
    ),
    _metric(
        "trajectory_shield_blocked_count",
        "trajectory_shield_debug",
        "Count of actions blocked by the trajectory shield.",
    ),
    _metric(
        "trajectory_shield_blocked_sun_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to sun collisions.",
    ),
    _metric(
        "trajectory_shield_blocked_bounds_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to map bounds.",
    ),
    _metric(
        "trajectory_shield_blocked_unintended_hit_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to unintended hits.",
    ),
    _metric(
        "trajectory_shield_blocked_horizon_count",
        "trajectory_shield_debug",
        "Count of actions blocked due to shield horizon limits.",
    ),
    _metric(
        "trajectory_shield_fallback_noop_count",
        "trajectory_shield_debug",
        "Count of shielded decisions that fell back to noop.",
    ),
    _metric(
        "trajectory_shield_legal_non_noop_rate",
        "trajectory_shield_debug",
        "Fraction of originally non-noop decisions that remained legal after shielding.",
    ),
    _metric(
        "historical_pool_size",
        "historical_pool",
        "Valid historical snapshot count in the pool.",
    ),
    _metric(
        "historical_pool_capacity",
        "historical_pool",
        "Configured historical snapshot pool capacity.",
    ),
    _metric(
        "historical_snapshot_ids",
        "historical_pool",
        "Snapshot identifiers currently stored in the historical pool.",
        record_kinds=("event",),
    ),
    _metric(
        "historical_snapshot_ages_updates",
        "historical_pool",
        "Snapshot ages in updates for the historical pool.",
        record_kinds=("event",),
    ),
    _metric(
        "reseed_events",
        "events",
        "Embedded seed reseed events emitted during the update.",
    ),
    _metric(
        "curriculum_phase_events",
        "events",
        "Embedded curriculum and historical snapshot events emitted during the update.",
    ),
    _metric(
        "event",
        "events",
        "Sparse event record type.",
        record_kinds=("event",),
        protected=True,
    ),
    _metric(
        "checkpoint_status",
        "events",
        "Checkpoint pipeline status for a checkpoint_result event.",
        record_kinds=("event",),
        protected=True,
    ),
    _metric(
        "checkpoint_final",
        "events",
        "Whether the checkpoint_result event corresponds to the final checkpoint.",
        record_kinds=("event",),
        protected=True,
    ),
    _metric(
        "checkpoint_reason",
        "events",
        "Checkpoint pipeline reason string.",
        record_kinds=("event",),
        protected=True,
    ),
    _metric(
        "checkpoint_error",
        "events",
        "Checkpoint pipeline error text, if any.",
        record_kinds=("event",),
        protected=True,
    ),
    _metric(
        "snapshot_id",
        "events",
        "Historical snapshot identifier for a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "snapshot_slot",
        "events",
        "Historical snapshot slot for a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "historical_snapshot_evicted",
        "events",
        "Whether a historical snapshot event replaced an existing slot.",
        record_kinds=("event",),
    ),
    _metric(
        "from_stage",
        "events",
        "Previous curriculum stage for a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "to_stage",
        "events",
        "Next curriculum stage for a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "stage",
        "events",
        "Current curriculum stage for a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "reason",
        "events",
        "Human-readable reason attached to a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "metric",
        "events",
        "Metric name attached to a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "metric_value",
        "events",
        "Metric value attached to a sparse event record.",
        record_kinds=("event",),
    ),
    _metric(
        "threshold",
        "events",
        "Threshold value attached to a sparse event record.",
        record_kinds=("event",),
    ),
)

_curriculum_prob_metrics = tuple(
    _metric(
        f"curriculum_family_prob_{family}",
        "curriculum",
        f"Sampling probability for the {family} opponent family in the active curriculum stage.",
    )
    for family in (
        "latest",
        "historical",
        "random",
        "noop",
        "nearest_sniper",
        "turtle",
        "opportunistic",
    )
)

METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = _METRICS + _curriculum_prob_metrics
METRIC_DEFINITIONS_BY_NAME: dict[str, MetricDefinition] = {
    definition.name: definition for definition in METRIC_DEFINITIONS
}

ROLLOUT_SCALAR_ORDER: tuple[str, ...] = LOGGED_ROLLOUT_SCALAR_KEYS

ROLLOUT_OUTPUT_METRIC_NAMES: frozenset[str] = frozenset(
    name for name in ROLLOUT_SCALAR_ORDER if name in METRIC_DEFINITIONS_BY_NAME
)

ROLLOUT_INTERNAL_REQUIRED_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "samples",
        "env_steps",
        "episode_done",
        "average_reward",
        "episode_reward_mean",
        "win_rate_2p",
        "first_place_rate_4p",
        "overall_win_rate",
        "survival_time",
        "score_share",
    }
)

PPO_METRIC_ORDER: tuple[str, ...] = (
    "policy_loss",
    "value_loss",
    "entropy",
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
    "debug_step_mask_sum",
    "debug_old_log_prob_finite",
    "debug_returns_finite",
    "debug_advantages_finite",
    "debug_ship_bucket_mask_any_min",
    "debug_ship_bucket_mask_all_false",
    "debug_source_mask_all_false",
    "debug_active_launch_all_false_bucket",
)

PPO_INTERNAL_REQUIRED_METRIC_NAMES: frozenset[str] = frozenset(
    {"approx_kl", "approx_kl_first_minibatch", "parity_logprob_delta_abs_mean"}
)

NON_SCALAR_UPDATE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "seed_scheduler_policy",
        "seed_scheduler_plateau_metric",
        "curriculum_stage_id",
        "reseed_events",
        "curriculum_phase_events",
    }
)


def metric_definition(name: str) -> MetricDefinition:
    try:
        return METRIC_DEFINITIONS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unregistered telemetry metric: {name}") from exc


def default_metric_groups_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        **{
            group_name: group_name in DEFAULT_ENABLED_GROUPS
            for group_name in METRIC_GROUPS
        }
    )


def enabled_metric_groups(metric_groups_cfg: Any | None) -> set[str]:
    if metric_groups_cfg is None:
        metric_groups_cfg = default_metric_groups_namespace()
    return {
        group_name
        for group_name in METRIC_GROUPS
        if bool(
            getattr(metric_groups_cfg, group_name, group_name in DEFAULT_ENABLED_GROUPS)
        )
    }


def metric_groups_cfg_from_config(cfg: Any | None) -> Any | None:
    if cfg is None:
        return None
    telemetry_cfg = getattr(cfg, "telemetry", None)
    if telemetry_cfg is None:
        return None
    return getattr(telemetry_cfg, "metric_groups", None)


def protected_metric_names(cfg: Any | None = None) -> set[str]:
    protected = set(KNOWN_PROTECTED_UPDATE_METRIC_NAMES)
    protected.update(KNOWN_SWEEP_OBJECTIVE_METRIC_NAMES)
    if cfg is None:
        return protected
    artifacts = getattr(cfg, "artifacts", None)
    checkpoint_retention = (
        getattr(artifacts, "checkpoint_retention", None)
        if artifacts is not None
        else None
    )
    if checkpoint_retention is not None:
        best_metric_name = str(
            getattr(checkpoint_retention, "best_metric_name", "") or ""
        ).strip()
        if best_metric_name:
            protected.add(best_metric_name)
    training = getattr(cfg, "training", None)
    plateau_metric = (
        str(getattr(training, "plateau_metric", "") or "").strip()
        if training is not None
        else ""
    )
    if plateau_metric:
        protected.add(plateau_metric)
    return protected


def validate_metric_name(name: str, *, record_kind: str = "update") -> None:
    definition = metric_definition(name)
    if record_kind not in definition.record_kinds:
        raise ValueError(
            f"Telemetry metric {name!r} is not valid for record kind {record_kind!r}."
        )


def validate_scalar_update_metric_name(name: str) -> None:
    validate_metric_name(name, record_kind="update")
    if name in NON_SCALAR_UPDATE_METRIC_NAMES:
        raise ValueError(f"Telemetry metric {name!r} is not a scalar update metric.")


def enabled_metric_names(
    metric_groups_cfg: Any | None,
    *,
    record_kind: str,
    extra_protected_names: set[str] | frozenset[str] | None = None,
) -> set[str]:
    enabled_groups = enabled_metric_groups(metric_groups_cfg)
    protected_names = set(extra_protected_names or set())
    names: set[str] = set()
    for definition in METRIC_DEFINITIONS:
        if record_kind not in definition.record_kinds:
            continue
        if (
            definition.group in enabled_groups
            or definition.protected
            or definition.name in protected_names
        ):
            names.add(definition.name)
    return names


def filter_metric_record(
    record: dict[str, Any],
    *,
    metric_groups_cfg: Any | None,
    record_kind: str,
    extra_protected_names: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    allowed_names = enabled_metric_names(
        metric_groups_cfg,
        record_kind=record_kind,
        extra_protected_names=extra_protected_names,
    )
    unknown_names = sorted(
        name
        for name in record
        if name not in METRIC_DEFINITIONS_BY_NAME
        or record_kind not in METRIC_DEFINITIONS_BY_NAME[name].record_kinds
    )
    if unknown_names:
        raise KeyError(
            "Unregistered telemetry fields for "
            f"{record_kind} record: {', '.join(unknown_names)}"
        )
    return {name: value for name, value in record.items() if name in allowed_names}


def filter_update_record(record: dict[str, Any], cfg: Any | None) -> dict[str, Any]:
    return filter_metric_record(
        record,
        metric_groups_cfg=metric_groups_cfg_from_config(cfg),
        record_kind="update",
        extra_protected_names=protected_metric_names(cfg),
    )


def filter_event_record(record: dict[str, Any], cfg: Any | None) -> dict[str, Any]:
    return filter_metric_record(
        record,
        metric_groups_cfg=metric_groups_cfg_from_config(cfg),
        record_kind="event",
        extra_protected_names=protected_metric_names(cfg),
    )


_ROLLOUT_ALWAYS_COMPUTE_KEYS: frozenset[str] = frozenset(
    {
        "samples",
        "env_steps",
        "episode_done",
        "average_reward",
        "episode_reward_mean",
        "episodes_2p",
        "episodes_4p",
        "wins_2p",
        "first_places_4p",
        "placement_4p_sum",
        "survival_time_sum",
        "score_share_sum",
        "ship_differential_sum",
    }
)


def rollout_collection_enabled_groups(cfg: Any | None) -> set[str]:
    """Metric groups that may be collected during rollout (respects lean mode)."""

    enabled = set(enabled_metric_groups(metric_groups_cfg_from_config(cfg)))
    training = getattr(cfg, "training", None) if cfg is not None else None
    if training is not None and bool(getattr(training, "lean_rollout_metrics", False)):
        enabled.discard("opponent_composition")
        enabled.discard("trajectory_shield_debug")
    return enabled


def prune_scalar_metrics[T](
    metrics: dict[str, T],
    allowed_keys: frozenset[str] | set[str],
) -> dict[str, T]:
    """Drop metric entries whose keys are not in ``allowed_keys``."""

    allowed = frozenset(allowed_keys)
    return {name: value for name, value in metrics.items() if name in allowed}


def required_rollout_scalar_names(cfg: Any | None) -> tuple[str, ...]:
    enabled_update_names = enabled_metric_names(
        metric_groups_cfg_from_config(cfg),
        record_kind="update",
        extra_protected_names=protected_metric_names(cfg),
    )
    required_names = set(ROLLOUT_INTERNAL_REQUIRED_METRIC_NAMES)
    required_names.update(enabled_update_names & ROLLOUT_OUTPUT_METRIC_NAMES)
    return tuple(name for name in ROLLOUT_SCALAR_ORDER if name in required_names)


def rollout_compute_scalar_keys(cfg: Any | None) -> frozenset[str]:
    """Rollout scalar keys to materialize during collection and merge paths."""

    enabled_groups = rollout_collection_enabled_groups(cfg)
    required = set(required_rollout_scalar_names(cfg)) - set(FINALIZED_ROLLOUT_RATE_KEYS)
    keys = set(_ROLLOUT_ALWAYS_COMPUTE_KEYS)
    keys.update(required)

    for key in ROLLOUT_INTERNAL_SCALAR_KEYS:
        definition = METRIC_DEFINITIONS_BY_NAME.get(key)
        if definition is not None and definition.group in enabled_groups:
            keys.add(key)

    for key in LOGGED_ROLLOUT_SCALAR_KEYS + ROLLOUT_CHUNK_ONLY_SCALAR_KEYS:
        if key in keys:
            continue
        definition = METRIC_DEFINITIONS_BY_NAME.get(key)
        if definition is not None and definition.group in enabled_groups:
            keys.add(key)

    keys.difference_update(FINALIZED_ROLLOUT_RATE_KEYS)
    return frozenset(keys)




def rollout_merge_scalar_keys(cfg: Any | None) -> frozenset[str]:
    """Rollout scalar keys retained after cross-chunk merge and rate finalization."""

    keys = set(rollout_compute_scalar_keys(cfg))
    keys.update(
        name
        for name in required_rollout_scalar_names(cfg)
        if name in FINALIZED_ROLLOUT_RATE_KEYS
    )
    training = getattr(cfg, "training", None) if cfg is not None else None
    plateau_metric = (
        str(getattr(training, "plateau_metric", "") or "").strip()
        if training is not None
        else ""
    )
    if plateau_metric:
        keys.add(plateau_metric)
    return frozenset(keys)


def required_ppo_metric_names(
    cfg: Any | None, available_metric_names: tuple[str, ...] | list[str] | set[str]
) -> tuple[str, ...]:
    enabled_update_names = enabled_metric_names(
        metric_groups_cfg_from_config(cfg),
        record_kind="update",
        extra_protected_names=protected_metric_names(cfg),
    )
    required_names = set(PPO_INTERNAL_REQUIRED_METRIC_NAMES)
    required_names.update(enabled_update_names)
    available_names = set(available_metric_names)
    return tuple(
        name
        for name in PPO_METRIC_ORDER
        if name in required_names and name in available_names
    )
