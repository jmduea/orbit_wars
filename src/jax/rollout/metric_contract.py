"""Canonical rollout metric key contract shared by JAX rollout and telemetry."""

from __future__ import annotations

from src.jax.rollout.planet_flow_metric_descriptors import (
    PLANET_FLOW_CONTROL_COUNT_KEYS,
    PLANET_FLOW_CONTROL_DELTA_KEYS,
    PLANET_FLOW_CONTROL_RATE_KEYS,
    PLANET_FLOW_COUNT_KEYS,
    PLANET_FLOW_RATE_KEYS,
)

TRAJECTORY_SHIELD_COUNT_KEYS: tuple[str, ...] = (
    "trajectory_shield_blocked_count",
    "trajectory_shield_blocked_sun_count",
    "trajectory_shield_blocked_bounds_count",
    "trajectory_shield_blocked_unintended_hit_count",
    "trajectory_shield_blocked_horizon_count",
    "trajectory_shield_fallback_noop_count",
)

OPPONENT_SLOT_COUNT_KEYS: tuple[str, ...] = (
    "opponent_slots_total",
    "opponent_slots_latest",
    "opponent_slots_historical",
    "opponent_slots_random",
    "opponent_slots_noop",
    "opponent_slots_nearest_sniper",
    "opponent_slots_turtle",
    "opponent_slots_opportunistic",
)

OPPONENT_SLOT_METRIC_KEYS: tuple[str, ...] = (
    *OPPONENT_SLOT_COUNT_KEYS,
    "opponent_historical_fallback_latest_slots",
)

# Chunk intermediates used for cross-chunk rate finalization; never logged to telemetry.
ROLLOUT_INTERNAL_SCALAR_KEYS: tuple[str, ...] = (
    "trajectory_shield_legal_non_noop_count",
    "trajectory_shield_original_non_noop_count",
    "launch_ship_count_sum",
    "active_launch_count",
)

# Extra per-chunk keys emitted by rollout collection but not aggregated or logged.
ROLLOUT_CHUNK_ONLY_SCALAR_KEYS: tuple[str, ...] = (
    "loss_sample_count_2p",
    "loss_sample_count_4p",
)

# Sum/count keys materialized once per rollout chunk before cross-chunk finalize.
BASE_ROLLOUT_SCALAR_KEYS: tuple[str, ...] = (
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
    *TRAJECTORY_SHIELD_COUNT_KEYS,
    "trajectory_shield_legal_non_noop_count",
    "trajectory_shield_original_non_noop_count",
    "launch_ship_count_sum",
    "active_launch_count",
    "trajectory_shield_legal_non_noop_rate",
    *OPPONENT_SLOT_METRIC_KEYS,
    "stop_rate",
    "mean_active_launches_per_turn",
    *PLANET_FLOW_COUNT_KEYS,
    *PLANET_FLOW_CONTROL_COUNT_KEYS,
)

# Rates derived only after cross-chunk or cross-group aggregation.
FINALIZED_ROLLOUT_RATE_KEYS: tuple[str, ...] = (
    "win_rate_2p",
    "first_place_rate_4p",
    "average_placement_4p",
    "survival_time",
    "score_share",
    "overall_win_rate",
    "mean_ships_per_launch",
    *PLANET_FLOW_RATE_KEYS,
    *PLANET_FLOW_CONTROL_RATE_KEYS,
    *PLANET_FLOW_CONTROL_DELTA_KEYS,
)

# Logged rollout scalars synced with telemetry registry update records.
LOGGED_ROLLOUT_SCALAR_KEYS: tuple[str, ...] = (
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
    "win_rate_2p",
    "first_place_rate_4p",
    "average_placement_4p",
    "overall_win_rate",
    "planet_flow_unreachable_demand_rate",
    "planet_flow_held_demand_rate",
    "planet_flow_emitted_ship_mass_rate",
    "planet_flow_capacity_drop_rate",
    "planet_flow_small_launch_rate",
    "planet_flow_duplicate_source_target_rate",
    "survival_time",
    "score_share",
    "trajectory_shield_blocked_count",
    "trajectory_shield_blocked_sun_count",
    "trajectory_shield_blocked_bounds_count",
    "trajectory_shield_blocked_unintended_hit_count",
    "trajectory_shield_blocked_horizon_count",
    "trajectory_shield_fallback_noop_count",
    "trajectory_shield_legal_non_noop_rate",
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
    "mean_ships_per_launch",
    *PLANET_FLOW_COUNT_KEYS,
    *PLANET_FLOW_CONTROL_COUNT_KEYS,
    *PLANET_FLOW_RATE_KEYS,
    *PLANET_FLOW_CONTROL_RATE_KEYS,
    *PLANET_FLOW_CONTROL_DELTA_KEYS,
)

ROLLOUT_ALLOWED_SCALAR_KEYS: frozenset[str] = frozenset(
    (
        *BASE_ROLLOUT_SCALAR_KEYS,
        *FINALIZED_ROLLOUT_RATE_KEYS,
        *ROLLOUT_CHUNK_ONLY_SCALAR_KEYS,
    )
)
