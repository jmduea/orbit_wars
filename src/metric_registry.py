from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

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
        "approx_kl", "losses", "Approximate KL divergence between old and new policy."
    ),
    _metric("total_loss", "losses", "Final weighted PPO loss used for optimization."),
    _metric("policy_loss_2p", "losses", "PPO policy loss for 2-player samples."),
    _metric("value_loss_2p", "losses", "PPO value loss for 2-player samples."),
    _metric("entropy_2p", "losses", "Action entropy for 2-player samples."),
    _metric("approx_kl_2p", "losses", "Approximate KL for 2-player samples."),
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
        "timing",
        "Wall-clock seconds spent collecting 2-player rollout groups.",
    ),
    _metric(
        "rollout_seconds_4p",
        "timing",
        "Wall-clock seconds spent collecting 4-player rollout groups.",
    ),
    _metric(
        "env_steps_per_sec_2p",
        "timing",
        "2-player environment steps processed per second over the full update.",
    ),
    _metric(
        "env_steps_per_sec_4p",
        "timing",
        "4-player environment steps processed per second over the full update.",
    ),
    _metric(
        "rollout_env_steps_per_sec_2p",
        "timing",
        "2-player environment steps processed per second during 2-player rollout collection.",
    ),
    _metric(
        "rollout_env_steps_per_sec_4p",
        "timing",
        "4-player environment steps processed per second during 4-player rollout collection.",
    ),
    _metric(
        "samples_per_sec_2p",
        "timing",
        "2-player learner decision samples processed per second over the full update.",
    ),
    _metric(
        "samples_per_sec_4p",
        "timing",
        "4-player learner decision samples processed per second over the full update.",
    ),
    _metric(
        "rollout_samples_per_sec_2p",
        "timing",
        "2-player learner decision samples processed per second during 2-player rollout collection.",
    ),
    _metric(
        "rollout_samples_per_sec_4p",
        "timing",
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
        "curriculum_phase_id",
        "curriculum",
        "Legacy stage label emitted for downstream consumers.",
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
        "opponent_composition",
        "opponent_composition",
        "Nested opponent family composition summary for WandB flattening.",
    ),
    _metric(
        "noop_percent",
        "action_decision",
        "Percent of learner decisions that selected the noop target.",
    ),
    _metric(
        "friendly_target_percent",
        "action_decision",
        "Percent of learner decisions that targeted friendly planets.",
    ),
    _metric(
        "enemy_target_percent",
        "action_decision",
        "Percent of learner decisions that targeted enemy planets.",
    ),
    _metric(
        "neutral_target_percent",
        "action_decision",
        "Percent of learner decisions that targeted neutral planets.",
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
        "won_non_noop_actions_per_step",
        "game_state",
        "Mean non-noop actions per step in winning episodes.",
    ),
    _metric(
        "lost_non_noop_actions_per_step",
        "game_state",
        "Mean non-noop actions per step in losing episodes.",
    ),
    _metric(
        "won_avg_fleet_launch_size",
        "game_state",
        "Mean launched fleet size in winning episodes.",
    ),
    _metric(
        "lost_avg_fleet_launch_size",
        "game_state",
        "Mean launched fleet size in losing episodes.",
    ),
    _metric(
        "won_avg_planets_owned",
        "game_state",
        "Mean owned planets in winning episodes.",
    ),
    _metric(
        "lost_avg_planets_owned",
        "game_state",
        "Mean owned planets in losing episodes.",
    ),
    _metric(
        "won_avg_planets_lost",
        "game_state",
        "Mean planets lost in winning episodes.",
    ),
    _metric(
        "lost_avg_planets_lost",
        "game_state",
        "Mean planets lost in losing episodes.",
    ),
    _metric(
        "won_avg_planets_taken",
        "game_state",
        "Mean planets captured in winning episodes.",
    ),
    _metric(
        "lost_avg_planets_taken",
        "game_state",
        "Mean planets captured in losing episodes.",
    ),
    _metric(
        "won_avg_garrisoned_ships_per_planet",
        "game_state",
        "Mean garrisoned ships per owned planet in winning episodes.",
    ),
    _metric(
        "lost_avg_garrisoned_ships_per_planet",
        "game_state",
        "Mean garrisoned ships per owned planet in losing episodes.",
    ),
    _metric(
        "won_avg_planet_diff",
        "game_state",
        "Mean planet-count delta in winning episodes.",
    ),
    _metric(
        "lost_avg_planet_diff",
        "game_state",
        "Mean planet-count delta in losing episodes.",
    ),
    _metric(
        "won_avg_production_diff",
        "game_state",
        "Mean production delta in winning episodes.",
    ),
    _metric(
        "lost_avg_production_diff",
        "game_state",
        "Mean production delta in losing episodes.",
    ),
    _metric(
        "won_avg_launch_fleet_speed",
        "game_state",
        "Mean launch fleet speed in winning episodes.",
    ),
    _metric(
        "lost_avg_launch_fleet_speed",
        "game_state",
        "Mean launch fleet speed in losing episodes.",
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
    ),
    _metric(
        "historical_snapshot_ages_updates",
        "historical_pool",
        "Snapshot ages in updates for the historical pool.",
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

ROLLOUT_SCALAR_ORDER: tuple[str, ...] = (
    "samples",
    "env_steps",
    "episode_done",
    "average_reward",
    "episode_reward_mean",
    "win_rate_2p",
    "first_place_rate_4p",
    "average_placement_4p",
    "overall_win_rate",
    "noop_percent",
    "friendly_target_percent",
    "enemy_target_percent",
    "neutral_target_percent",
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
    "won_non_noop_actions_per_step",
    "lost_non_noop_actions_per_step",
    "won_avg_fleet_launch_size",
    "lost_avg_fleet_launch_size",
    "won_avg_planets_owned",
    "lost_avg_planets_owned",
    "won_avg_planets_lost",
    "lost_avg_planets_lost",
    "won_avg_planets_taken",
    "lost_avg_planets_taken",
    "won_avg_garrisoned_ships_per_planet",
    "lost_avg_garrisoned_ships_per_planet",
    "won_avg_planet_diff",
    "lost_avg_planet_diff",
    "won_avg_production_diff",
    "lost_avg_production_diff",
    "won_avg_launch_fleet_speed",
    "lost_avg_launch_fleet_speed",
)

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
    "total_loss",
    "policy_loss_2p",
    "value_loss_2p",
    "entropy_2p",
    "approx_kl_2p",
    "total_loss_2p",
    "loss_sample_count_2p",
    "policy_loss_4p",
    "value_loss_4p",
    "entropy_4p",
    "approx_kl_4p",
    "total_loss_4p",
    "loss_sample_count_4p",
    "minibatches",
)

PPO_INTERNAL_REQUIRED_METRIC_NAMES: frozenset[str] = frozenset({"approx_kl"})

NON_SCALAR_UPDATE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "opponent_composition",
        "seed_scheduler_policy",
        "seed_scheduler_plateau_metric",
        "curriculum_stage_id",
        "curriculum_phase_id",
        "historical_snapshot_ids",
        "historical_snapshot_ages_updates",
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


def required_rollout_scalar_names(cfg: Any | None) -> tuple[str, ...]:
    enabled_update_names = enabled_metric_names(
        metric_groups_cfg_from_config(cfg),
        record_kind="update",
        extra_protected_names=protected_metric_names(cfg),
    )
    required_names = set(ROLLOUT_INTERNAL_REQUIRED_METRIC_NAMES)
    required_names.update(enabled_update_names & ROLLOUT_OUTPUT_METRIC_NAMES)
    return tuple(name for name in ROLLOUT_SCALAR_ORDER if name in required_names)


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
