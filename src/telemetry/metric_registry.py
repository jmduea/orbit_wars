from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.telemetry.metric_definition import MetricDefinition, metric
from src.telemetry.metrics.assembly import assemble_non_planet_flow_metrics

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
    "rollout_phase_timing",
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
        "preflight_sweep_score",
        "planet_flow_sweep_score",
        "win_rate_delta_10",
        "approx_kl_window_mean",
        "entropy_window_mean",
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

_METRICS: tuple[MetricDefinition, ...] = assemble_non_planet_flow_metrics()

_curriculum_prob_metrics = tuple(
    metric(
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

from src.telemetry.rollout_contract_builder import (  # noqa: E402
    FINALIZED_ROLLOUT_RATE_KEYS,
    LOGGED_ROLLOUT_SCALAR_KEYS,
    ROLLOUT_CHUNK_ONLY_SCALAR_KEYS,
    ROLLOUT_INTERNAL_SCALAR_KEYS,
)

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
    required = set(required_rollout_scalar_names(cfg)) - set(
        FINALIZED_ROLLOUT_RATE_KEYS
    )
    keys = set(_ROLLOUT_ALWAYS_COMPUTE_KEYS)
    keys.update(required)

    for key in ROLLOUT_INTERNAL_SCALAR_KEYS:
        definition = METRIC_DEFINITIONS_BY_NAME.get(key)
        if definition is not None and definition.group in enabled_groups:
            keys.add(key)
    if "debug" in enabled_groups:
        keys.update({"launch_ship_count_sum", "active_launch_count"})

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
