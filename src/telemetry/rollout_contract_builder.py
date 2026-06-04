"""Build and validate JAX rollout metric key contracts against the telemetry registry.

Canonical ``BASE_*`` / ``LOGGED_*`` tuples preserve stable rollout merge order.
Membership for registry-backed keys is validated via ``rollout_scalar_role`` on
``MetricDefinition``; call ``validate_rollout_contract_registry_alignment()`` from
tests (not at import time).
"""

from __future__ import annotations

from src.jax.rollout.planet_flow_metric_descriptors import (
    PLANET_FLOW_CONTROL_COUNT_KEYS,
    PLANET_FLOW_CONTROL_DELTA_KEYS,
    PLANET_FLOW_CONTROL_RATE_KEYS,
    PLANET_FLOW_COUNT_KEYS,
    PLANET_FLOW_RATE_KEYS,
)

ROLLOUT_SCALAR_ROLES: frozenset[str] = frozenset(
    {"base_sum", "internal", "finalized_rate", "chunk_only"}
)

# Rollout collection keys not registered in METRIC_DEFINITIONS (emit-only).
_ROLLOUT_ONLY_INTERNAL_KEYS: frozenset[str] = frozenset(
    {
        "trajectory_shield_legal_non_noop_count",
        "trajectory_shield_original_non_noop_count",
        "launch_ship_count_sum",
        "active_launch_count",
    }
)

_ROLLOUT_ONLY_BASE_SUM_KEYS: frozenset[str] = frozenset(
    {
        "env_steps",
        "episode_done",
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

ROLLOUT_INTERNAL_SCALAR_KEYS: tuple[str, ...] = (
    "trajectory_shield_legal_non_noop_count",
    "trajectory_shield_original_non_noop_count",
    "launch_ship_count_sum",
    "active_launch_count",
)

ROLLOUT_CHUNK_ONLY_SCALAR_KEYS: tuple[str, ...] = (
    "loss_sample_count_2p",
    "loss_sample_count_4p",
)

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


def registry_names_for_role(
    definitions: tuple[MetricDefinition, ...],
    role: str,
) -> frozenset[str]:
    return frozenset(
        definition.name
        for definition in definitions
        if definition.rollout_scalar_role == role
    )


def _registry_names_for_role(role: str) -> frozenset[str]:
    from src.telemetry.metric_registry import METRIC_DEFINITIONS

    return registry_names_for_role(METRIC_DEFINITIONS, role)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_rollout_contract_registry_alignment() -> None:
    """Fail fast when registry rollout roles drift from the canonical contract tuples."""

    from src.telemetry.metric_registry import (
        METRIC_DEFINITIONS,
        METRIC_DEFINITIONS_BY_NAME,
    )

    _check(
        _registry_names_for_role("chunk_only")
        == frozenset(ROLLOUT_CHUNK_ONLY_SCALAR_KEYS),
        "chunk_only rollout_scalar_role keys drifted from ROLLOUT_CHUNK_ONLY_SCALAR_KEYS",
    )

    internal_registry = _registry_names_for_role("internal")
    _check(
        internal_registry == frozenset(), "internal rollout_scalar_role must stay empty"
    )
    _check(
        frozenset(ROLLOUT_INTERNAL_SCALAR_KEYS) == _ROLLOUT_ONLY_INTERNAL_KEYS,
        "ROLLOUT_INTERNAL_SCALAR_KEYS drifted from _ROLLOUT_ONLY_INTERNAL_KEYS",
    )

    base_registry = _registry_names_for_role("base_sum")
    planet_flow_keys = frozenset(
        PLANET_FLOW_COUNT_KEYS
        + PLANET_FLOW_CONTROL_COUNT_KEYS
        + PLANET_FLOW_RATE_KEYS
        + PLANET_FLOW_CONTROL_RATE_KEYS
        + PLANET_FLOW_CONTROL_DELTA_KEYS
    )
    expected_base_registry = (
        frozenset(BASE_ROLLOUT_SCALAR_KEYS)
        - _ROLLOUT_ONLY_BASE_SUM_KEYS
        - _ROLLOUT_ONLY_INTERNAL_KEYS
        - frozenset(FINALIZED_ROLLOUT_RATE_KEYS)
        - frozenset(ROLLOUT_CHUNK_ONLY_SCALAR_KEYS)
        - planet_flow_keys
    )
    _check(
        base_registry == expected_base_registry,
        "base_sum rollout_scalar_role keys drifted from BASE_ROLLOUT_SCALAR_KEYS",
    )
    _check(
        base_registry <= frozenset(BASE_ROLLOUT_SCALAR_KEYS),
        "base_sum registry keys must be a subset of BASE_ROLLOUT_SCALAR_KEYS",
    )

    finalized_registry = _registry_names_for_role("finalized_rate")
    expected_finalized_registry = frozenset(FINALIZED_ROLLOUT_RATE_KEYS) - planet_flow_keys
    _check(
        finalized_registry == expected_finalized_registry,
        "finalized_rate rollout_scalar_role keys drifted from FINALIZED_ROLLOUT_RATE_KEYS",
    )
    _check(
        finalized_registry <= frozenset(FINALIZED_ROLLOUT_RATE_KEYS),
        "finalized_rate registry keys must be a subset of FINALIZED_ROLLOUT_SCALAR_KEYS",
    )

    for definition in METRIC_DEFINITIONS:
        role = definition.rollout_scalar_role
        if role is None:
            continue
        if role not in ROLLOUT_SCALAR_ROLES:
            raise ValueError(f"Unknown rollout_scalar_role {role!r} on {definition.name!r}")
        if definition.name not in ROLLOUT_ALLOWED_SCALAR_KEYS:
            raise ValueError(
                f"Registry metric {definition.name!r} has rollout role {role!r} "
                "but is absent from ROLLOUT_ALLOWED_SCALAR_KEYS"
            )

    contract_registry_names = frozenset(
        name for name in ROLLOUT_ALLOWED_SCALAR_KEYS if name in METRIC_DEFINITIONS_BY_NAME
    )
    registry_rollout_names = frozenset(
        definition.name
        for definition in METRIC_DEFINITIONS
        if definition.rollout_scalar_role is not None
    )
    _check(
        registry_rollout_names <= contract_registry_names,
        "rollout_scalar_role registry names must be registered contract keys",
    )
    _check(
        registry_rollout_names
        == (
            _registry_names_for_role("base_sum")
            | _registry_names_for_role("finalized_rate")
            | _registry_names_for_role("chunk_only")
            | _registry_names_for_role("internal")
        ),
        "rollout_scalar_role registry names must match role partitions",
    )

    for name in LOGGED_ROLLOUT_SCALAR_KEYS:
        if name in planet_flow_keys:
            continue
        if name in METRIC_DEFINITIONS_BY_NAME and name not in registry_rollout_names:
            raise ValueError(
                f"Logged rollout key {name!r} is registered but missing rollout_scalar_role"
            )
