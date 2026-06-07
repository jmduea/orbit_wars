"""Canonical rollout metric key contract shared by JAX rollout and telemetry."""

from __future__ import annotations

from src.jax.rollout.planet_flow_metric_descriptors import (
    PLANET_FLOW_CONTROL_COUNT_KEYS,
    PLANET_FLOW_CONTROL_DELTA_KEYS,
    PLANET_FLOW_CONTROL_RATE_KEYS,
    PLANET_FLOW_COUNT_KEYS,
    PLANET_FLOW_RATE_KEYS,
)
from src.telemetry.rollout_contract_builder import (
    BASE_ROLLOUT_SCALAR_KEYS,
    FINALIZED_ROLLOUT_RATE_KEYS,
    LOGGED_ROLLOUT_SCALAR_KEYS,
    OPPONENT_SLOT_COUNT_KEYS,
    OPPONENT_SLOT_METRIC_KEYS,
    ROLLOUT_ALLOWED_SCALAR_KEYS,
    ROLLOUT_CHUNK_ONLY_SCALAR_KEYS,
    ROLLOUT_INTERNAL_SCALAR_KEYS,
    TRAJECTORY_SHIELD_COUNT_KEYS,
)
