from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.config import TrainConfig
from src.jax.rollout.metric_contract import FINALIZED_ROLLOUT_RATE_KEYS
from src.jax.rollout.planet_flow_metric_descriptors import (
    PLANET_FLOW_CONTROL_COUNT_KEYS,
    PLANET_FLOW_CONTROL_DELTA_KEYS,
    PLANET_FLOW_COUNT_KEYS,
)
from src.jax.train.metrics import finalize_cross_chunk_rate_metrics, sum_metric_dicts
from src.telemetry.metric_registry import (
    METRIC_DEFINITIONS_BY_NAME,
    rollout_compute_scalar_keys,
)
from src.telemetry.planet_flow_registry import planet_flow_metric_definitions


def _planet_flow_count_metrics() -> dict[str, jnp.ndarray]:
    return {
        "episodes_2p": jnp.array(1.0),
        "episodes_4p": jnp.array(0.0),
        "wins_2p": jnp.array(1.0),
        "first_places_4p": jnp.array(0.0),
        "placement_4p_sum": jnp.array(0.0),
        "episode_done": jnp.array(1.0),
        "survival_time_sum": jnp.array(10.0),
        "score_share_sum": jnp.array(0.5),
        "planet_flow_demanded_mass_sum": jnp.array(200.0),
        "planet_flow_unreachable_demand_mass_sum": jnp.array(50.0),
        "planet_flow_held_demand_mass_sum": jnp.array(25.0),
        "planet_flow_requested_ship_mass_sum": jnp.array(20.0),
        "planet_flow_emitted_ship_mass_sum": jnp.array(15.0),
        "planet_flow_capacity_dropped_launch_count": jnp.array(1.0),
        "planet_flow_emitted_launch_count": jnp.array(3.0),
        "planet_flow_small_launch_count": jnp.array(1.0),
        "planet_flow_duplicate_source_target_count": jnp.array(0.0),
        "planet_flow_control_demanded_mass_sum": jnp.array(250.0),
        "planet_flow_control_unreachable_demand_mass_sum": jnp.array(125.0),
        "planet_flow_control_held_demand_mass_sum": jnp.array(50.0),
        "planet_flow_control_requested_ship_mass_sum": jnp.array(12.0),
        "planet_flow_control_emitted_ship_mass_sum": jnp.array(6.0),
        "planet_flow_control_capacity_dropped_launch_count": jnp.array(2.0),
        "planet_flow_control_emitted_launch_count": jnp.array(2.0),
        "planet_flow_control_small_launch_count": jnp.array(2.0),
        "planet_flow_control_duplicate_source_target_count": jnp.array(1.0),
    }


def test_planet_flow_rates_finalize_from_count_keys() -> None:
    metrics = _planet_flow_count_metrics()

    finalized = finalize_cross_chunk_rate_metrics(metrics)

    assert float(finalized["planet_flow_unreachable_demand_rate"]) == pytest.approx(
        0.25
    )
    assert float(finalized["planet_flow_held_demand_rate"]) == pytest.approx(0.125)
    assert float(finalized["planet_flow_emitted_ship_mass_rate"]) == pytest.approx(
        0.75
    )
    assert float(finalized["planet_flow_capacity_drop_rate"]) == pytest.approx(0.25)
    assert float(finalized["planet_flow_small_launch_rate"]) == pytest.approx(1.0 / 3.0)
    assert float(finalized["planet_flow_duplicate_source_target_rate"]) == 0.0
    assert float(finalized["planet_flow_control_unreachable_demand_rate"]) == 0.5
    assert float(finalized["planet_flow_control_emitted_ship_mass_rate"]) == 0.5
    assert float(
        finalized["planet_flow_emitted_launch_count_delta_vs_control"]
    ) == 1.0
    assert float(
        finalized["planet_flow_unreachable_demand_rate_delta_vs_control"]
    ) == pytest.approx(-0.25)


def test_planet_flow_rates_zero_when_demand_below_floor() -> None:
    metrics = _planet_flow_count_metrics()
    metrics["planet_flow_demanded_mass_sum"] = jnp.array(4.0)
    metrics["planet_flow_control_demanded_mass_sum"] = jnp.array(4.0)

    finalized = finalize_cross_chunk_rate_metrics(metrics)

    assert float(finalized["planet_flow_held_demand_rate"]) == 0.0
    assert float(finalized["planet_flow_unreachable_demand_rate"]) == 0.0
    assert float(finalized["planet_flow_control_held_demand_rate"]) == 0.0


def test_planet_flow_rates_finalize_for_single_metric_dict() -> None:
    finalized = sum_metric_dicts([_planet_flow_count_metrics()])

    assert float(finalized["overall_win_rate"]) == 1.0
    assert float(finalized["planet_flow_unreachable_demand_rate"]) == pytest.approx(
        0.25
    )


def test_planet_flow_metrics_are_registered_for_action_decision_group() -> None:
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.action_decision = True

    keys = rollout_compute_scalar_keys(cfg)

    assert "planet_flow_demanded_mass_sum" in keys
    assert "planet_flow_control_demanded_mass_sum" in keys
    assert "planet_flow_unreachable_demand_rate" in METRIC_DEFINITIONS_BY_NAME
    assert (
        "planet_flow_unreachable_demand_rate_delta_vs_control"
        in METRIC_DEFINITIONS_BY_NAME
    )
    assert (
        METRIC_DEFINITIONS_BY_NAME["planet_flow_unreachable_demand_rate"].group
        == "action_decision"
    )


def test_planet_flow_descriptors_align_contract_and_registry() -> None:
    registry_names = {definition.name for definition in planet_flow_metric_definitions()}

    assert set(PLANET_FLOW_COUNT_KEYS).issubset(registry_names)
    assert set(PLANET_FLOW_CONTROL_COUNT_KEYS).issubset(registry_names)
    assert set(PLANET_FLOW_CONTROL_DELTA_KEYS).issubset(registry_names)
    assert set(PLANET_FLOW_CONTROL_DELTA_KEYS).issubset(set(FINALIZED_ROLLOUT_RATE_KEYS))


def test_planet_flow_control_delta_keys_finalize_to_finite_numeric_values() -> None:
    finalized = finalize_cross_chunk_rate_metrics(_planet_flow_count_metrics())

    for key in PLANET_FLOW_CONTROL_DELTA_KEYS:
        assert key in finalized
        value = float(finalized[key])
        assert np.isfinite(value)
