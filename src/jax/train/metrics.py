from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.rollout.planet_flow_metric_descriptors import PLANET_FLOW_RATE_DELTA_SUFFIXES
from src.jax.rollout.metrics import trajectory_shield_legal_rate
from src.jax.train.sweep_score import PLANET_FLOW_MIN_DEMAND_MASS
from src.telemetry.metric_registry import (
    prune_scalar_metrics,
    rollout_merge_scalar_keys,
)

_PLANET_FLOW_PREFIX = "planet_flow"
_PLANET_FLOW_CONTROL_PREFIX = "planet_flow_control"


def _finalize_planet_flow_rates(
    metrics: dict[str, jax.Array],
    *,
    prefix: str,
) -> None:
    demanded = metrics[f"{prefix}_demanded_mass_sum"]
    requested_ship_mass = metrics[f"{prefix}_requested_ship_mass_sum"]
    emitted_launches = metrics[f"{prefix}_emitted_launch_count"]
    attempted_launches = (
        metrics[f"{prefix}_emitted_launch_count"]
        + metrics[f"{prefix}_capacity_dropped_launch_count"]
    )
    metrics[f"{prefix}_unreachable_demand_rate"] = jnp.where(
        demanded >= PLANET_FLOW_MIN_DEMAND_MASS,
        metrics[f"{prefix}_unreachable_demand_mass_sum"] / demanded,
        0.0,
    )
    metrics[f"{prefix}_held_demand_rate"] = jnp.where(
        demanded >= PLANET_FLOW_MIN_DEMAND_MASS,
        metrics[f"{prefix}_held_demand_mass_sum"] / demanded,
        0.0,
    )
    metrics[f"{prefix}_emitted_ship_mass_rate"] = jnp.where(
        requested_ship_mass > 0.0,
        metrics[f"{prefix}_emitted_ship_mass_sum"] / requested_ship_mass,
        0.0,
    )
    metrics[f"{prefix}_capacity_drop_rate"] = jnp.where(
        attempted_launches > 0.0,
        metrics[f"{prefix}_capacity_dropped_launch_count"] / attempted_launches,
        0.0,
    )
    metrics[f"{prefix}_small_launch_rate"] = jnp.where(
        emitted_launches > 0.0,
        metrics[f"{prefix}_small_launch_count"] / emitted_launches,
        0.0,
    )
    metrics[f"{prefix}_duplicate_source_target_rate"] = jnp.where(
        emitted_launches > 0.0,
        metrics[f"{prefix}_duplicate_source_target_count"] / emitted_launches,
        0.0,
    )


def _finalize_planet_flow_control_deltas(metrics: dict[str, jax.Array]) -> None:
    metrics["planet_flow_emitted_launch_count_delta_vs_control"] = (
        metrics["planet_flow_emitted_launch_count"]
        - metrics["planet_flow_control_emitted_launch_count"]
    )
    metrics["planet_flow_emitted_ship_mass_delta_vs_control"] = (
        metrics["planet_flow_emitted_ship_mass_sum"]
        - metrics["planet_flow_control_emitted_ship_mass_sum"]
    )
    for name in PLANET_FLOW_RATE_DELTA_SUFFIXES:
        metrics[f"planet_flow_{name}_delta_vs_control"] = (
            metrics[f"planet_flow_{name}"] - metrics[f"planet_flow_control_{name}"]
        )


def finalize_cross_chunk_rate_metrics(
    metrics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Derived rates that exist only after cross-chunk or cross-group aggregation."""

    metrics["win_rate_2p"] = jnp.where(
        metrics["episodes_2p"] > 0.0,
        metrics["wins_2p"] / metrics["episodes_2p"],
        0.0,
    )
    metrics["first_place_rate_4p"] = jnp.where(
        metrics["episodes_4p"] > 0.0,
        metrics["first_places_4p"] / metrics["episodes_4p"],
        0.0,
    )
    metrics["average_placement_4p"] = jnp.where(
        metrics["episodes_4p"] > 0.0,
        metrics["placement_4p_sum"] / metrics["episodes_4p"],
        0.0,
    )
    metrics["survival_time"] = jnp.where(
        metrics["episode_done"] > 0.0,
        metrics["survival_time_sum"] / metrics["episode_done"],
        0.0,
    )
    metrics["score_share"] = jnp.where(
        metrics["episode_done"] > 0.0,
        metrics["score_share_sum"] / metrics["episode_done"],
        0.0,
    )
    metrics["overall_win_rate"] = jnp.where(
        metrics["episode_done"] > 0.0,
        (metrics["wins_2p"] + metrics["first_places_4p"]) / metrics["episode_done"],
        0.0,
    )
    if "planet_flow_demanded_mass_sum" in metrics:
        _finalize_planet_flow_rates(metrics, prefix=_PLANET_FLOW_PREFIX)
        if "planet_flow_control_demanded_mass_sum" in metrics:
            _finalize_planet_flow_rates(metrics, prefix=_PLANET_FLOW_CONTROL_PREFIX)
            _finalize_planet_flow_control_deltas(metrics)
    if "launch_ship_count_sum" in metrics:
        metrics["mean_ships_per_launch"] = jnp.where(
            metrics["active_launch_count"] > 0.0,
            metrics["launch_ship_count_sum"] / metrics["active_launch_count"],
            0.0,
        )
    return metrics


def merge_metric_dicts(
    metrics_by_chunk: list[dict[str, jax.Array]],
) -> dict[str, jax.Array]:
    """Sum per-chunk rollout metrics while preserving the per-chunk key set."""

    if len(metrics_by_chunk) == 1:
        return metrics_by_chunk[0]
    metrics = jax.tree.map(lambda *xs: jnp.stack(xs).sum(axis=0), *metrics_by_chunk)
    reward_weight = jnp.maximum(metrics["env_steps"], 1.0)
    metrics["average_reward"] = (
        jnp.stack(
            [chunk["average_reward"] * chunk["env_steps"] for chunk in metrics_by_chunk]
        ).sum()
        / reward_weight
    )
    metrics["episode_reward_mean"] = jnp.where(
        metrics["episode_done"] > 0.0,
        jnp.stack(
            [
                chunk["episode_reward_mean"] * chunk["episode_done"]
                for chunk in metrics_by_chunk
            ]
        ).sum()
        / metrics["episode_done"],
        0.0,
    )
    if "valid_non_noop_target_rows" in metrics:
        metrics["valid_non_noop_targets_per_row"] = jnp.where(
            metrics["valid_non_noop_target_rows"] > 0.0,
            metrics["valid_non_noop_targets_sum"]
            / metrics["valid_non_noop_target_rows"],
            0.0,
        )
        metrics["only_noop_fraction"] = jnp.where(
            metrics["valid_non_noop_target_rows"] > 0.0,
            metrics["only_noop_rows"] / metrics["valid_non_noop_target_rows"],
            0.0,
        )
    if (
        "trajectory_shield_legal_non_noop_count" in metrics
        and "trajectory_shield_original_non_noop_count" in metrics
    ):
        metrics["trajectory_shield_legal_non_noop_rate"] = trajectory_shield_legal_rate(
            legal=metrics["trajectory_shield_legal_non_noop_count"],
            original=metrics["trajectory_shield_original_non_noop_count"],
        )
    return metrics


def sum_metric_dicts(
    metrics_by_chunk: list[dict[str, jax.Array]],
) -> dict[str, jax.Array]:
    if len(metrics_by_chunk) == 1:
        return finalize_cross_chunk_rate_metrics(dict(metrics_by_chunk[0]))
    return finalize_cross_chunk_rate_metrics(merge_metric_dicts(metrics_by_chunk))


def prune_merged_rollout_metrics(
    metrics: dict[str, jax.Array], cfg: TrainConfig
) -> dict[str, jax.Array]:
    return prune_scalar_metrics(metrics, rollout_merge_scalar_keys(cfg))
