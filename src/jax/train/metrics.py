from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.rollout.metrics import trajectory_shield_legal_rate
from src.telemetry.metric_registry import (
    prune_scalar_metrics,
    rollout_merge_scalar_keys,
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
        return metrics_by_chunk[0]
    return finalize_cross_chunk_rate_metrics(merge_metric_dicts(metrics_by_chunk))


def prune_merged_rollout_metrics(
    metrics: dict[str, jax.Array], cfg: TrainConfig
) -> dict[str, jax.Array]:
    return prune_scalar_metrics(metrics, rollout_merge_scalar_keys(cfg))
