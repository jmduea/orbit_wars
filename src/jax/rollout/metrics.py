from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.ship_action import ship_count_for_action
from src.jax.rollout.metric_contract import (
    BASE_ROLLOUT_SCALAR_KEYS,
    FINALIZED_ROLLOUT_RATE_KEYS,
    OPPONENT_SLOT_METRIC_KEYS,
    PLANET_FLOW_CONTROL_COUNT_KEYS,
    PLANET_FLOW_COUNT_KEYS,
    TRAJECTORY_SHIELD_COUNT_KEYS,
)
from src.telemetry.metric_registry import (
    prune_scalar_metrics,
    rollout_collection_enabled_groups,
    rollout_compute_scalar_keys,
)

ZERO_F32 = jnp.array(0.0, dtype=jnp.float32)

# Backward-compatible alias for train/tests imports.
_BASE_ROLLOUT_SCALAR_KEYS = BASE_ROLLOUT_SCALAR_KEYS

__all__ = (
    "BASE_ROLLOUT_SCALAR_KEYS",
    "FINALIZED_ROLLOUT_RATE_KEYS",
    "OPPONENT_SLOT_METRIC_KEYS",
    "TRAJECTORY_SHIELD_COUNT_KEYS",
    "_BASE_ROLLOUT_SCALAR_KEYS",
    "rollout_metrics",
    "trajectory_shield_legal_rate",
)


def _safe_rate(num: jax.Array, denom: jax.Array) -> jax.Array:
    return jnp.where(denom > 0.0, num / denom, ZERO_F32)


def trajectory_shield_legal_rate(*, legal: jax.Array, original: jax.Array) -> jax.Array:
    return _safe_rate(legal, original)


def _binary_terminal_only(cfg: TrainConfig) -> bool:
    """True when terminal reward is pure ±1 binary with no step shaping."""

    reward_cfg = cfg.reward
    if reward_cfg.terminal_reward_mode.strip().lower() != "binary_win":
        return False
    return not (
        reward_cfg.early_terminal_reward_shaping_enabled
        or float(reward_cfg.reward_capture_planet) != 0.0
        or float(reward_cfg.reward_ship_delta) != 0.0
        or float(reward_cfg.reward_production_delta) != 0.0
    )


def _episode_first_place_sum(
    data: dict[str, jax.Array], done_float: jax.Array, cfg: TrainConfig
) -> jax.Array:
    """Count first-place finishes for overall_win_rate.

    For pure ``binary_win`` (preflight / noop gates), wins follow the terminal
    reward sign on done steps so ``overall_win_rate`` stays consistent with
    ``episode_reward_mean``. Other terminal modes keep ``terminal_is_first``.
    """

    if _binary_terminal_only(cfg):
        return ((data["reward"] * done_float) > 0.0).astype(jnp.float32).sum()
    return (data["terminal_is_first"] * done_float).sum()


def _base_episode_metrics(
    *,
    data: dict[str, jax.Array],
    cfg: TrainConfig,
) -> dict[str, jax.Array]:
    done_float = data["done"].astype(jnp.float32)
    episode_done = done_float.sum()
    episodes_2p = jnp.where(cfg.task.player_count == 2, episode_done, ZERO_F32)
    episodes_4p = jnp.where(cfg.task.player_count == 4, episode_done, ZERO_F32)
    first_place_sum = _episode_first_place_sum(data, done_float, cfg)
    return {
        "done_float": done_float,
        "reward_mean": data["reward"].mean(),
        "episode_done": episode_done,
        "episode_reward_sum": (data["reward"] * done_float).sum(),
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "first_place_sum": first_place_sum,
        "placement_4p_sum": jnp.where(
            cfg.task.player_count == 4,
            (data["terminal_placement"] * done_float).sum(),
            ZERO_F32,
        ),
        "survival_time_sum": (data["terminal_survival_time"] * done_float).sum(),
        "score_share_sum": (data["terminal_score_share"] * done_float).sum(),
        "ship_differential_sum": (
            data["terminal_ship_differential"] * done_float
        ).sum(),
    }


def _core_metric_fields(
    *,
    base: dict[str, jax.Array],
    cfg: TrainConfig,
    env_count: int,
    samples: jax.Array,
    compute_keys: frozenset[str],
    data: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    episode_done = base["episode_done"]
    episodes_2p = base["episodes_2p"]
    episodes_4p = base["episodes_4p"]
    first_place_sum = base["first_place_sum"]

    metrics: dict[str, jax.Array] = {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * env_count,
            dtype=jnp.float32,
        ),
        "samples": samples,
        "episode_done": episode_done,
        "average_reward": base["reward_mean"],
        "episode_reward_mean": _safe_rate(base["episode_reward_sum"], episode_done),
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "wins_2p": jnp.where(cfg.task.player_count == 2, first_place_sum, ZERO_F32),
        "first_places_4p": jnp.where(
            cfg.task.player_count == 4, first_place_sum, ZERO_F32
        ),
        "placement_4p_sum": base["placement_4p_sum"],
        "survival_time_sum": base["survival_time_sum"],
        "score_share_sum": base["score_share_sum"],
        "ship_differential_sum": base["ship_differential_sum"],
    }

    if any(key in compute_keys for key in OPPONENT_SLOT_METRIC_KEYS):
        metrics.update(
            {
                key: data[key].sum()
                for key in OPPONENT_SLOT_METRIC_KEYS
                if key in compute_keys and key in data
            }
        )

    if any(key in compute_keys for key in PLANET_FLOW_COUNT_KEYS):
        metrics.update(
            {
                key: data[key].sum()
                for key in PLANET_FLOW_COUNT_KEYS
                if key in compute_keys and key in data
            }
        )
    if any(key in compute_keys for key in PLANET_FLOW_CONTROL_COUNT_KEYS):
        metrics.update(
            {
                key: data[key].sum()
                for key in PLANET_FLOW_CONTROL_COUNT_KEYS
                if key in compute_keys and key in data
            }
        )

    if any(key in compute_keys for key in TRAJECTORY_SHIELD_COUNT_KEYS):
        metrics.update({key: ZERO_F32 for key in TRAJECTORY_SHIELD_COUNT_KEYS if key in compute_keys})
        if "trajectory_shield_legal_non_noop_count" in compute_keys:
            metrics["trajectory_shield_legal_non_noop_count"] = ZERO_F32
        if "trajectory_shield_original_non_noop_count" in compute_keys:
            metrics["trajectory_shield_original_non_noop_count"] = ZERO_F32
        if "trajectory_shield_legal_non_noop_rate" in compute_keys:
            metrics["trajectory_shield_legal_non_noop_rate"] = ZERO_F32

    if "stop_rate" in compute_keys:
        metrics["stop_rate"] = ZERO_F32
    if "mean_active_launches_per_turn" in compute_keys:
        metrics["mean_active_launches_per_turn"] = ZERO_F32
    if "launch_ship_count_sum" in compute_keys:
        metrics["launch_ship_count_sum"] = ZERO_F32
    if "active_launch_count" in compute_keys:
        metrics["active_launch_count"] = ZERO_F32
    if "loss_sample_count_2p" in compute_keys:
        metrics["loss_sample_count_2p"] = ZERO_F32
    if "loss_sample_count_4p" in compute_keys:
        metrics["loss_sample_count_4p"] = ZERO_F32

    return prune_scalar_metrics(metrics, compute_keys)


def _apply_shield_metrics(metrics: dict[str, jax.Array], data: dict[str, jax.Array]) -> None:
    original_non_noop = data.get("trajectory_shield_original_non_noop_count")
    if original_non_noop is None:
        return
    legal_non_noop = data["trajectory_shield_legal_non_noop_count"].sum()
    original_total = original_non_noop.sum()
    metrics["trajectory_shield_legal_non_noop_count"] = legal_non_noop
    metrics["trajectory_shield_original_non_noop_count"] = original_total
    metrics["trajectory_shield_legal_non_noop_rate"] = trajectory_shield_legal_rate(
        legal=legal_non_noop,
        original=original_total,
    )
    for key in TRAJECTORY_SHIELD_COUNT_KEYS:
        if key in metrics:
            metrics[key] = data[key].sum()


def _apply_factorized_metrics(metrics: dict[str, jax.Array], data: dict[str, jax.Array]) -> None:
    stop_flag = data.get("stop_flag")
    step_mask = data.get("step_mask")
    ship_bucket = data.get("ship_bucket")
    if stop_flag is None or step_mask is None or ship_bucket is None:
        return
    active = step_mask.astype(jnp.float32)
    active_sum = active.sum()
    stop_sum = (stop_flag.astype(jnp.float32) * active).sum()
    non_stop = active * (1.0 - stop_flag.astype(jnp.float32))
    ship_fraction = data.get("ship_fraction")
    if ship_fraction is not None:
        launch_sum = (non_stop * (ship_fraction.astype(jnp.float32) > 0.0)).sum()
    else:
        launch_sum = (non_stop * (ship_bucket.astype(jnp.float32) > 0.0)).sum()
    turn_count = jnp.asarray(stop_flag.shape[0] * stop_flag.shape[1], dtype=jnp.float32)
    if "stop_rate" in metrics:
        metrics["stop_rate"] = _safe_rate(stop_sum, active_sum)
    if "mean_active_launches_per_turn" in metrics:
        metrics["mean_active_launches_per_turn"] = _safe_rate(launch_sum, turn_count)


def _apply_planet_flow_metrics(
    metrics: dict[str, jax.Array],
    data: dict[str, jax.Array],
) -> None:
    emitted_launches = data.get("planet_flow_emitted_launch_count")
    if emitted_launches is None:
        return
    turn_count = jnp.asarray(
        emitted_launches.shape[0] * emitted_launches.shape[1], dtype=jnp.float32
    )
    if "stop_rate" in metrics:
        metrics["stop_rate"] = ZERO_F32
    if "mean_active_launches_per_turn" in metrics:
        metrics["mean_active_launches_per_turn"] = _safe_rate(
            emitted_launches.sum(), turn_count
        )


def _apply_launch_sizing_metrics(
    metrics: dict[str, jax.Array],
    data: dict[str, jax.Array],
    cfg: TrainConfig,
) -> None:
    if "launch_ship_count_sum" not in metrics and "active_launch_count" not in metrics:
        return

    emitted_launches = data.get("planet_flow_emitted_launch_count")
    emitted_ship_mass = data.get("planet_flow_emitted_ship_mass_sum")
    if emitted_launches is not None and emitted_ship_mass is not None:
        metrics["launch_ship_count_sum"] = emitted_ship_mass.sum()
        metrics["active_launch_count"] = emitted_launches.sum()
        return

    source_index = data.get("source_index")
    initial_planet_ships = data.get("initial_planet_ships")
    stop_flag = data.get("stop_flag")
    step_mask = data.get("step_mask")
    ship_bucket = data.get("ship_bucket")
    if (
        source_index is None
        or initial_planet_ships is None
        or stop_flag is None
        or step_mask is None
        or ship_bucket is None
    ):
        return

    active = step_mask.astype(jnp.float32)
    non_stop = active * (1.0 - stop_flag.astype(jnp.float32))
    available_ships = jnp.squeeze(
        jnp.take_along_axis(
            initial_planet_ships[..., None, :],
            source_index[..., None],
            axis=-1,
        ),
        axis=-1,
    )
    ship_counts = ship_count_for_action(
        available_ships,
        ship_bucket,
        data.get("ship_fraction"),
        cfg,
    )
    launch_active = non_stop * (ship_counts > 0.0).astype(jnp.float32)
    metrics["launch_ship_count_sum"] = (ship_counts * launch_active).sum()
    metrics["active_launch_count"] = launch_active.sum()


def rollout_metrics(
    *,
    data: dict[str, jax.Array],
    cfg: TrainConfig,
    env_count: int,
) -> dict[str, jax.Array]:
    """Compute rollout metrics compatible with microbatch aggregation."""

    compute_keys = rollout_compute_scalar_keys(cfg)
    collection_groups = rollout_collection_enabled_groups(cfg)
    base = _base_episode_metrics(data=data, cfg=cfg)
    if "planet_flow_target_bucket" in data:
        samples = jnp.asarray(
            data["planet_flow_target_bucket"].shape[0]
            * data["planet_flow_target_bucket"].shape[1],
            dtype=jnp.float32,
        )
    else:
        samples = jnp.asarray(data["target_index"].size, dtype=jnp.float32)
    metrics = _core_metric_fields(
        base=base,
        cfg=cfg,
        env_count=env_count,
        samples=samples,
        compute_keys=compute_keys,
        data=data,
    )
    if "trajectory_shield_debug" in collection_groups:
        _apply_shield_metrics(metrics, data)
    if "action_decision" in collection_groups:
        if "planet_flow_target_bucket" in data:
            _apply_planet_flow_metrics(metrics, data)
        else:
            _apply_factorized_metrics(metrics, data)
    if "debug" in collection_groups:
        _apply_launch_sizing_metrics(metrics, data, cfg)
    return metrics
