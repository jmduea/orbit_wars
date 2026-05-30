from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.opponents.jax_actions.sampling import OPPONENT_SLOT_COUNT_KEYS

ZERO_F32 = jnp.array(0.0, dtype=jnp.float32)

TRAJECTORY_SHIELD_COUNT_KEYS: tuple[str, ...] = (
    "trajectory_shield_blocked_count",
    "trajectory_shield_blocked_sun_count",
    "trajectory_shield_blocked_bounds_count",
    "trajectory_shield_blocked_unintended_hit_count",
    "trajectory_shield_blocked_horizon_count",
    "trajectory_shield_fallback_noop_count",
)

OPPONENT_SLOT_METRIC_KEYS: tuple[str, ...] = (
    *OPPONENT_SLOT_COUNT_KEYS,
    "opponent_historical_fallback_latest_slots",
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
    "trajectory_shield_legal_non_noop_rate",
    *OPPONENT_SLOT_METRIC_KEYS,
    "stop_rate",
    "mean_active_launches_per_turn",
)

# Rates derived only after cross-chunk or cross-group aggregation.
FINALIZED_ROLLOUT_RATE_KEYS: tuple[str, ...] = (
    "win_rate_2p",
    "first_place_rate_4p",
    "average_placement_4p",
    "survival_time",
    "score_share",
    "overall_win_rate",
)

# Backward-compatible alias for train/tests imports.
_BASE_ROLLOUT_SCALAR_KEYS = BASE_ROLLOUT_SCALAR_KEYS


def _safe_rate(num: jax.Array, denom: jax.Array) -> jax.Array:
    return jnp.where(denom > 0.0, num / denom, ZERO_F32)


def trajectory_shield_legal_rate(*, legal: jax.Array, original: jax.Array) -> jax.Array:
    return _safe_rate(legal, original)


def _base_episode_metrics(
    *,
    data: dict[str, jax.Array],
    cfg: TrainConfig,
) -> dict[str, jax.Array]:
    done_float = data["done"].astype(jnp.float32)
    episode_done = done_float.sum()
    episodes_2p = jnp.where(cfg.task.player_count == 2, episode_done, ZERO_F32)
    episodes_4p = jnp.where(cfg.task.player_count == 4, episode_done, ZERO_F32)
    first_place_sum = (data["terminal_is_first"] * done_float).sum()
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
    include_opponent_slots: bool,
    data: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    episode_done = base["episode_done"]
    episodes_2p = base["episodes_2p"]
    episodes_4p = base["episodes_4p"]
    first_place_sum = base["first_place_sum"]

    opponent_slots = {
        key: (data[key].sum() if include_opponent_slots and key in data else ZERO_F32)
        for key in OPPONENT_SLOT_METRIC_KEYS
    }

    return {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * env_count,
            dtype=jnp.float32,
        ),
        "samples": samples,
        **{key: ZERO_F32 for key in TRAJECTORY_SHIELD_COUNT_KEYS},
        "trajectory_shield_legal_non_noop_count": ZERO_F32,
        "trajectory_shield_original_non_noop_count": ZERO_F32,
        "trajectory_shield_legal_non_noop_rate": ZERO_F32,
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
        **opponent_slots,
        "stop_rate": ZERO_F32,
        "mean_active_launches_per_turn": ZERO_F32,
        "loss_sample_count_2p": ZERO_F32,
        "loss_sample_count_4p": ZERO_F32,
    }


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
    metrics["stop_rate"] = _safe_rate(stop_sum, active_sum)
    metrics["mean_active_launches_per_turn"] = _safe_rate(launch_sum, turn_count)


def rollout_metrics(
    *,
    data: dict[str, jax.Array],
    cfg: TrainConfig,
    env_count: int,
) -> dict[str, jax.Array]:
    """Compute rollout metrics compatible with microbatch aggregation."""

    base = _base_episode_metrics(data=data, cfg=cfg)
    samples = data["target_index"].astype(jnp.float32).size
    metrics = _core_metric_fields(
        base=base,
        cfg=cfg,
        env_count=env_count,
        samples=samples,
        include_opponent_slots=not cfg.training.lean_rollout_metrics,
        data=data,
    )
    if not cfg.training.lean_rollout_metrics:
        _apply_shield_metrics(metrics, data)
    _apply_factorized_metrics(metrics, data)
    return metrics
