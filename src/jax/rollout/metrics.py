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
    "decision_count",
    "noop_count",
    "friendly_target_count",
    "enemy_target_count",
    "neutral_target_count",
    *TRAJECTORY_SHIELD_COUNT_KEYS,
    "trajectory_shield_legal_non_noop_count",
    "trajectory_shield_original_non_noop_count",
    "trajectory_shield_legal_non_noop_rate",
    "overall_win_rate",
    "noop_percent",
    "friendly_target_percent",
    "enemy_target_percent",
    "neutral_target_percent",
    "opponent_current_slots",
    "opponent_random_slots",
    "opponent_snapshot_slots",
    *OPPONENT_SLOT_METRIC_KEYS,
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
    "stop_rate",
    "mean_active_launches_per_turn",
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
    decision_count = jnp.asarray(data["target_index"].size, dtype=jnp.float32)
    noop_count = (data["target_index"] == 0).astype(jnp.float32).sum()
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
        "decision_count": decision_count,
        "noop_count": noop_count,
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
    decision_count = base["decision_count"]
    noop_count = base["noop_count"]

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
        "valid_non_noop_targets_sum": ZERO_F32,
        "valid_non_noop_target_rows": decision_count,
        "only_noop_rows": ZERO_F32,
        "valid_non_noop_targets_per_row": ZERO_F32,
        "only_noop_fraction": ZERO_F32,
        **{key: ZERO_F32 for key in TRAJECTORY_SHIELD_COUNT_KEYS},
        "trajectory_shield_legal_non_noop_count": ZERO_F32,
        "trajectory_shield_original_non_noop_count": ZERO_F32,
        "trajectory_shield_legal_non_noop_rate": ZERO_F32,
        "episode_done": episode_done,
        "win_episode_rows": ZERO_F32,
        "loss_episode_rows": ZERO_F32,
        "non_noop_count": ZERO_F32,
        "launched_ship_count": ZERO_F32,
        "launched_ship_total": ZERO_F32,
        "launched_ship_speed_total": ZERO_F32,
        "won_planets_owned_total": ZERO_F32,
        "lost_planets_owned_total": ZERO_F32,
        "won_planets_lost_total": ZERO_F32,
        "lost_planets_lost_total": ZERO_F32,
        "won_planets_taken_total": ZERO_F32,
        "lost_planets_taken_total": ZERO_F32,
        "won_garrisoned_ships_per_planet_total": ZERO_F32,
        "lost_garrisoned_ships_per_planet_total": ZERO_F32,
        "won_planet_diff_total": ZERO_F32,
        "lost_planet_diff_total": ZERO_F32,
        "won_production_diff_total": ZERO_F32,
        "lost_production_diff_total": ZERO_F32,
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
        "decision_count": decision_count,
        "noop_count": noop_count,
        "friendly_target_count": ZERO_F32,
        "enemy_target_count": ZERO_F32,
        "neutral_target_count": ZERO_F32,
        "overall_win_rate": _safe_rate(first_place_sum, episode_done),
        "noop_percent": jnp.where(
            decision_count > 0.0, (noop_count / decision_count) * 100.0, ZERO_F32
        ),
        "friendly_target_percent": ZERO_F32,
        "enemy_target_percent": ZERO_F32,
        "neutral_target_percent": ZERO_F32,
        **opponent_slots,
        "opponent_current_slots": opponent_slots["opponent_slots_latest"],
        "opponent_random_slots": opponent_slots["opponent_slots_random"],
        "opponent_snapshot_slots": opponent_slots["opponent_slots_historical"],
        "won_non_noop_actions_per_step": ZERO_F32,
        "lost_non_noop_actions_per_step": ZERO_F32,
        "won_avg_fleet_launch_size": ZERO_F32,
        "lost_avg_fleet_launch_size": ZERO_F32,
        "won_avg_planets_owned": ZERO_F32,
        "lost_avg_planets_owned": ZERO_F32,
        "won_avg_planets_lost": ZERO_F32,
        "lost_avg_planets_lost": ZERO_F32,
        "won_avg_planets_taken": ZERO_F32,
        "lost_avg_planets_taken": ZERO_F32,
        "won_avg_garrisoned_ships_per_planet": ZERO_F32,
        "lost_avg_garrisoned_ships_per_planet": ZERO_F32,
        "won_avg_planet_diff": ZERO_F32,
        "lost_avg_planet_diff": ZERO_F32,
        "won_avg_production_diff": ZERO_F32,
        "lost_avg_production_diff": ZERO_F32,
        "won_avg_launch_fleet_speed": ZERO_F32,
        "lost_avg_launch_fleet_speed": ZERO_F32,
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
