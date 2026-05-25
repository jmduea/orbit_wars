from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig


def _zero_scalar() -> jax.Array:
    return jnp.array(0.0, dtype=jnp.float32)


def _base_episode_metrics(
    *,
    data: dict[str, jax.Array],
    cfg: TrainConfig,
) -> dict[str, jax.Array]:
    """Shared rollout reductions used by full and lean diagnostics."""

    row_mask = jnp.ones_like(data["target_index"], dtype=jnp.float32)
    done_float = data["done"].astype(jnp.float32)
    reward_mean = data["reward"].mean()
    episode_done = done_float.sum()
    episode_reward_sum = (data["reward"] * done_float).sum()
    episodes_2p = jnp.where(cfg.task.player_count == 2, episode_done, 0.0)
    episodes_4p = jnp.where(cfg.task.player_count == 4, episode_done, 0.0)
    first_place_sum = (data["terminal_is_first"] * done_float).sum()
    placement_4p_sum = jnp.where(
        cfg.task.player_count == 4,
        (data["terminal_placement"] * done_float).sum(),
        0.0,
    )
    survival_time_sum = (data["terminal_survival_time"] * done_float).sum()
    score_share_sum = (data["terminal_score_share"] * done_float).sum()
    selected_target = data["target_index"]
    decision_count = row_mask.sum()
    noop_count = ((selected_target == 0).astype(jnp.float32) * row_mask).sum()
    return {
        "row_mask": row_mask,
        "done_float": done_float,
        "reward_mean": reward_mean,
        "episode_done": episode_done,
        "episode_reward_sum": episode_reward_sum,
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "first_place_sum": first_place_sum,
        "placement_4p_sum": placement_4p_sum,
        "survival_time_sum": survival_time_sum,
        "score_share_sum": score_share_sum,
        "selected_target": selected_target,
        "decision_count": decision_count,
        "noop_count": noop_count,
    }


def _core_metric_fields(
    *,
    base: dict[str, jax.Array],
    cfg: TrainConfig,
    env_count: int,
    samples: jax.Array,
    zero: jax.Array,
    include_opponent_slots: bool,
    data: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    episode_done = base["episode_done"]
    episodes_2p = base["episodes_2p"]
    episodes_4p = base["episodes_4p"]
    first_place_sum = base["first_place_sum"]
    placement_4p_sum = base["placement_4p_sum"]
    survival_time_sum = base["survival_time_sum"]
    score_share_sum = base["score_share_sum"]
    decision_count = base["decision_count"]
    noop_count = base["noop_count"]
    reward_mean = base["reward_mean"]
    episode_reward_sum = base["episode_reward_sum"]

    opponent_slot_keys = (
        "opponent_slots_total",
        "opponent_slots_latest",
        "opponent_slots_historical",
        "opponent_slots_random",
        "opponent_slots_noop",
        "opponent_slots_nearest_sniper",
        "opponent_slots_turtle",
        "opponent_slots_opportunistic",
        "opponent_historical_fallback_latest_slots",
    )
    opponent_slots = {
        key: (data[key].sum() if include_opponent_slots and key in data else zero)
        for key in opponent_slot_keys
    }

    return {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * env_count,
            dtype=jnp.float32,
        ),
        "samples": samples,
        "valid_non_noop_targets_sum": zero,
        "valid_non_noop_target_rows": base["row_mask"].sum(),
        "only_noop_rows": zero,
        "valid_non_noop_targets_per_row": zero,
        "only_noop_fraction": zero,
        "trajectory_shield_blocked_count": zero,
        "trajectory_shield_blocked_sun_count": zero,
        "trajectory_shield_blocked_bounds_count": zero,
        "trajectory_shield_blocked_unintended_hit_count": zero,
        "trajectory_shield_blocked_horizon_count": zero,
        "trajectory_shield_fallback_noop_count": zero,
        "trajectory_shield_legal_non_noop_count": zero,
        "trajectory_shield_original_non_noop_count": zero,
        "trajectory_shield_legal_non_noop_rate": zero,
        "episode_done": episode_done,
        "win_episode_rows": zero,
        "loss_episode_rows": zero,
        "non_noop_count": zero,
        "launched_ship_count": zero,
        "launched_ship_total": zero,
        "launched_ship_speed_total": zero,
        "won_planets_owned_total": zero,
        "lost_planets_owned_total": zero,
        "won_planets_lost_total": zero,
        "lost_planets_lost_total": zero,
        "won_planets_taken_total": zero,
        "lost_planets_taken_total": zero,
        "won_garrisoned_ships_per_planet_total": zero,
        "lost_garrisoned_ships_per_planet_total": zero,
        "won_planet_diff_total": zero,
        "lost_planet_diff_total": zero,
        "won_production_diff_total": zero,
        "lost_production_diff_total": zero,
        "average_reward": reward_mean,
        "episode_reward_mean": jnp.where(
            episode_done > 0.0, episode_reward_sum / episode_done, 0.0
        ),
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "wins_2p": jnp.where(cfg.task.player_count == 2, first_place_sum, 0.0),
        "first_places_4p": jnp.where(cfg.task.player_count == 4, first_place_sum, 0.0),
        "placement_4p_sum": placement_4p_sum,
        "survival_time_sum": survival_time_sum,
        "score_share_sum": score_share_sum,
        "decision_count": decision_count,
        "noop_count": noop_count,
        "friendly_target_count": zero,
        "enemy_target_count": zero,
        "neutral_target_count": zero,
        "win_rate_2p": jnp.where(episodes_2p > 0.0, first_place_sum / episodes_2p, 0.0),
        "first_place_rate_4p": jnp.where(
            episodes_4p > 0.0, first_place_sum / episodes_4p, 0.0
        ),
        "average_placement_4p": jnp.where(
            episodes_4p > 0.0, placement_4p_sum / episodes_4p, 0.0
        ),
        "survival_time": jnp.where(
            episode_done > 0.0, survival_time_sum / episode_done, 0.0
        ),
        "score_share": jnp.where(
            episode_done > 0.0, score_share_sum / episode_done, 0.0
        ),
        "noop_percent": jnp.where(
            decision_count > 0.0, (noop_count / decision_count) * 100.0, 0.0
        ),
        "friendly_target_percent": zero,
        "enemy_target_percent": zero,
        "neutral_target_percent": zero,
        "overall_win_rate": jnp.where(
            episode_done > 0.0, first_place_sum / episode_done, 0.0
        ),
        **opponent_slots,
        "opponent_current_slots": opponent_slots["opponent_slots_latest"],
        "opponent_random_slots": opponent_slots["opponent_slots_random"],
        "opponent_snapshot_slots": opponent_slots["opponent_slots_historical"],
        "won_non_noop_actions_per_step": zero,
        "lost_non_noop_actions_per_step": zero,
        "won_avg_fleet_launch_size": zero,
        "lost_avg_fleet_launch_size": zero,
        "won_avg_planets_owned": zero,
        "lost_avg_planets_owned": zero,
        "won_avg_planets_lost": zero,
        "lost_avg_planets_lost": zero,
        "won_avg_planets_taken": zero,
        "lost_avg_planets_taken": zero,
        "won_avg_garrisoned_ships_per_planet": zero,
        "lost_avg_garrisoned_ships_per_planet": zero,
        "won_avg_planet_diff": zero,
        "lost_avg_planet_diff": zero,
        "won_avg_production_diff": zero,
        "lost_avg_production_diff": zero,
        "won_avg_launch_fleet_speed": zero,
        "lost_avg_launch_fleet_speed": zero,
        "loss_sample_count_2p": zero,
        "loss_sample_count_4p": zero,
    }


def _apply_shield_metrics(metrics: dict[str, jax.Array], data: dict[str, jax.Array]) -> None:
    original_non_noop = data.get("trajectory_shield_original_non_noop_count")
    if original_non_noop is None:
        return
    legal_non_noop = data["trajectory_shield_legal_non_noop_count"].sum()
    original_total = original_non_noop.sum()
    metrics["trajectory_shield_legal_non_noop_count"] = legal_non_noop
    metrics["trajectory_shield_original_non_noop_count"] = original_total
    metrics["trajectory_shield_legal_non_noop_rate"] = jnp.where(
        original_total > 0.0,
        legal_non_noop / original_total,
        0.0,
    )
    for key in (
        "trajectory_shield_blocked_count",
        "trajectory_shield_blocked_sun_count",
        "trajectory_shield_blocked_bounds_count",
        "trajectory_shield_blocked_unintended_hit_count",
        "trajectory_shield_blocked_horizon_count",
        "trajectory_shield_fallback_noop_count",
    ):
        metrics[key] = data[key].sum()


def rollout_metrics(
    *,
    data: dict[str, jax.Array],
    cfg: TrainConfig,
    env_count: int,
) -> dict[str, jax.Array]:
    """Compute rollout metrics compatible with microbatch aggregation."""

    base = _base_episode_metrics(data=data, cfg=cfg)
    zero = _zero_scalar()
    samples = data["target_index"].astype(jnp.float32).size
    metrics = _core_metric_fields(
        base=base,
        cfg=cfg,
        env_count=env_count,
        samples=samples,
        zero=zero,
        include_opponent_slots=not cfg.training.lean_rollout_metrics,
        data=data,
    )
    if not cfg.training.lean_rollout_metrics:
        _apply_shield_metrics(metrics, data)
    return metrics
