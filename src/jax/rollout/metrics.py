from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.features.registry import (
    candidate_feature_schema,
    global_feature_schema,
    self_feature_schema,
)
from src.game.constants import MAX_FLEET_SPEED, MAX_PLANETS, MAX_PRODUCTION
from src.jax.env import fleet_speed
from src.jax.features import JaxTurnBatch
from src.jax.rollout.types import JaxTransitionBatch
from src.opponents.jax_actions.builders import ship_count_for_bucket_jax



def _zero_scalar() -> jax.Array:
    return jnp.array(0.0, dtype=jnp.float32)


def _base_episode_metrics(
    *,
    data: dict[str, jax.Array],
    transitions: JaxTransitionBatch,
    turn_batch: JaxTurnBatch,
    cfg: TrainConfig,
) -> dict[str, jax.Array]:
    """Shared rollout reductions used by full and lean diagnostics."""

    row_mask = transitions.decision_mask.astype(jnp.float32)
    done_float = data["done"].astype(jnp.float32)
    reward_mean = data["reward"].mean()
    episode_done = done_float.sum()
    episode_reward_sum = (data["reward"] * done_float).sum()
    episodes_2p = jnp.where(cfg.task.player_count == 2, episode_done, 0.0)
    episodes_4p = jnp.where(cfg.task.player_count == 4, episode_done, 0.0)
    first_place_sum = (data["terminal_is_first"] * done_float).sum()
    placement_4p_sum = jnp.where(
        cfg.task.player_count == 4, (data["terminal_placement"] * done_float).sum(), 0.0
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


def _rollout_diagnostics(
    *,
    data: dict[str, jax.Array],
    transitions: JaxTransitionBatch,
    turn_batch: JaxTurnBatch,
    cfg: TrainConfig,
    opponent_slots: jax.Array,
    snapshot_share: jax.Array,
    current_share: jax.Array,
    random_share: jax.Array,
) -> dict[str, jax.Array]:
    self_schema = self_feature_schema(cfg.task)
    candidate_schema = candidate_feature_schema(cfg.task)
    global_schema = global_feature_schema(cfg.task)

    base = _base_episode_metrics(
        data=data, transitions=transitions, turn_batch=turn_batch, cfg=cfg
    )
    row_mask = base["row_mask"]
    done_float = base["done_float"]
    reward_mean = base["reward_mean"]
    episode_done = base["episode_done"]
    episode_reward_sum = base["episode_reward_sum"]
    episodes_2p = base["episodes_2p"]
    episodes_4p = base["episodes_4p"]
    first_place_sum = base["first_place_sum"]
    placement_4p_sum = base["placement_4p_sum"]
    survival_time_sum = base["survival_time_sum"]
    score_share_sum = base["score_share_sum"]
    selected_target = base["selected_target"]
    decision_count = base["decision_count"]
    noop_count = base["noop_count"]

    valid_non_noop_targets = (
        data["candidate_mask"][..., 1:].astype(jnp.float32).sum(axis=-1)
    )
    valid_non_noop_targets_sum = (valid_non_noop_targets[..., None] * row_mask).sum()
    valid_non_noop_target_rows = row_mask.sum()
    only_noop_rows = (
        (valid_non_noop_targets[..., None] <= 0.0).astype(jnp.float32) * row_mask
    ).sum()
    only_noop_fraction = jnp.where(
        valid_non_noop_target_rows > 0.0,
        only_noop_rows / valid_non_noop_target_rows,
        0.0,
    )
    non_noop_count = (((selected_target != 0).astype(jnp.float32)) * row_mask).sum()
    source_ships = (
        data["self_features"][..., self_schema.slice("source_ships")].squeeze(-1)
        * cfg.task.max_ships
    )[..., None]
    launched_ships = ship_count_for_bucket_jax(
        source_ships, data["ship_bucket"], cfg.task.ship_bucket_count
    )
    launched_ship_mask = (selected_target != 0).astype(jnp.float32) * row_mask
    launched_ship_count = launched_ship_mask.sum()
    launched_ship_total = (launched_ships * launched_ship_mask).sum()
    launched_ship_speed_total = (
        launched_ship_mask * fleet_speed(launched_ships, MAX_FLEET_SPEED)
    ).sum()

    terminal_row_mask = row_mask * done_float[..., None, None]
    win_row_mask = terminal_row_mask * data["terminal_is_first"][..., None, None]
    loss_row_mask = terminal_row_mask * (
        1.0 - data["terminal_is_first"][..., None, None]
    )
    win_episode_rows = win_row_mask.sum()
    loss_episode_rows = loss_row_mask.sum()

    planet_fractions_slice = global_schema.slice("planet_fractions")
    ship_fractions_slice = global_schema.slice("ship_fractions")
    planet_delta_slots_slice = global_schema.slice("planet_delta_slots")
    owner_production_slice = global_schema.slice("owner_relative_production")

    planet_fractions = data["global_features"][..., planet_fractions_slice]
    ship_fractions = data["global_features"][..., ship_fractions_slice]
    planet_delta_slots = data["global_features"][..., planet_delta_slots_slice]
    owner_production = data["global_features"][..., owner_production_slice]

    my_planets = planet_fractions[..., 0] * MAX_PLANETS
    my_garrison_ships = ship_fractions[..., 0] * (MAX_PLANETS * cfg.task.max_ships)
    planet_delta = planet_delta_slots[..., 0] * MAX_PLANETS
    production_diff = owner_production[..., 0] * MAX_PRODUCTION
    planet_diff = planet_delta
    planets_taken_step = jnp.maximum(planet_delta, 0.0)
    planets_lost_step = jnp.maximum(-planet_delta, 0.0)
    selected_candidate_features = jnp.take_along_axis(
        data["candidate_features"][..., None, :, :],
        selected_target[..., None, None].repeat(
            data["candidate_features"].shape[-1], axis=-1
        ),
        axis=4,
    ).squeeze(axis=4)

    target_ownership_slice = candidate_schema.slice("target_ownership_flags")
    target_ownership = selected_candidate_features[..., target_ownership_slice]

    neutral_target_count = (target_ownership[..., 0] * row_mask).sum()
    friendly_target_count = (target_ownership[..., 1] * row_mask).sum()
    enemy_target_count = (target_ownership[..., 2] * row_mask).sum()
    garrisoned_ships_per_planet = my_garrison_ships / jnp.maximum(my_planets, 1.0)
    won_planets_owned_total = (my_planets[..., None] * win_row_mask).sum()
    lost_planets_owned_total = (my_planets[..., None] * loss_row_mask).sum()
    won_planets_lost_total = (planets_lost_step[..., None] * win_row_mask).sum()
    lost_planets_lost_total = (planets_lost_step[..., None] * loss_row_mask).sum()
    won_planets_taken_total = (planets_taken_step[..., None] * win_row_mask).sum()
    lost_planets_taken_total = (planets_taken_step[..., None] * loss_row_mask).sum()
    won_garrisoned_ships_per_planet_total = (
        garrisoned_ships_per_planet[..., None] * win_row_mask
    ).sum()
    lost_garrisoned_ships_per_planet_total = (
        garrisoned_ships_per_planet[..., None] * loss_row_mask
    ).sum()
    won_planet_diff_total = (planet_diff[..., None] * win_row_mask).sum()
    lost_planet_diff_total = (planet_diff[..., None] * loss_row_mask).sum()
    won_production_diff_total = (production_diff[..., None] * win_row_mask).sum()
    lost_production_diff_total = (production_diff[..., None] * loss_row_mask).sum()

    metrics = {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * turn_batch.self_features.shape[0],
            dtype=jnp.float32,
        ),
        "samples": transitions.decision_mask.astype(jnp.float32).sum(),
        "valid_non_noop_targets_sum": valid_non_noop_targets_sum,
        "valid_non_noop_target_rows": valid_non_noop_target_rows,
        "only_noop_rows": only_noop_rows,
        "valid_non_noop_targets_per_row": jnp.where(
            valid_non_noop_target_rows > 0.0,
            valid_non_noop_targets_sum / valid_non_noop_target_rows,
            0.0,
        ),
        "only_noop_fraction": only_noop_fraction,
        "trajectory_shield_blocked_count": data[
            "trajectory_shield_blocked_count"
        ].sum(),
        "trajectory_shield_blocked_sun_count": data[
            "trajectory_shield_blocked_sun_count"
        ].sum(),
        "trajectory_shield_blocked_bounds_count": data[
            "trajectory_shield_blocked_bounds_count"
        ].sum(),
        "trajectory_shield_blocked_unintended_hit_count": data[
            "trajectory_shield_blocked_unintended_hit_count"
        ].sum(),
        "trajectory_shield_blocked_horizon_count": data[
            "trajectory_shield_blocked_horizon_count"
        ].sum(),
        "trajectory_shield_fallback_noop_count": data[
            "trajectory_shield_fallback_noop_count"
        ].sum(),
        "trajectory_shield_legal_non_noop_count": data[
            "trajectory_shield_legal_non_noop_count"
        ].sum(),
        "trajectory_shield_original_non_noop_count": data[
            "trajectory_shield_original_non_noop_count"
        ].sum(),
        "trajectory_shield_legal_non_noop_rate": jnp.where(
            data["trajectory_shield_original_non_noop_count"].sum() > 0.0,
            data["trajectory_shield_legal_non_noop_count"].sum()
            / data["trajectory_shield_original_non_noop_count"].sum(),
            0.0,
        ),
        "episode_done": episode_done,
        "win_episode_rows": win_episode_rows,
        "loss_episode_rows": loss_episode_rows,
        "non_noop_count": non_noop_count,
        "launched_ship_count": launched_ship_count,
        "launched_ship_total": launched_ship_total,
        "launched_ship_speed_total": launched_ship_speed_total,
        "won_planets_owned_total": won_planets_owned_total,
        "lost_planets_owned_total": lost_planets_owned_total,
        "won_planets_lost_total": won_planets_lost_total,
        "lost_planets_lost_total": lost_planets_lost_total,
        "won_planets_taken_total": won_planets_taken_total,
        "lost_planets_taken_total": lost_planets_taken_total,
        "won_garrisoned_ships_per_planet_total": won_garrisoned_ships_per_planet_total,
        "lost_garrisoned_ships_per_planet_total": lost_garrisoned_ships_per_planet_total,
        "won_planet_diff_total": won_planet_diff_total,
        "lost_planet_diff_total": lost_planet_diff_total,
        "won_production_diff_total": won_production_diff_total,
        "lost_production_diff_total": lost_production_diff_total,
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
        "friendly_target_count": friendly_target_count,
        "enemy_target_count": enemy_target_count,
        "neutral_target_count": neutral_target_count,
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
        "friendly_target_percent": jnp.where(
            decision_count > 0.0, (friendly_target_count / decision_count) * 100.0, 0.0
        ),
        "enemy_target_percent": jnp.where(
            decision_count > 0.0, (enemy_target_count / decision_count) * 100.0, 0.0
        ),
        "neutral_target_percent": jnp.where(
            decision_count > 0.0, (neutral_target_count / decision_count) * 100.0, 0.0
        ),
        "overall_win_rate": jnp.where(
            episode_done > 0.0, first_place_sum / episode_done, 0.0
        ),
        "opponent_slots_total": data["opponent_slots_total"].sum(),
        "opponent_slots_latest": data["opponent_slots_latest"].sum(),
        "opponent_slots_historical": data["opponent_slots_historical"].sum(),
        "opponent_slots_random": data["opponent_slots_random"].sum(),
        "opponent_slots_noop": data["opponent_slots_noop"].sum(),
        "opponent_slots_nearest_sniper": data["opponent_slots_nearest_sniper"].sum(),
        "opponent_slots_turtle": data["opponent_slots_turtle"].sum(),
        "opponent_slots_opportunistic": data["opponent_slots_opportunistic"].sum(),
        "opponent_historical_fallback_latest_slots": data[
            "opponent_historical_fallback_latest_slots"
        ].sum(),
        "opponent_current_slots": data["opponent_slots_latest"].sum(),
        "opponent_random_slots": data["opponent_slots_random"].sum(),
        "opponent_snapshot_slots": data["opponent_slots_historical"].sum(),
        "won_non_noop_actions_per_step": jnp.where(
            win_episode_rows > 0.0,
            (non_noop_count * done_float.sum())
            / jnp.maximum(win_episode_rows * done_float.sum(), 1.0),
            0.0,
        ),
        "lost_non_noop_actions_per_step": jnp.where(
            loss_episode_rows > 0.0,
            (non_noop_count * done_float.sum())
            / jnp.maximum(loss_episode_rows * done_float.sum(), 1.0),
            0.0,
        ),
        "won_avg_fleet_launch_size": jnp.where(
            win_episode_rows > 0.0,
            launched_ship_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
        "lost_avg_fleet_launch_size": jnp.where(
            loss_episode_rows > 0.0,
            launched_ship_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
        "won_avg_planets_owned": jnp.where(
            win_episode_rows > 0.0,
            won_planets_owned_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planets_owned": jnp.where(
            loss_episode_rows > 0.0,
            lost_planets_owned_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_planets_lost": jnp.where(
            win_episode_rows > 0.0,
            won_planets_lost_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planets_lost": jnp.where(
            loss_episode_rows > 0.0,
            lost_planets_lost_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_planets_taken": jnp.where(
            win_episode_rows > 0.0,
            won_planets_taken_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planets_taken": jnp.where(
            loss_episode_rows > 0.0,
            lost_planets_taken_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_garrisoned_ships_per_planet": jnp.where(
            win_episode_rows > 0.0,
            won_garrisoned_ships_per_planet_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_garrisoned_ships_per_planet": jnp.where(
            loss_episode_rows > 0.0,
            lost_garrisoned_ships_per_planet_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_planet_diff": jnp.where(
            win_episode_rows > 0.0,
            won_planet_diff_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_planet_diff": jnp.where(
            loss_episode_rows > 0.0,
            lost_planet_diff_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_production_diff": jnp.where(
            win_episode_rows > 0.0,
            won_production_diff_total / win_episode_rows,
            0.0,
        ),
        "lost_avg_production_diff": jnp.where(
            loss_episode_rows > 0.0,
            lost_production_diff_total / loss_episode_rows,
            0.0,
        ),
        "won_avg_launch_fleet_speed": jnp.where(
            win_episode_rows > 0.0,
            launched_ship_speed_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
        "lost_avg_launch_fleet_speed": jnp.where(
            loss_episode_rows > 0.0,
            launched_ship_speed_total / jnp.maximum(launched_ship_count, 1.0),
            0.0,
        ),
    }
    return metrics


def _rollout_diagnostics_lean(
    *,
    data: dict[str, jax.Array],
    transitions: JaxTransitionBatch,
    turn_batch: JaxTurnBatch,
    cfg: TrainConfig,
) -> dict[str, jax.Array]:
    """Compute rollout metrics without per-step shield/opponent scan payloads."""

    base = _base_episode_metrics(
        data=data, transitions=transitions, turn_batch=turn_batch, cfg=cfg
    )
    row_mask = base["row_mask"]
    done_float = base["done_float"]
    reward_mean = base["reward_mean"]
    episode_done = base["episode_done"]
    episode_reward_sum = base["episode_reward_sum"]
    episodes_2p = base["episodes_2p"]
    episodes_4p = base["episodes_4p"]
    first_place_sum = base["first_place_sum"]
    placement_4p_sum = base["placement_4p_sum"]
    survival_time_sum = base["survival_time_sum"]
    score_share_sum = base["score_share_sum"]
    selected_target = base["selected_target"]
    decision_count = base["decision_count"]
    noop_count = base["noop_count"]
    zero = _zero_scalar()
    return {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * turn_batch.self_features.shape[0],
            dtype=jnp.float32,
        ),
        "samples": transitions.decision_mask.astype(jnp.float32).sum(),
        "valid_non_noop_targets_sum": zero,
        "valid_non_noop_target_rows": row_mask.sum(),
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
        "opponent_slots_total": zero,
        "opponent_slots_latest": zero,
        "opponent_slots_historical": zero,
        "opponent_slots_random": zero,
        "opponent_slots_noop": zero,
        "opponent_slots_nearest_sniper": zero,
        "opponent_slots_turtle": zero,
        "opponent_slots_opportunistic": zero,
        "opponent_historical_fallback_latest_slots": zero,
        "opponent_current_slots": zero,
        "opponent_random_slots": zero,
        "opponent_snapshot_slots": zero,
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
    }
