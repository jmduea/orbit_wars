"""Shared rollout scan-step kernels for ``collect`` and ``collect_timed``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

import jax
from src.artifacts.checkpoint_compat import is_planet_flow_pointer_decoder
from src.config import TrainConfig
from src.jax.action_sampling import _sample_shielded_sequence_with_params
from src.jax.decoder_carry import reset_decoder_hidden_on_done
from src.jax.env import (
    assign_learner_players,
    batched_reset,
    batched_reset_with_pool,
    batched_step,
    batched_step_multi_player,
)
from src.jax.features import TurnBatch
from src.jax.map_pool.load import MapPoolConstants
from src.jax.normalization import ObservationNormState, normalize_turn_batch
from src.jax.ship_action import is_continuous_ship_mode
from src.opponents.constants import (
    OPPONENT_FAMILY_NAMES,
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
)
from src.opponents.jax_actions.builders import (
    build_action_from_factored_batch,
    owned_planet_ships,
)
from src.opponents.jax_actions.sampling import (
    _encode_four_player_turn_batches,
    _flatten_four_player_turn_batches,
    _opponent_count_metrics,
    _sample_flat_four_player_actions,
    _sample_opponent_player_action,
    _single_stage_family_id,
    _unflatten_player_major_action,
    _maybe_effective_single_family_id,
)
from src.opponents.pool import sample_opponent_type_ids_jax
from src.opponents.curriculum import StageView

from .metrics import OPPONENT_SLOT_METRIC_KEYS


def _policy_turn_batch(
    batch: TurnBatch,
    norm_state: ObservationNormState | None,
    cfg: TrainConfig,
) -> TurnBatch:
    if norm_state is None or not cfg.model.normalize_observations:
        return batch
    return normalize_turn_batch(batch, norm_state, cfg.model)


def _is_latest_only_family_weights(weights: dict[str, float]) -> bool:
    latest = float(weights.get("latest", 0.0))
    if latest <= 0.0:
        return False
    return all(
        family == "latest" or float(weights.get(family, 0.0)) <= 0.0
        for family in OPPONENT_FAMILY_NAMES
    )


def _static_latest_only_self_play_sample_enabled(
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
) -> bool:
    """Return whether learner/latest can share one policy sample from cfg alone."""

    if is_planet_flow_pointer_decoder(cfg.model):
        return False
    if cfg.opponents.dispatch != "self":
        return False
    if opponent_params_by_player is not None:
        return False
    if bool(cfg.curriculum.enabled):
        stages = list(cfg.curriculum.stages or [])
        if len(stages) != 1:
            return False
        weights = dict(stages[0].get("opponent_families", {}) or {})
        return _is_latest_only_family_weights(weights)
    return _is_latest_only_family_weights(dict(cfg.opponents.mix.weights or {}))


def _shield_transition_fields(diagnostics) -> dict[str, jax.Array]:
    return {
        "trajectory_shield_blocked_count": diagnostics.blocked_count,
        "trajectory_shield_blocked_sun_count": diagnostics.blocked_sun_count,
        "trajectory_shield_blocked_bounds_count": diagnostics.blocked_bounds_count,
        "trajectory_shield_blocked_unintended_hit_count": (
            diagnostics.blocked_unintended_hit_count
        ),
        "trajectory_shield_blocked_horizon_count": diagnostics.blocked_horizon_count,
        "trajectory_shield_fallback_noop_count": diagnostics.fallback_noop_count,
        "trajectory_shield_legal_non_noop_count": diagnostics.legal_non_noop_count,
        "trajectory_shield_original_non_noop_count": (
            diagnostics.original_non_noop_count
        ),
    }


def _planet_flow_transition_fields(
    pf_diag,
    pf_control_diag,
) -> dict[str, jax.Array]:
    return {
        "planet_flow_demanded_mass_sum": pf_diag.demanded_mass,
        "planet_flow_unreachable_demand_mass_sum": pf_diag.unreachable_demand_mass,
        "planet_flow_held_demand_mass_sum": pf_diag.held_demand_mass,
        "planet_flow_requested_ship_mass_sum": pf_diag.requested_ship_mass,
        "planet_flow_emitted_ship_mass_sum": pf_diag.emitted_ship_mass,
        "planet_flow_capacity_dropped_launch_count": pf_diag.capacity_dropped_launches,
        "planet_flow_emitted_launch_count": pf_diag.emitted_launch_count,
        "planet_flow_small_launch_count": pf_diag.small_launch_count,
        "planet_flow_duplicate_source_target_count": (
            pf_diag.duplicate_source_target_count
        ),
        "planet_flow_control_demanded_mass_sum": pf_control_diag.demanded_mass,
        "planet_flow_control_unreachable_demand_mass_sum": (
            pf_control_diag.unreachable_demand_mass
        ),
        "planet_flow_control_held_demand_mass_sum": pf_control_diag.held_demand_mass,
        "planet_flow_control_requested_ship_mass_sum": (
            pf_control_diag.requested_ship_mass
        ),
        "planet_flow_control_emitted_ship_mass_sum": pf_control_diag.emitted_ship_mass,
        "planet_flow_control_capacity_dropped_launch_count": (
            pf_control_diag.capacity_dropped_launches
        ),
        "planet_flow_control_emitted_launch_count": pf_control_diag.emitted_launch_count,
        "planet_flow_control_small_launch_count": pf_control_diag.small_launch_count,
        "planet_flow_control_duplicate_source_target_count": (
            pf_control_diag.duplicate_source_target_count
        ),
    }


def _take_leading_axis_rows(tree: object, indices: jax.Array, row_count: int) -> object:
    """Gather rows from pytrees whose leading axis is the flat player-seat axis."""

    def take_leaf(value):
        if (
            isinstance(value, jax.Array)
            and value.ndim > 0
            and value.shape[0] == row_count
        ):
            return jnp.take(value, indices, axis=0)
        return value

    return jax.tree.map(take_leaf, tree)


def _take_player_action_by_env(action, player_ids: jax.Array):
    """Gather one player action per env from ``(env, player, ...)`` actions."""

    return jax.tree.map(
        lambda value: jnp.take_along_axis(
            value,
            player_ids.reshape((player_ids.shape[0], 1) + (1,) * (value.ndim - 2)),
            axis=1,
        ).squeeze(axis=1),
        action,
    )


def _sample_all_player_latest_policy_actions(
    *,
    key: jax.Array,
    state,
    raw_batch: TurnBatch,
    train_state,
    policy: object,
    cfg: TrainConfig,
    env_count: int,
    norm_state: ObservationNormState | None,
    decoder_hidden: jax.Array | None,
):
    """Sample learner and latest-opponent seats in one flat policy decode."""

    player_count = int(cfg.task.player_count)
    flat_game, flat_raw_batch = _flatten_four_player_turn_batches(
        state, cfg.task, env_count
    )
    flat_policy_batch = _policy_turn_batch(flat_raw_batch, norm_state, cfg)
    flat_count = player_count * env_count
    flat_decoder_hidden = None
    if decoder_hidden is not None:
        player_ids = jnp.arange(player_count, dtype=jnp.int32)
        learner_mask = player_ids[:, None] == state.learner_player[None, :]
        flat_decoder_hidden = jnp.where(
            learner_mask[..., None],
            decoder_hidden[None, :, :],
            jnp.zeros(
                (player_count, env_count, decoder_hidden.shape[-1]),
                dtype=decoder_hidden.dtype,
            ),
        ).reshape((flat_count, decoder_hidden.shape[-1]))
    flat_sample = _sample_shielded_sequence_with_params(
        key,
        flat_game,
        flat_policy_batch,
        train_state.params,
        policy,
        cfg,
        deterministic=False,
        decoder_hidden_in=flat_decoder_hidden,
    )
    flat_action = build_action_from_factored_batch(
        flat_game,
        flat_raw_batch,
        flat_sample.source_index,
        flat_sample.target_slot,
        flat_sample.ship_bucket,
        flat_sample.stop_flag,
        flat_sample.step_mask,
        cfg,
        ship_fraction=flat_sample.ship_fraction,
    )
    multi_player_action = _unflatten_player_major_action(
        flat_action, player_count, env_count
    )
    learner_rows = state.learner_player * env_count + jnp.arange(
        env_count, dtype=jnp.int32
    )
    learner_sample = _take_leading_axis_rows(flat_sample, learner_rows, flat_count)
    learner_action = build_action_from_factored_batch(
        state.game,
        raw_batch,
        learner_sample.source_index,
        learner_sample.target_slot,
        learner_sample.ship_bucket,
        learner_sample.stop_flag,
        learner_sample.step_mask,
        cfg,
        ship_fraction=learner_sample.ship_fraction,
    )
    return learner_sample, learner_action, multi_player_action


def _build_transition(
    *,
    state,
    batch: TurnBatch,
    result,
    cfg: TrainConfig,
    env_count: int,
    value: jax.Array,
    log_prob: jax.Array,
    include_opponent_metrics: bool,
    include_shield_metrics: bool,
    family_counts: dict[str, jax.Array] | None,
    historical_fallback_slots: jax.Array,
    decoder_hidden: jax.Array | None = None,
    sample=None,
    ship_bucket_mask: jax.Array | None = None,
    target_index: jax.Array | None = None,
    ship_bucket: jax.Array | None = None,
    source_index: jax.Array | None = None,
    target_slot: jax.Array | None = None,
    stop_flag: jax.Array | None = None,
    step_mask: jax.Array | None = None,
    ship_fraction: jax.Array | None = None,
    shield_diagnostics=None,
    planet_flow_target_bucket: jax.Array | None = None,
    planet_flow_target_pressure: jax.Array | None = None,
    planet_flow_target_mask: jax.Array | None = None,
    planet_flow_diagnostics=None,
    planet_flow_control_diagnostics=None,
) -> dict:
    transition = {
        "planet_features": batch.planet_features,
        "planet_mask": batch.planet_mask,
        "edge_features": batch.edge_features,
        "edge_mask": batch.edge_mask,
        "edge_src_ids": batch.edge_src_ids,
        "edge_tgt_ids": batch.edge_tgt_ids,
        "global_features": batch.global_features,
        "theta_ref": batch.theta_ref,
        "player_count": jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32),
        "initial_planet_ships": owned_planet_ships(state.game),
        "value": value,
        "reward": result.reward,
        "done": result.done,
        "terminal_is_first": result.terminal_is_first,
        "terminal_placement": result.terminal_placement,
        "terminal_survival_time": result.terminal_survival_time,
        "terminal_score_share": result.terminal_score_share,
        "terminal_ship_differential": result.terminal_ship_differential,
    }
    if is_planet_flow_pointer_decoder(cfg.model):
        transition.update(
            {
                "log_prob": log_prob,
                "planet_flow_target_bucket": planet_flow_target_bucket,
                "planet_flow_target_pressure": planet_flow_target_pressure,
                "planet_flow_target_mask": planet_flow_target_mask,
            }
        )
    else:
        if sample is not None:
            transition.update(
                {
                    "ship_bucket_mask": sample.ship_bucket_mask,
                    "target_index": sample.target_index,
                    "ship_bucket": sample.ship_bucket,
                    "source_index": sample.source_index,
                    "target_slot": sample.target_slot,
                    "stop_flag": sample.stop_flag,
                    "step_mask": sample.step_mask,
                    "log_prob": sample.log_prob,
                }
            )
            if decoder_hidden is not None:
                transition["decoder_hidden"] = decoder_hidden
            if is_continuous_ship_mode(cfg):
                transition["ship_fraction"] = sample.ship_fraction
            if include_shield_metrics:
                transition.update(_shield_transition_fields(sample.diagnostics))
        else:
            transition.update(
                {
                    "ship_bucket_mask": ship_bucket_mask,
                    "target_index": target_index,
                    "ship_bucket": ship_bucket,
                    "source_index": source_index,
                    "target_slot": target_slot,
                    "stop_flag": stop_flag,
                    "step_mask": step_mask,
                    "log_prob": log_prob,
                }
            )
            if decoder_hidden is not None:
                transition["decoder_hidden"] = decoder_hidden
            if is_continuous_ship_mode(cfg):
                transition["ship_fraction"] = ship_fraction
            if include_shield_metrics and shield_diagnostics is not None:
                transition.update(_shield_transition_fields(shield_diagnostics))
    if include_opponent_metrics and family_counts is not None:
        transition.update(
            {
                key: (
                    historical_fallback_slots
                    if key == "opponent_historical_fallback_latest_slots"
                    else family_counts[key]
                )
                for key in OPPONENT_SLOT_METRIC_KEYS
            }
        )
    if is_planet_flow_pointer_decoder(cfg.model):
        assert planet_flow_diagnostics is not None
        assert planet_flow_control_diagnostics is not None
        transition.update(
            _planet_flow_transition_fields(
                planet_flow_diagnostics,
                planet_flow_control_diagnostics,
            )
        )
    return transition


def _reset_on_done(
    *,
    state,
    next_state,
    result,
    reset_key: jax.Array,
    env_count: int,
    env_indices: jax.Array,
    map_pool: MapPoolConstants | None,
    pool_reset_fn: Callable[[jax.Array, jax.Array], tuple[object, TurnBatch]]
    | None = None,
    cfg: TrainConfig,
    carry_enabled: bool,
    fresh_decoder_hidden,
) -> tuple[object, TurnBatch]:
    def maybe_reset(new, old):
        cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
        return jnp.where(cond, new, old)

    reset_keys = jax.random.split(reset_key, env_count)
    reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
    if map_pool is not None:
        map_ids = (reset_episode_counts + env_indices) % jnp.asarray(
            map_pool.pool_size, dtype=jnp.int32
        )
        if pool_reset_fn is not None:
            reset_states, reset_batches = pool_reset_fn(reset_keys, map_ids)
        else:
            reset_states, reset_batches = batched_reset_with_pool(
                reset_keys, cfg.task, map_pool, map_ids
            )
    else:
        reset_states, reset_batches = batched_reset(reset_keys, cfg.task)
    reset_states, reset_batches = assign_learner_players(
        reset_states,
        env_indices,
        reset_episode_counts,
        cfg.task,
        cfg.opponents.alternate_player_sides,
    )
    if carry_enabled:
        reset_states = reset_states._replace(decoder_hidden=fresh_decoder_hidden)
    merged_state = jax.tree.map(maybe_reset, reset_states, next_state)
    merged_batch = jax.tree.map(maybe_reset, reset_batches, result.batch)
    if carry_enabled:
        merged_state = merged_state._replace(
            decoder_hidden=reset_decoder_hidden_on_done(
                merged_state.decoder_hidden,
                result.done,
                fresh_decoder_hidden,
            )
        )
    return merged_state, merged_batch


@dataclass
class OpponentPhaseContext:
    effective_type_ids: jax.Array
    single_family: jax.Array
    effective_single_family_id: jax.Array
    family_counts: dict[str, jax.Array] | None
    historical_fallback_slots: jax.Array


def _sample_opponent_phase_context(
    *,
    opp_key: jax.Array,
    env_count: int,
    learner_player: jax.Array,
    cfg: TrainConfig,
    active_stage_view: StageView,
    include_opponent_metrics: bool,
) -> OpponentPhaseContext:
    single_family_id = _single_stage_family_id(active_stage_view)
    effective_single_family_id = _maybe_effective_single_family_id(
        single_family_id, active_stage_view
    )
    single_family = single_family_id >= 0
    opponent_type_ids = sample_opponent_type_ids_jax(
        jax.random.fold_in(opp_key, 9973),
        env_count,
        cfg.task.player_count,
        ids=active_stage_view.family_ids,
        probs=active_stage_view.family_probs,
    )
    opponent_type_ids = jnp.where(
        single_family,
        jnp.full(
            (env_count, cfg.task.player_count),
            single_family_id,
            dtype=jnp.int32,
        ),
        opponent_type_ids,
    )
    has_historical = jnp.any(active_stage_view.snapshot_valid_mask)
    effective_type_ids = jnp.where(
        (opponent_type_ids == OPPONENT_HISTORICAL) & jnp.logical_not(has_historical),
        active_stage_view.fallback_family_id,
        opponent_type_ids,
    )
    family_counts: dict[str, jax.Array] | None = None
    historical_fallback_slots = jnp.array(0.0, dtype=jnp.float32)
    if include_opponent_metrics:
        family_counts = _opponent_count_metrics(effective_type_ids, learner_player)
        historical_fallback_slots = (
            (
                (
                    (opponent_type_ids == OPPONENT_HISTORICAL)
                    & (effective_type_ids == OPPONENT_LATEST)
                )
                & (
                    jnp.arange(cfg.task.player_count, dtype=jnp.int32)[None, :]
                    != learner_player[:, None]
                )
            )
            .astype(jnp.float32)
            .sum()
        )
    return OpponentPhaseContext(
        effective_type_ids=effective_type_ids,
        single_family=single_family,
        effective_single_family_id=effective_single_family_id,
        family_counts=family_counts,
        historical_fallback_slots=historical_fallback_slots,
    )


def _sample_legacy_four_player_actions(
    *,
    cfg: TrainConfig,
    state,
    learner_action,
    opp_key: jax.Array,
    opponent_ctx: OpponentPhaseContext,
    train_state,
    policy: object,
    active_stage_view: StageView,
    historical_params_pool: dict | None,
    opponent_params_by_player: tuple[dict, ...] | None,
    env_count: int,
):
    player_ids = jnp.arange(cfg.task.player_count, dtype=jnp.int32)
    player_games, player_batches = _encode_four_player_turn_batches(
        state, cfg.task, env_count
    )
    per_player_action = jax.vmap(
        lambda player_id: _sample_opponent_player_action(
            opp_key,
            effective_type_ids=opponent_ctx.effective_type_ids,
            single_family=opponent_ctx.single_family,
            effective_single_family_id=opponent_ctx.effective_single_family_id,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=active_stage_view,
            historical_params_pool=historical_params_pool,
            player_id=player_id,
            player_games=player_games,
            player_batches=player_batches,
            opponent_params_by_player=opponent_params_by_player,
            learner_action=learner_action,
            learner_player=state.learner_player,
        )
    )(player_ids)
    return jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), per_player_action)


def _sample_four_player_actions(
    *,
    cfg: TrainConfig,
    state,
    learner_action,
    opp_key: jax.Array,
    opponent_ctx: OpponentPhaseContext,
    train_state,
    policy: object,
    active_stage_view: StageView,
    historical_params_pool: dict | None,
    opponent_params_by_player: tuple[dict, ...] | None,
    env_count: int,
):
    if opponent_params_by_player is not None:
        return _sample_legacy_four_player_actions(
            cfg=cfg,
            state=state,
            learner_action=learner_action,
            opp_key=opp_key,
            opponent_ctx=opponent_ctx,
            train_state=train_state,
            policy=policy,
            active_stage_view=active_stage_view,
            historical_params_pool=historical_params_pool,
            opponent_params_by_player=opponent_params_by_player,
            env_count=env_count,
        )

    flat_game, flat_batch = _flatten_four_player_turn_batches(
        state, cfg.task, env_count
    )
    return _sample_flat_four_player_actions(
        opp_key,
        flat_game=flat_game,
        flat_batch=flat_batch,
        learner_action=learner_action,
        learner_player=state.learner_player,
        effective_type_ids=opponent_ctx.effective_type_ids,
        single_family=opponent_ctx.single_family,
        effective_single_family_id=opponent_ctx.effective_single_family_id,
        train_state=train_state,
        policy=policy,
        cfg=cfg,
        stage_view=active_stage_view,
        historical_params_pool=historical_params_pool,
    )


def _env_step_with_opponents(
    *,
    cfg: TrainConfig,
    state,
    learner_action,
    opp_key: jax.Array,
    opp_batch_cache: TurnBatch,
    opponent_ctx: OpponentPhaseContext,
    train_state,
    policy: object,
    active_stage_view: StageView,
    historical_params_pool: dict | None,
    opponent_params_by_player: tuple[dict, ...] | None,
    env_count: int,
    multi_player_step_fn: Callable[[object, object], tuple[object, object]]
    | None = None,
) -> tuple[object, object]:
    if cfg.task.player_count == 2:
        opp_game = state.game._replace(
            player=(1 - state.learner_player).astype(jnp.int32)
        )
        opponent_action = _sample_opponent_player_action(
            opp_key,
            effective_type_ids=opponent_ctx.effective_type_ids,
            single_family=opponent_ctx.single_family,
            effective_single_family_id=opponent_ctx.effective_single_family_id,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=active_stage_view,
            historical_params_pool=historical_params_pool,
            opp_game=opp_game,
            opp_batch=opp_batch_cache,
            opponent_params_by_player=opponent_params_by_player,
        )
        return batched_step(
            state, learner_action, opponent_action, cfg.task, cfg.reward
        )

    multi_player_action = _sample_four_player_actions(
        cfg=cfg,
        state=state,
        learner_action=learner_action,
        opp_key=opp_key,
        opponent_ctx=opponent_ctx,
        train_state=train_state,
        policy=policy,
        active_stage_view=active_stage_view,
        historical_params_pool=historical_params_pool,
        opponent_params_by_player=opponent_params_by_player,
        env_count=env_count,
    )
    if multi_player_step_fn is not None:
        return multi_player_step_fn(state, multi_player_action)
    return batched_step_multi_player(state, multi_player_action, cfg.task, cfg.reward)
