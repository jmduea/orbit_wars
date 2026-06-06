"""Rollout scan-step helpers extracted from ``collect_rollout_jax`` for readability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

import jax
from src.artifacts.checkpoint_compat import is_planet_flow_pointer_decoder
from src.config import TrainConfig
from src.jax.action_codec import (
    PlanetFlowPolicyOutput,
    sample_planet_flow_pressure_action,
)
from src.jax.action_sampling import _sample_shielded_sequence_with_params
from src.jax.decoder_carry import reset_decoder_hidden_on_done
from src.jax.env import (
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
    batched_step_multi_player,
)
from src.jax.features import TurnBatch, encode_turn
from src.jax.planet_flow import (
    compile_planet_flow_action,
    compile_seeded_random_planet_flow_control,
    planet_flow_sampling_target_mask,
)
from src.jax.rollout.types import JaxTrainState
from src.jax.ship_action import is_continuous_ship_mode
from src.opponents.constants import OPPONENT_HISTORICAL, OPPONENT_LATEST
from src.opponents.jax_actions.builders import (
    build_action_from_factored_batch,
    owned_planet_ships,
)
from src.opponents.jax_actions.sampling import (
    _maybe_effective_single_family_id,
    _opponent_count_metrics,
    _sample_opponent_player_action,
    _single_stage_family_id,
)
from src.opponents.pool import sample_opponent_type_ids_jax
from src.training.curriculum import StageView

from .metrics import OPPONENT_SLOT_METRIC_KEYS


@dataclass(frozen=True, slots=True)
class OpponentSlotState:
    """Resolved per-step opponent family ids and optional composition metrics."""

    effective_type_ids: jax.Array
    single_family: jax.Array
    effective_single_family_id: jax.Array
    family_counts: dict[str, jax.Array]
    historical_fallback_slots: jax.Array


@dataclass(frozen=True, slots=True)
class LearnerStepResult:
    """Policy sample outputs consumed by env step and transition recording."""

    learner_action: Any
    value: jax.Array
    log_prob: jax.Array
    decoder_hidden_out: Any
    shield_diagnostics: Any
    planet_flow_diagnostics: Any
    planet_flow_control_diagnostics: Any
    factorized_replay: dict[str, Any] | None
    planet_flow_replay: dict[str, Any] | None


def encode_opponent_perspective_batch(
    game,
    learner_player: jax.Array,
    cfg: TrainConfig,
) -> TurnBatch:
    """Encode turn features from the non-learner player perspective (2p)."""

    opp_game = game._replace(player=(1 - learner_player).astype(jnp.int32))
    return jax.vmap(lambda row: encode_turn(row, cfg.task))(opp_game)


def maybe_refresh_2p_opponent_batch(
    skip_refresh: jax.Array,
    cached_batch: TurnBatch,
    game,
    learner_player: jax.Array,
    cfg: TrainConfig,
) -> TurnBatch:
    """Keep cached opponent batch or re-encode after env step (noop paths skip)."""

    def keep_cached(_: None) -> TurnBatch:
        return cached_batch

    def refresh(_: None) -> TurnBatch:
        return encode_opponent_perspective_batch(game, learner_player, cfg)

    return jax.lax.cond(skip_refresh, keep_cached, refresh, None)


def resolve_opponent_slots(
    opp_key: jax.Array,
    *,
    env_count: int,
    cfg: TrainConfig,
    stage_view: StageView,
    learner_player: jax.Array,
    include_metrics: bool,
) -> OpponentSlotState:
    """Sample and normalize opponent family ids for this rollout step."""

    single_family_id = _single_stage_family_id(stage_view)
    effective_single_family_id = _maybe_effective_single_family_id(
        single_family_id, stage_view
    )
    single_family = single_family_id >= 0
    opponent_type_ids = sample_opponent_type_ids_jax(
        jax.random.fold_in(opp_key, 9973),
        env_count,
        cfg.task.player_count,
        ids=stage_view.family_ids,
        probs=stage_view.family_probs,
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
    has_historical = jnp.any(stage_view.snapshot_valid_mask)
    effective_type_ids = jnp.where(
        (opponent_type_ids == OPPONENT_HISTORICAL) & jnp.logical_not(has_historical),
        stage_view.fallback_family_id,
        opponent_type_ids,
    )
    family_counts: dict[str, jax.Array] = {}
    historical_fallback_slots = jnp.array(0.0, dtype=jnp.float32)
    if include_metrics:
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
    return OpponentSlotState(
        effective_type_ids=effective_type_ids,
        single_family=single_family,
        effective_single_family_id=effective_single_family_id,
        family_counts=family_counts,
        historical_fallback_slots=historical_fallback_slots,
    )


def sample_learner_step(
    learner_key: jax.Array,
    *,
    state: JaxEnvState,
    policy_batch: TurnBatch,
    batch: TurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    decoder_hidden: Any,
    env_count: int,
) -> LearnerStepResult:
    """Sample learner action via planet-flow or factorized policy path."""

    if is_planet_flow_pointer_decoder(cfg.model):
        player_count = jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32)
        output = policy.apply(
            train_state.params,
            policy_batch,
            player_count=player_count,
        )
        if not isinstance(output, PlanetFlowPolicyOutput):
            raise TypeError(
                "planet_flow_target_heatmap policy must return PlanetFlowPolicyOutput."
            )
        pressure_action = sample_planet_flow_pressure_action(
            learner_key,
            output,
            jnp.asarray(
                cfg.model.planet_flow.pressure_bucket_values,
                dtype=jnp.float32,
            ),
            planet_flow_sampling_target_mask(state.game, batch),
            deterministic=False,
        )
        compile_result = compile_planet_flow_action(
            state.game,
            batch,
            pressure_action.target_pressure,
            cfg,
        )
        control_result = compile_seeded_random_planet_flow_control(
            jax.random.fold_in(learner_key, 104_729),
            state.game,
            batch,
            cfg,
        )
        return LearnerStepResult(
            learner_action=compile_result.action,
            value=output.value,
            log_prob=pressure_action.log_prob,
            decoder_hidden_out=decoder_hidden,
            shield_diagnostics=None,
            planet_flow_diagnostics=compile_result.diagnostics,
            planet_flow_control_diagnostics=control_result.diagnostics,
            factorized_replay=None,
            planet_flow_replay={
                "planet_flow_target_bucket": pressure_action.target_bucket,
                "planet_flow_target_pressure": pressure_action.target_pressure,
                "planet_flow_target_mask": pressure_action.target_mask,
            },
        )

    sample = _sample_shielded_sequence_with_params(
        learner_key,
        state.game,
        policy_batch,
        train_state.params,
        policy,
        cfg,
        deterministic=False,
        decoder_hidden_in=decoder_hidden,
    )
    learner_action = build_action_from_factored_batch(
        state.game,
        batch,
        sample.source_index,
        sample.target_slot,
        sample.ship_bucket,
        sample.stop_flag,
        sample.step_mask,
        cfg,
        ship_fraction=sample.ship_fraction,
    )
    replay = {
        "ship_bucket_mask": sample.ship_bucket_mask,
        "target_index": sample.target_index,
        "ship_bucket": sample.ship_bucket,
        "source_index": sample.source_index,
        "target_slot": sample.target_slot,
        "stop_flag": sample.stop_flag,
        "step_mask": sample.step_mask,
        "log_prob": sample.log_prob,
        "ship_fraction": sample.ship_fraction,
    }
    return LearnerStepResult(
        learner_action=learner_action,
        value=sample.value,
        log_prob=sample.log_prob,
        decoder_hidden_out=sample.decoder_hidden_out,
        shield_diagnostics=sample.diagnostics,
        planet_flow_diagnostics=None,
        planet_flow_control_diagnostics=None,
        factorized_replay=replay,
        planet_flow_replay=None,
    )


def build_four_player_turn_batches(
    state: JaxEnvState,
    *,
    env_count: int,
    cfg: TrainConfig,
) -> tuple[Any, TurnBatch]:
    """Per-player games and encoded batches for 4p opponent sampling."""

    player_ids = jnp.arange(cfg.task.player_count, dtype=jnp.int32)
    player_games = jax.vmap(
        lambda player_id: state.game._replace(
            player=jnp.full_like(state.game.step, player_id, dtype=jnp.int32)
        )
    )(player_ids)
    flat_player_games = jax.tree.map(
        lambda x: x.reshape((cfg.task.player_count * env_count,) + x.shape[2:]),
        player_games,
    )
    flat_player_batch = jax.vmap(lambda game: encode_turn(game, cfg.task))(
        flat_player_games
    )
    player_batches = jax.tree.map(
        lambda x: x.reshape((cfg.task.player_count, env_count) + x.shape[1:]),
        flat_player_batch,
    )
    return player_games, player_batches


def step_env_with_opponents(
    *,
    cfg: TrainConfig,
    state: JaxEnvState,
    learner_action: Any,
    opp_key: jax.Array,
    opp_batch_cache: TurnBatch,
    slots: OpponentSlotState,
    train_state: JaxTrainState,
    policy: object,
    stage_view: StageView,
    historical_params_pool: dict | None,
    opponent_params_by_player: tuple[dict, ...] | None,
    env_count: int,
) -> tuple[JaxEnvState, Any]:
    """Advance env one step with sampled opponent actions (2p or 4p)."""

    if cfg.task.player_count == 2:
        opp_game = state.game._replace(
            player=(1 - state.learner_player).astype(jnp.int32)
        )
        opponent_action = _sample_opponent_player_action(
            opp_key,
            effective_type_ids=slots.effective_type_ids,
            single_family=slots.single_family,
            effective_single_family_id=slots.effective_single_family_id,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=stage_view,
            historical_params_pool=historical_params_pool,
            opp_game=opp_game,
            opp_batch=opp_batch_cache,
        )
        return batched_step(state, learner_action, opponent_action, cfg.task, cfg.reward)

    player_games, player_batches = build_four_player_turn_batches(
        state, env_count=env_count, cfg=cfg
    )
    player_ids = jnp.arange(cfg.task.player_count, dtype=jnp.int32)
    per_player_action = jax.vmap(
        lambda player_id: _sample_opponent_player_action(
            opp_key,
            effective_type_ids=slots.effective_type_ids,
            single_family=slots.single_family,
            effective_single_family_id=slots.effective_single_family_id,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=stage_view,
            historical_params_pool=historical_params_pool,
            player_id=player_id,
            player_games=player_games,
            player_batches=player_batches,
            opponent_params_by_player=opponent_params_by_player,
            learner_action=learner_action,
            learner_player=state.learner_player,
        )
    )(player_ids)
    multi_player_action = jax.tree.map(
        lambda x: jnp.moveaxis(x, 0, 1), per_player_action
    )
    return batched_step_multi_player(
        state, multi_player_action, cfg.task, cfg.reward
    )


def merge_done_episode_resets(
    *,
    result,
    next_state: JaxEnvState,
    state: JaxEnvState,
    reset_key: jax.Array,
    env_count: int,
    env_indices: jax.Array,
    cfg: TrainConfig,
    carry_enabled: bool,
    fresh_decoder_hidden: Any,
) -> tuple[JaxEnvState, TurnBatch]:
    """Reset finished env rows and merge back into the active carry."""

    def maybe_reset(new, old):
        cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
        return jnp.where(cond, new, old)

    def reset_branch(_: None) -> tuple[JaxEnvState, TurnBatch]:
        reset_keys = jax.random.split(reset_key, env_count)
        reset_states, reset_batches = batched_reset(reset_keys, cfg.task)
        reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
        reset_states, reset_batches = assign_learner_players(
            reset_states,
            env_indices,
            reset_episode_counts,
            cfg.task,
            cfg.opponents.mode.alternate_player_sides,
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

    return jax.lax.cond(
        jnp.any(result.done),
        reset_branch,
        lambda _: (next_state, result.batch),
        operand=None,
    )


def _planet_flow_metric_fields(pf_diag, pf_control_diag) -> dict[str, jax.Array]:
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


def build_scan_transition(
    *,
    batch: TurnBatch,
    state: JaxEnvState,
    result,
    learner: LearnerStepResult,
    slots: OpponentSlotState,
    cfg: TrainConfig,
    env_count: int,
    carry_enabled: bool,
    decoder_hidden: Any,
    include_opponent_metrics: bool,
    include_shield_metrics: bool,
) -> dict[str, Any]:
    """Assemble per-step transition dict for ``jax.lax.scan`` output."""

    transition: dict[str, Any] = {
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
        "value": learner.value,
        "reward": result.reward,
        "done": result.done,
        "terminal_is_first": result.terminal_is_first,
        "terminal_placement": result.terminal_placement,
        "terminal_survival_time": result.terminal_survival_time,
        "terminal_score_share": result.terminal_score_share,
        "terminal_ship_differential": result.terminal_ship_differential,
    }
    if learner.planet_flow_replay is not None:
        transition["log_prob"] = learner.log_prob
        transition.update(learner.planet_flow_replay)
    elif learner.factorized_replay is not None:
        replay = dict(learner.factorized_replay)
        ship_fraction = replay.pop("ship_fraction")
        transition.update(replay)
        if carry_enabled:
            transition["decoder_hidden"] = decoder_hidden
        if is_continuous_ship_mode(cfg):
            transition["ship_fraction"] = ship_fraction
    if include_opponent_metrics:
        transition.update(
            {
                key: (
                    slots.historical_fallback_slots
                    if key == "opponent_historical_fallback_latest_slots"
                    else slots.family_counts[key]
                )
                for key in OPPONENT_SLOT_METRIC_KEYS
            }
        )
    if include_shield_metrics and learner.shield_diagnostics is not None:
        diag = learner.shield_diagnostics
        transition.update(
            {
                "trajectory_shield_blocked_count": diag.blocked_count,
                "trajectory_shield_blocked_sun_count": diag.blocked_sun_count,
                "trajectory_shield_blocked_bounds_count": diag.blocked_bounds_count,
                "trajectory_shield_blocked_unintended_hit_count": (
                    diag.blocked_unintended_hit_count
                ),
                "trajectory_shield_blocked_horizon_count": diag.blocked_horizon_count,
                "trajectory_shield_fallback_noop_count": diag.fallback_noop_count,
                "trajectory_shield_legal_non_noop_count": diag.legal_non_noop_count,
                "trajectory_shield_original_non_noop_count": diag.original_non_noop_count,
            }
        )
    if learner.planet_flow_diagnostics is not None:
        assert learner.planet_flow_control_diagnostics is not None
        transition.update(
            _planet_flow_metric_fields(
                learner.planet_flow_diagnostics,
                learner.planet_flow_control_diagnostics,
            )
        )
    return transition
