"""Host-instrumented rollout collect for per-phase wall-clock breakdown (opt-in)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.action_sampling import _sample_shielded_sequence_with_params
from src.jax.decoder_carry import (
    decoder_carry_enabled,
    empty_decoder_hidden,
    reset_decoder_hidden_on_done,
)
from src.jax.env import (
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_reset_with_pool,
    batched_step,
    batched_step_multi_player,
)
from src.jax.features import TurnBatch, encode_turn
from src.jax.map_pool.load import MapPoolConstants
from src.jax.normalization import ObservationNormState
from src.jax.ppo_update import gae_returns_and_advantages
from src.jax.rollout.collect import _policy_turn_batch
from src.jax.rollout.phase_timing import ROLLOUT_PHASE_TIMING_KEYS
from src.jax.rollout.types import JaxTrainState, JaxTransitionBatch
from src.jax.ship_action import is_continuous_ship_mode
from src.opponents.constants import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    is_noop_jax_training_opponent_mode,
    validate_jax_training_opponent_mode,
)
from src.opponents.jax_actions.builders import (
    build_action_from_factored_batch,
    owned_planet_ships,
)
from src.opponents.jax_actions.sampling import (
    _four_player_step_action,
    _maybe_effective_single_family_id,
    _opponent_count_metrics,
    _sample_opponent_2p_action,
    _single_stage_family_id,
)
from src.opponents.pool import sample_opponent_type_ids_jax
from src.telemetry.metric_registry import rollout_collection_enabled_groups
from src.training.curriculum import StageView, default_stage_view

from .metrics import OPPONENT_SLOT_METRIC_KEYS, rollout_metrics


@dataclass
class _PhaseAccumulator:
    policy: float = 0.0
    opponent: float = 0.0
    env_step: float = 0.0
    reset: float = 0.0
    post_step: float = 0.0

    def as_metric_dict(self) -> dict[str, jnp.ndarray]:
        total = (
            self.policy + self.opponent + self.env_step + self.reset + self.post_step
        )
        total = max(total, 1e-9)
        return {
            "rollout_phase_policy_seconds": jnp.asarray(self.policy, dtype=jnp.float32),
            "rollout_phase_opponent_seconds": jnp.asarray(
                self.opponent, dtype=jnp.float32
            ),
            "rollout_phase_env_step_seconds": jnp.asarray(
                self.env_step, dtype=jnp.float32
            ),
            "rollout_phase_reset_seconds": jnp.asarray(self.reset, dtype=jnp.float32),
            "rollout_phase_post_step_seconds": jnp.asarray(
                self.post_step, dtype=jnp.float32
            ),
            "rollout_phase_measured_total_seconds": jnp.asarray(
                total, dtype=jnp.float32
            ),
            "rollout_phase_policy_fraction": jnp.asarray(
                self.policy / total, dtype=jnp.float32
            ),
            "rollout_phase_opponent_fraction": jnp.asarray(
                self.opponent / total, dtype=jnp.float32
            ),
            "rollout_phase_env_step_fraction": jnp.asarray(
                self.env_step / total, dtype=jnp.float32
            ),
            "rollout_phase_reset_fraction": jnp.asarray(
                self.reset / total, dtype=jnp.float32
            ),
            "rollout_phase_post_step_fraction": jnp.asarray(
                self.post_step / total, dtype=jnp.float32
            ),
        }


def _sync(tree) -> None:
    leaves = jax.tree.leaves(tree)
    if leaves:
        jax.block_until_ready(leaves[0])


def _timed_call(accumulator: _PhaseAccumulator, field_name: str, fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    _sync(out)
    setattr(
        accumulator,
        field_name,
        getattr(accumulator, field_name) + (time.perf_counter() - start),
    )
    return out


def _build_transition(
    *,
    state,
    batch,
    sample,
    result,
    cfg: TrainConfig,
    decoder_hidden,
    include_opponent_metrics: bool,
    include_shield_metrics: bool,
    family_counts: dict[str, jax.Array] | None,
    historical_fallback_slots: jax.Array,
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
        "player_count": jnp.full(
            (batch.planet_features.shape[0],),
            cfg.task.player_count,
            dtype=jnp.int32,
        ),
        "ship_bucket_mask": sample.ship_bucket_mask,
        "target_index": sample.target_index,
        "ship_bucket": sample.ship_bucket,
        "source_index": sample.source_index,
        "target_slot": sample.target_slot,
        "stop_flag": sample.stop_flag,
        "step_mask": sample.step_mask,
        "log_prob": sample.log_prob,
        "initial_planet_ships": owned_planet_ships(state.game),
        "value": sample.value,
        "reward": result.reward,
        "done": result.done,
        "terminal_is_first": result.terminal_is_first,
        "terminal_placement": result.terminal_placement,
        "terminal_survival_time": result.terminal_survival_time,
        "terminal_score_share": result.terminal_score_share,
        "terminal_ship_differential": result.terminal_ship_differential,
    }
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
    if include_shield_metrics:
        transition.update(
            {
                "trajectory_shield_blocked_count": sample.diagnostics.blocked_count,
                "trajectory_shield_blocked_sun_count": sample.diagnostics.blocked_sun_count,
                "trajectory_shield_blocked_bounds_count": sample.diagnostics.blocked_bounds_count,
                "trajectory_shield_blocked_unintended_hit_count": sample.diagnostics.blocked_unintended_hit_count,
                "trajectory_shield_blocked_horizon_count": sample.diagnostics.blocked_horizon_count,
                "trajectory_shield_fallback_noop_count": sample.diagnostics.fallback_noop_count,
                "trajectory_shield_legal_non_noop_count": sample.diagnostics.legal_non_noop_count,
                "trajectory_shield_original_non_noop_count": sample.diagnostics.original_non_noop_count,
            }
        )
    if decoder_hidden is not None:
        transition["decoder_hidden"] = decoder_hidden
    if is_continuous_ship_mode(cfg):
        transition["ship_fraction"] = sample.ship_fraction
    return transition


def collect_rollout_jax_timed(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: TurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None = None,
    stage_view: StageView | None = None,
    historical_params_pool: dict | None = None,
    update: int = 0,
    env_index_offset: int | jax.Array = 0,
    norm_state: ObservationNormState | None = None,
    map_pool: MapPoolConstants | None = None,
):
    del update
    validate_jax_training_opponent_mode(cfg.opponents.mode.opponent)
    skip_opp_batch_refresh = (
        cfg.task.player_count == 2
        and is_noop_jax_training_opponent_mode(cfg.opponents.mode.opponent)
    )

    env_count = turn_batch.planet_features.shape[0]
    env_indices = jnp.arange(env_count, dtype=jnp.int32) + jnp.asarray(
        env_index_offset, dtype=jnp.int32
    )
    active_stage_view = default_stage_view(cfg) if stage_view is None else stage_view
    carry_enabled = decoder_carry_enabled(cfg)
    fresh_decoder_hidden = (
        empty_decoder_hidden(env_count, cfg.model.hidden_size)
        if carry_enabled
        else None
    )
    if carry_enabled:
        initial_decoder_hidden = (
            env_state.decoder_hidden
            if env_state.decoder_hidden is not None
            else fresh_decoder_hidden
        )
        env_state = env_state._replace(decoder_hidden=initial_decoder_hidden)
    else:
        initial_decoder_hidden = None

    collection_groups = rollout_collection_enabled_groups(cfg)
    include_opponent_metrics = "opponent_composition" in collection_groups
    include_shield_metrics = "trajectory_shield_debug" in collection_groups

    params = train_state.params

    @jax.jit
    def policy_phase(key_in, state_in, batch_in, decoder_hidden_in):
        policy_batch = _policy_turn_batch(batch_in, norm_state, cfg)
        key_p, _learner_key, _opp_key, _reset_key = jax.random.split(key_in, 4)
        sample = _sample_shielded_sequence_with_params(
            key_p,
            state_in.game,
            policy_batch,
            params,
            policy,
            cfg,
            deterministic=False,
            decoder_hidden_in=decoder_hidden_in,
        )
        learner_action = build_action_from_factored_batch(
            state_in.game,
            batch_in,
            sample.source_index,
            sample.target_slot,
            sample.ship_bucket,
            sample.stop_flag,
            sample.step_mask,
            cfg,
            ship_fraction=sample.ship_fraction,
        )
        return sample, learner_action

    @jax.jit
    def opponent_phase_2p(
        key_in,
        state_in,
        opp_batch_cache,
        learner_action,
        effective_type_ids,
        single_family,
        effective_single_family_id,
    ):
        opponent_action = _sample_opponent_2p_action(
            key_in,
            state_in.game._replace(
                player=(1 - state_in.learner_player).astype(jnp.int32)
            ),
            opp_batch_cache,
            effective_type_ids=effective_type_ids,
            single_family=single_family,
            effective_single_family_id=effective_single_family_id,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=active_stage_view,
            historical_params_pool=historical_params_pool,
        )
        return opponent_action

    @jax.jit
    def env_step_2p(state_in, learner_action, opponent_action):
        return batched_step(
            state_in, learner_action, opponent_action, cfg.task, cfg.reward
        )

    @jax.jit
    def opp_encode_2p(next_state):
        opp_game = next_state.game._replace(
            player=(1 - next_state.learner_player).astype(jnp.int32)
        )
        return jax.vmap(lambda game: encode_turn(game, cfg.task))(opp_game)

    @jax.jit
    def env_step_4p(state_in, multi_player_action):
        return batched_step_multi_player(
            state_in, multi_player_action, cfg.task, cfg.reward
        )

    phases = _PhaseAccumulator()
    transitions_by_step: list[dict] = []
    state = env_state
    batch = turn_batch
    decoder_hidden = initial_decoder_hidden

    if cfg.task.player_count == 2:
        if skip_opp_batch_refresh:
            opp_batch_cache = batch
        else:
            initial_opp_game = state.game._replace(
                player=(1 - state.learner_player).astype(jnp.int32)
            )
            opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                initial_opp_game
            )
    else:
        opp_batch_cache = batch

    rollout_steps = int(cfg.training.rollout_steps)
    for _step in range(rollout_steps):
        key, subkey = jax.random.split(key)
        sample, learner_action = _timed_call(
            phases, "policy", policy_phase, subkey, state, batch, decoder_hidden
        )

        key, _learner_key, opp_key, reset_key = jax.random.split(key, 4)
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
            (opponent_type_ids == OPPONENT_HISTORICAL)
            & jnp.logical_not(has_historical),
            active_stage_view.fallback_family_id,
            opponent_type_ids,
        )
        family_counts: dict[str, jax.Array] | None = None
        historical_fallback_slots = jnp.array(0.0, dtype=jnp.float32)
        if include_opponent_metrics:
            family_counts = _opponent_count_metrics(
                effective_type_ids, state.learner_player
            )
            historical_fallback_slots = (
                (
                    (
                        (opponent_type_ids == OPPONENT_HISTORICAL)
                        & (effective_type_ids == OPPONENT_LATEST)
                    )
                    & (
                        jnp.arange(cfg.task.player_count, dtype=jnp.int32)[None, :]
                        != state.learner_player[:, None]
                    )
                )
                .astype(jnp.float32)
                .sum()
            )

        if cfg.task.player_count == 2:
            opponent_action = _timed_call(
                phases,
                "opponent",
                opponent_phase_2p,
                opp_key,
                state,
                opp_batch_cache,
                learner_action,
                effective_type_ids,
                single_family,
                effective_single_family_id,
            )
            next_state, result = _timed_call(
                phases,
                "env_step",
                env_step_2p,
                state,
                learner_action,
                opponent_action,
            )
        else:
            post_start = time.perf_counter()
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
            per_player_action = jax.vmap(
                lambda player_id: _four_player_step_action(
                    player_id,
                    opp_key=opp_key,
                    player_games=player_games,
                    player_batches=player_batches,
                    effective_type_ids=effective_type_ids,
                    single_family=single_family,
                    effective_single_family_id=effective_single_family_id,
                    learner_action=learner_action,
                    learner_player=state.learner_player,
                    train_state=train_state,
                    policy=policy,
                    cfg=cfg,
                    opponent_params_by_player=opponent_params_by_player,
                    active_stage_view=active_stage_view,
                    historical_params_pool=historical_params_pool,
                )
            )(player_ids)
            multi_player_action = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, 1), per_player_action
            )
            _sync(multi_player_action)
            phases.opponent += time.perf_counter() - post_start

            next_state, result = _timed_call(
                phases,
                "env_step",
                env_step_4p,
                state,
                multi_player_action,
            )

        post_start = time.perf_counter()
        if carry_enabled:
            next_decoder_hidden = reset_decoder_hidden_on_done(
                sample.decoder_hidden_out,
                result.done,
                fresh_decoder_hidden,
            )
            next_state = next_state._replace(decoder_hidden=next_decoder_hidden)

        done_any = bool(jax.device_get(jnp.any(result.done)))
        if done_any:
            reset_start = time.perf_counter()

            def maybe_reset(new, old):
                cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
                return jnp.where(cond, new, old)

            reset_keys = jax.random.split(reset_key, env_count)
            reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
            if map_pool is not None:
                map_ids = (reset_episode_counts + env_indices) % jnp.asarray(
                    map_pool.pool_size, dtype=jnp.int32
                )
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
                cfg.opponents.mode.alternate_player_sides,
            )
            if carry_enabled:
                reset_states = reset_states._replace(
                    decoder_hidden=fresh_decoder_hidden
                )
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
            next_state, next_batch = merged_state, merged_batch
            _sync(next_state)
            phases.reset += time.perf_counter() - reset_start
        else:
            next_state = next_state
            next_batch = result.batch

        if cfg.task.player_count == 2 and not skip_opp_batch_refresh:
            encode_start = time.perf_counter()
            opp_batch_cache = opp_encode_2p(next_state)
            _sync(opp_batch_cache)
            phases.opponent += time.perf_counter() - encode_start

        if carry_enabled:
            decoder_hidden = next_state.decoder_hidden
        phases.post_step += time.perf_counter() - post_start

        transitions_by_step.append(
            _build_transition(
                state=state,
                batch=batch,
                sample=sample,
                result=result,
                cfg=cfg,
                decoder_hidden=decoder_hidden if carry_enabled else None,
                include_opponent_metrics=include_opponent_metrics,
                include_shield_metrics=include_shield_metrics,
                family_counts=family_counts,
                historical_fallback_slots=historical_fallback_slots,
            )
        )
        state = next_state
        batch = next_batch

    data = {
        key_name: jnp.stack([step[key_name] for step in transitions_by_step], axis=0)
        for key_name in transitions_by_step[0]
    }
    returns_step, advantages_step = gae_returns_and_advantages(
        data["reward"],
        data["value"],
        data["done"],
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
    )
    transition_kwargs = {
        "planet_features": data["planet_features"],
        "planet_mask": data["planet_mask"],
        "edge_features": data["edge_features"],
        "edge_mask": data["edge_mask"],
        "edge_src_ids": data["edge_src_ids"],
        "edge_tgt_ids": data["edge_tgt_ids"],
        "global_features": data["global_features"],
        "theta_ref": data["theta_ref"],
        "player_count": data["player_count"],
        "ship_bucket_mask": data["ship_bucket_mask"],
        "target_index": data["target_index"],
        "ship_bucket": data["ship_bucket"],
        "log_prob": data["log_prob"],
        "returns": returns_step,
        "advantages": advantages_step,
        "source_index": data["source_index"],
        "target_slot": data["target_slot"],
        "stop_flag": data["stop_flag"],
        "step_mask": data["step_mask"],
        "initial_planet_ships": data["initial_planet_ships"],
    }
    if carry_enabled:
        transition_kwargs["decoder_hidden"] = data["decoder_hidden"]
    if is_continuous_ship_mode(cfg):
        transition_kwargs["ship_fraction"] = data["ship_fraction"]
    transitions = JaxTransitionBatch(**transition_kwargs)
    metrics = rollout_metrics(data=data, cfg=cfg, env_count=env_count)
    metrics.update(phases.as_metric_dict())
    return key, state, batch, transitions, metrics


__all__ = ["collect_rollout_jax_timed", "ROLLOUT_PHASE_TIMING_KEYS"]
