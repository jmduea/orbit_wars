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
from src.jax.env import JaxEnvState
from src.jax.features import TurnBatch
from src.jax.map_pool.load import MapPoolConstants
from src.jax.normalization import ObservationNormState
from src.jax.ppo_update import gae_returns_and_advantages
from src.jax.rollout.collect_kernel import (
    _build_transition,
    _policy_turn_batch,
    _reset_on_done,
    _sample_opponent_phase_context,
)
from src.jax.rollout.phase_timing import ROLLOUT_PHASE_TIMING_KEYS
from src.jax.rollout.types import JaxTrainState, JaxTransitionBatch
from src.jax.ship_action import is_continuous_ship_mode
from src.opponents.constants import validate_jax_training_opponent_mode
from src.opponents.jax_actions.builders import (
    build_action_from_factored_batch,
)
from src.opponents.jax_actions.sampling import (
    _encode_opponent_turn_batch_2p,
    _encode_four_player_turn_batches,
    _four_player_step_action,
    _initial_opponent_batch_cache_2p,
    should_skip_opponent_batch_refresh_2p,
)
from src.telemetry.metric_registry import rollout_collection_enabled_groups
from src.training.curriculum import StageView, default_stage_view

from .metrics import rollout_metrics


@dataclass
class _PhaseAccumulator:
    policy: float = 0.0
    opponent_sample: float = 0.0
    opponent_encode: float = 0.0
    env_step: float = 0.0
    reset: float = 0.0
    post_step: float = 0.0

    @property
    def opponent(self) -> float:
        return self.opponent_sample + self.opponent_encode

    def as_metric_dict(self) -> dict[str, jnp.ndarray]:
        opponent_total = self.opponent
        total = (
            self.policy + opponent_total + self.env_step + self.reset + self.post_step
        )
        total = max(total, 1e-9)
        return {
            "rollout_phase_policy_seconds": jnp.asarray(self.policy, dtype=jnp.float32),
            "rollout_phase_opponent_seconds": jnp.asarray(
                opponent_total, dtype=jnp.float32
            ),
            "rollout_phase_opponent_sample_seconds": jnp.asarray(
                self.opponent_sample, dtype=jnp.float32
            ),
            "rollout_phase_opponent_encode_seconds": jnp.asarray(
                self.opponent_encode, dtype=jnp.float32
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
                opponent_total / total, dtype=jnp.float32
            ),
            "rollout_phase_opponent_sample_fraction": jnp.asarray(
                self.opponent_sample / total, dtype=jnp.float32
            ),
            "rollout_phase_opponent_encode_fraction": jnp.asarray(
                self.opponent_encode / total, dtype=jnp.float32
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
    active_stage_view = default_stage_view(cfg) if stage_view is None else stage_view
    skip_opp_batch_refresh = should_skip_opponent_batch_refresh_2p(
        cfg, active_stage_view
    )

    env_count = turn_batch.planet_features.shape[0]
    env_indices = jnp.arange(env_count, dtype=jnp.int32) + jnp.asarray(
        env_index_offset, dtype=jnp.int32
    )
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
        from src.opponents.jax_actions.sampling import _sample_opponent_2p_action

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
        from src.jax.env import batched_step

        return batched_step(
            state_in, learner_action, opponent_action, cfg.task, cfg.reward
        )

    @jax.jit
    def opp_encode_2p(next_state):
        return _encode_opponent_turn_batch_2p(
            next_state.game, next_state.learner_player, cfg.task
        )

    @jax.jit
    def env_step_4p(state_in, multi_player_action):
        from src.jax.env import batched_step_multi_player

        return batched_step_multi_player(
            state_in, multi_player_action, cfg.task, cfg.reward
        )

    phases = _PhaseAccumulator()
    transitions_by_step: list[dict] = []
    state = env_state
    batch = turn_batch
    decoder_hidden = initial_decoder_hidden

    if cfg.task.player_count == 2:
        opp_batch_cache = _initial_opponent_batch_cache_2p(
            env_state=state,
            turn_batch=batch,
            task=cfg.task,
            skip_opp_batch_refresh=skip_opp_batch_refresh,
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
        opponent_ctx = _sample_opponent_phase_context(
            opp_key=opp_key,
            env_count=env_count,
            learner_player=state.learner_player,
            cfg=cfg,
            active_stage_view=active_stage_view,
            include_opponent_metrics=include_opponent_metrics,
        )

        if cfg.task.player_count == 2:
            opponent_action = _timed_call(
                phases,
                "opponent_sample",
                opponent_phase_2p,
                opp_key,
                state,
                opp_batch_cache,
                learner_action,
                opponent_ctx.effective_type_ids,
                opponent_ctx.single_family,
                opponent_ctx.effective_single_family_id,
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
            encode_start = time.perf_counter()
            player_ids = jnp.arange(cfg.task.player_count, dtype=jnp.int32)
            player_games, player_batches = _encode_four_player_turn_batches(
                state, cfg.task, env_count
            )
            _sync(player_batches)
            phases.opponent_encode += time.perf_counter() - encode_start

            sample_start = time.perf_counter()
            per_player_action = jax.vmap(
                lambda player_id: _four_player_step_action(
                    player_id,
                    opp_key=opp_key,
                    player_games=player_games,
                    player_batches=player_batches,
                    effective_type_ids=opponent_ctx.effective_type_ids,
                    single_family=opponent_ctx.single_family,
                    effective_single_family_id=opponent_ctx.effective_single_family_id,
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
            phases.opponent_sample += time.perf_counter() - sample_start

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
            next_state, next_batch = _reset_on_done(
                state=state,
                next_state=next_state,
                result=result,
                reset_key=reset_key,
                env_count=env_count,
                env_indices=env_indices,
                map_pool=map_pool,
                cfg=cfg,
                carry_enabled=carry_enabled,
                fresh_decoder_hidden=fresh_decoder_hidden,
            )
            _sync(next_state)
            phases.reset += time.perf_counter() - reset_start
        else:
            next_batch = result.batch

        if cfg.task.player_count == 2 and not bool(
            jax.device_get(skip_opp_batch_refresh)
        ):
            encode_start = time.perf_counter()
            opp_batch_cache = opp_encode_2p(next_state)
            _sync(opp_batch_cache)
            phases.opponent_encode += time.perf_counter() - encode_start

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
                env_count=env_count,
                value=sample.value,
                log_prob=sample.log_prob,
                decoder_hidden=decoder_hidden if carry_enabled else None,
                include_opponent_metrics=include_opponent_metrics,
                include_shield_metrics=include_shield_metrics,
                family_counts=opponent_ctx.family_counts,
                historical_fallback_slots=opponent_ctx.historical_fallback_slots,
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
