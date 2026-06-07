from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.artifacts.checkpoint_compat import is_planet_flow_pointer_decoder
from src.config import TrainConfig
from src.jax.action_codec import (
    PlanetFlowPolicyOutput,
    sample_planet_flow_pressure_action,
)
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
from src.jax.planet_flow import (
    compile_planet_flow_action,
    compile_seeded_random_planet_flow_control,
    planet_flow_sampling_target_mask,
)
from src.jax.ppo_update import gae_returns_and_advantages
from src.jax.rollout.collect_kernel import (
    _build_transition,
    _env_step_with_opponents,
    _policy_turn_batch,
    _reset_on_done,
    _sample_opponent_phase_context,
)
from src.jax.rollout.types import (
    FactorizedActionReplay,
    JaxTrainState,
    JaxTransitionBatch,
    PlanetFlowActionReplay,
)
from src.jax.ship_action import is_continuous_ship_mode
from src.opponents.constants import validate_jax_training_opponent_mode
from src.opponents.jax_actions.builders import build_action_from_factored_batch
from src.opponents.jax_actions.sampling import (
    _select_opp_batch_cache_2p,
    should_skip_opponent_batch_refresh_2p,
)
from src.telemetry.metric_registry import rollout_collection_enabled_groups
from src.training.curriculum import StageView, default_stage_view

from .metrics import rollout_metrics


def collect_rollout_jax(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: TurnBatch,
    train_state: JaxTrainState,
    policy: nn.Module,
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
    if cfg.task.player_count not in (2, 4):
        raise ValueError(
            "v2 rollout supports env.player_count of 2 or 4; "
            f"got {cfg.task.player_count}."
        )
    validate_jax_training_opponent_mode(cfg.opponents.mode.opponent)

    env_count = turn_batch.planet_features.shape[0]
    env_indices = jnp.arange(env_count, dtype=jnp.int32) + jnp.asarray(
        env_index_offset, dtype=jnp.int32
    )
    active_stage_view = default_stage_view(cfg) if stage_view is None else stage_view
    # Noop opponents ignore edge features; skip per-step opponent re-encode (mode or stage).
    skip_opp_batch_refresh = should_skip_opponent_batch_refresh_2p(
        cfg, active_stage_view
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

    def scan_step(carry, _):
        key, state, batch, opp_batch_cache, decoder_hidden = carry
        policy_batch = _policy_turn_batch(batch, norm_state, cfg)
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
        planet_flow_target_bucket = None
        planet_flow_target_pressure = None
        planet_flow_target_mask = None
        planet_flow_diagnostics = None
        planet_flow_control_diagnostics = None
        ship_bucket_mask = None
        target_index = None
        ship_bucket = None
        source_index = None
        target_slot = None
        stop_flag = None
        step_mask = None
        ship_fraction = None
        shield_diagnostics = None
        if is_planet_flow_pointer_decoder(cfg.model):
            player_count = jnp.full(
                (env_count,), cfg.task.player_count, dtype=jnp.int32
            )
            output = policy.apply(
                train_state.params,
                policy_batch,
                player_count=player_count,
            )
            if not isinstance(output, PlanetFlowPolicyOutput):
                raise TypeError(
                    "planet_flow_target_heatmap policy must return "
                    "PlanetFlowPolicyOutput."
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
                # pyrefly: ignore [implicit-import]
                jax.random.fold_in(learner_key, 104_729),
                state.game,
                batch,
                cfg,
            )
            learner_action = compile_result.action
            planet_flow_diagnostics = compile_result.diagnostics
            planet_flow_control_diagnostics = control_result.diagnostics
            decoder_hidden_out = decoder_hidden
            log_prob = pressure_action.log_prob
            value = output.value
            ship_fraction = None
            planet_flow_target_bucket = pressure_action.target_bucket
            planet_flow_target_pressure = pressure_action.target_pressure
            planet_flow_target_mask = pressure_action.target_mask
            shield_diagnostics = None
        else:
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
            decoder_hidden_out = sample.decoder_hidden_out
            ship_bucket_mask = sample.ship_bucket_mask
            target_index = sample.target_index
            ship_bucket = sample.ship_bucket
            source_index = sample.source_index
            target_slot = sample.target_slot
            stop_flag = sample.stop_flag
            step_mask = sample.step_mask
            log_prob = sample.log_prob
            value = sample.value
            ship_fraction = sample.ship_fraction
            shield_diagnostics = sample.diagnostics
            planet_flow_control_diagnostics = None

        opponent_ctx = _sample_opponent_phase_context(
            opp_key=opp_key,
            env_count=env_count,
            learner_player=state.learner_player,
            cfg=cfg,
            active_stage_view=active_stage_view,
            include_opponent_metrics=include_opponent_metrics,
        )
        next_state, result = _env_step_with_opponents(
            cfg=cfg,
            state=state,
            learner_action=learner_action,
            opp_key=opp_key,
            opp_batch_cache=opp_batch_cache,
            opponent_ctx=opponent_ctx,
            train_state=train_state,
            policy=policy,
            active_stage_view=active_stage_view,
            historical_params_pool=historical_params_pool,
            opponent_params_by_player=opponent_params_by_player,
            env_count=env_count,
        )

        if carry_enabled:
            next_decoder_hidden = reset_decoder_hidden_on_done(
                decoder_hidden_out,
                result.done,
                fresh_decoder_hidden,
            )
            next_state = next_state._replace(decoder_hidden=next_decoder_hidden)

        next_state, next_batch = jax.lax.cond(
            jnp.any(result.done),
            lambda _: _reset_on_done(
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
            ),
            lambda _: (next_state, result.batch),
            operand=None,
        )
        if cfg.task.player_count == 2:
            next_opp_batch_cache = _select_opp_batch_cache_2p(
                skip_refresh=skip_opp_batch_refresh,
                cached=opp_batch_cache,
                env_state=next_state,
                task=cfg.task,
            )
        else:
            next_opp_batch_cache = opp_batch_cache

        transition = _build_transition(
            state=state,
            batch=batch,
            result=result,
            cfg=cfg,
            env_count=env_count,
            value=value,
            log_prob=log_prob,
            include_opponent_metrics=include_opponent_metrics,
            include_shield_metrics=include_shield_metrics,
            family_counts=opponent_ctx.family_counts,
            historical_fallback_slots=opponent_ctx.historical_fallback_slots,
            decoder_hidden=decoder_hidden if carry_enabled else None,
            ship_bucket_mask=ship_bucket_mask,
            target_index=target_index,
            ship_bucket=ship_bucket,
            source_index=source_index,
            target_slot=target_slot,
            stop_flag=stop_flag,
            step_mask=step_mask,
            ship_fraction=ship_fraction,
            shield_diagnostics=shield_diagnostics,
            planet_flow_target_bucket=planet_flow_target_bucket,
            planet_flow_target_pressure=planet_flow_target_pressure,
            planet_flow_target_mask=planet_flow_target_mask,
            planet_flow_diagnostics=planet_flow_diagnostics,
            planet_flow_control_diagnostics=planet_flow_control_diagnostics,
        )
        next_carry_decoder_hidden = next_state.decoder_hidden if carry_enabled else None
        return (
            key,
            next_state,
            next_batch,
            next_opp_batch_cache,
            next_carry_decoder_hidden,
        ), transition

    if cfg.task.player_count == 2:
        initial_opp_batch_cache = _select_opp_batch_cache_2p(
            skip_refresh=skip_opp_batch_refresh,
            cached=turn_batch,
            env_state=env_state,
            task=cfg.task,
        )
    else:
        initial_opp_batch_cache = turn_batch

    (_, env_state, turn_batch, _, _), data = jax.lax.scan(
        scan_step,
        (key, env_state, turn_batch, initial_opp_batch_cache, initial_decoder_hidden),
        None,
        length=cfg.training.rollout_steps,
    )
    returns_step, advantages_step = gae_returns_and_advantages(
        data["reward"],
        data["value"],
        data["done"],
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
    )
    returns = returns_step
    advantages = advantages_step
    if is_planet_flow_pointer_decoder(cfg.model):
        action_replay = PlanetFlowActionReplay(
            target_bucket=data["planet_flow_target_bucket"],
            target_pressure=data["planet_flow_target_pressure"],
            target_mask=data["planet_flow_target_mask"],
            log_prob=data["log_prob"],
        )
    else:
        replay_kwargs = {
            "ship_bucket_mask": data["ship_bucket_mask"],
            "target_index": data["target_index"],
            "ship_bucket": data["ship_bucket"],
            "log_prob": data["log_prob"],
            "source_index": data["source_index"],
            "target_slot": data["target_slot"],
            "stop_flag": data["stop_flag"],
            "step_mask": data["step_mask"],
        }
        if carry_enabled:
            replay_kwargs["decoder_hidden"] = data["decoder_hidden"]
        if is_continuous_ship_mode(cfg):
            replay_kwargs["ship_fraction"] = data["ship_fraction"]
        action_replay = FactorizedActionReplay(**replay_kwargs)
    transitions = JaxTransitionBatch(
        planet_features=data["planet_features"],
        planet_mask=data["planet_mask"],
        edge_features=data["edge_features"],
        edge_mask=data["edge_mask"],
        edge_src_ids=data["edge_src_ids"],
        edge_tgt_ids=data["edge_tgt_ids"],
        global_features=data["global_features"],
        theta_ref=data["theta_ref"],
        player_count=data["player_count"],
        returns=returns,
        advantages=advantages,
        action_replay=action_replay,
        initial_planet_ships=data["initial_planet_ships"],
    )
    metrics = rollout_metrics(data=data, cfg=cfg, env_count=env_count)
    return key, env_state, turn_batch, transitions, metrics
