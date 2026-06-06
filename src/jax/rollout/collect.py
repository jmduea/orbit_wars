from __future__ import annotations

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
from src.jax.env import (
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
    batched_step_multi_player,
)
from src.jax.features import TurnBatch, encode_turn
from src.jax.normalization import ObservationNormState, normalize_turn_batch
from src.jax.planet_flow import (
    compile_planet_flow_action,
    compile_seeded_random_planet_flow_control,
    planet_flow_sampling_target_mask,
)
from src.jax.ppo_update import gae_returns_and_advantages
from src.jax.rollout.types import (
    FactorizedActionReplay,
    JaxTrainState,
    JaxTransitionBatch,
    PlanetFlowActionReplay,
)
from src.jax.ship_action import is_continuous_ship_mode
from src.opponents.constants import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
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
    should_skip_opponent_batch_refresh_2p,
)
from src.opponents.pool import sample_opponent_type_ids_jax
from src.telemetry.metric_registry import rollout_collection_enabled_groups
from src.training.curriculum import StageView, default_stage_view

from .metrics import OPPONENT_SLOT_METRIC_KEYS, rollout_metrics


def _policy_turn_batch(
    batch: TurnBatch,
    norm_state: ObservationNormState | None,
    cfg: TrainConfig,
) -> TurnBatch:
    if norm_state is None or not cfg.model.normalize_observations:
        return batch
    return normalize_turn_batch(batch, norm_state, cfg.model)


def collect_rollout_jax(
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
        if is_planet_flow_pointer_decoder(cfg.model):
            player_count = jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32)
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
        family_counts: dict[str, jax.Array] = {}
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
            opp_game = state.game._replace(
                player=(1 - state.learner_player).astype(jnp.int32)
            )
            opponent_action = _sample_opponent_2p_action(
                opp_key,
                opp_game,
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
            next_state, result = batched_step(
                state, learner_action, opponent_action, cfg.task, cfg.reward
            )
        else:
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
            next_state, result = batched_step_multi_player(
                state, multi_player_action, cfg.task, cfg.reward
            )

        if carry_enabled:
            next_decoder_hidden = reset_decoder_hidden_on_done(
                decoder_hidden_out,
                result.done,
                fresh_decoder_hidden,
            )
            next_state = next_state._replace(decoder_hidden=next_decoder_hidden)

        def maybe_reset(new, old):
            cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
            return jnp.where(cond, new, old)

        def reset_branch(_):
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
            return merged_state, merged_batch

        next_state, next_batch = jax.lax.cond(
            jnp.any(result.done),
            reset_branch,
            lambda _: (next_state, result.batch),
            operand=None,
        )
        if cfg.task.player_count == 2:
            if skip_opp_batch_refresh:
                next_opp_batch_cache = opp_batch_cache
            else:
                next_opp_game = next_state.game._replace(
                    player=(1 - next_state.learner_player).astype(jnp.int32)
                )
                next_opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                    next_opp_game
                )
        else:
            next_opp_batch_cache = opp_batch_cache

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
                (env_count,), cfg.task.player_count, dtype=jnp.int32
            ),
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
            if carry_enabled:
                transition["decoder_hidden"] = decoder_hidden
            if is_continuous_ship_mode(cfg):
                transition["ship_fraction"] = ship_fraction
        if include_opponent_metrics:
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
        if include_shield_metrics and shield_diagnostics is not None:
            transition.update(
                {
                    "trajectory_shield_blocked_count": shield_diagnostics.blocked_count,
                    "trajectory_shield_blocked_sun_count": shield_diagnostics.blocked_sun_count,
                    "trajectory_shield_blocked_bounds_count": shield_diagnostics.blocked_bounds_count,
                    "trajectory_shield_blocked_unintended_hit_count": shield_diagnostics.blocked_unintended_hit_count,
                    "trajectory_shield_blocked_horizon_count": shield_diagnostics.blocked_horizon_count,
                    "trajectory_shield_fallback_noop_count": shield_diagnostics.fallback_noop_count,
                    "trajectory_shield_legal_non_noop_count": shield_diagnostics.legal_non_noop_count,
                    "trajectory_shield_original_non_noop_count": shield_diagnostics.original_non_noop_count,
                }
            )
        if is_planet_flow_pointer_decoder(cfg.model):
            pf_diag = planet_flow_diagnostics
            pf_control_diag = planet_flow_control_diagnostics
            assert pf_diag is not None
            assert pf_control_diag is not None
            transition.update(
                {
                    "planet_flow_demanded_mass_sum": pf_diag.demanded_mass,
                    "planet_flow_unreachable_demand_mass_sum": (
                        pf_diag.unreachable_demand_mass
                    ),
                    "planet_flow_held_demand_mass_sum": pf_diag.held_demand_mass,
                    "planet_flow_requested_ship_mass_sum": pf_diag.requested_ship_mass,
                    "planet_flow_emitted_ship_mass_sum": pf_diag.emitted_ship_mass,
                    "planet_flow_capacity_dropped_launch_count": (
                        pf_diag.capacity_dropped_launches
                    ),
                    "planet_flow_emitted_launch_count": pf_diag.emitted_launch_count,
                    "planet_flow_small_launch_count": pf_diag.small_launch_count,
                    "planet_flow_duplicate_source_target_count": (
                        pf_diag.duplicate_source_target_count
                    ),
                    "planet_flow_control_demanded_mass_sum": (
                        pf_control_diag.demanded_mass
                    ),
                    "planet_flow_control_unreachable_demand_mass_sum": (
                        pf_control_diag.unreachable_demand_mass
                    ),
                    "planet_flow_control_held_demand_mass_sum": (
                        pf_control_diag.held_demand_mass
                    ),
                    "planet_flow_control_requested_ship_mass_sum": (
                        pf_control_diag.requested_ship_mass
                    ),
                    "planet_flow_control_emitted_ship_mass_sum": (
                        pf_control_diag.emitted_ship_mass
                    ),
                    "planet_flow_control_capacity_dropped_launch_count": (
                        pf_control_diag.capacity_dropped_launches
                    ),
                    "planet_flow_control_emitted_launch_count": (
                        pf_control_diag.emitted_launch_count
                    ),
                    "planet_flow_control_small_launch_count": (
                        pf_control_diag.small_launch_count
                    ),
                    "planet_flow_control_duplicate_source_target_count": (
                        pf_control_diag.duplicate_source_target_count
                    ),
                }
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
        if skip_opp_batch_refresh:
            # Noop opponents ignore edge features; learner batch has the right shape.
            initial_opp_batch_cache = turn_batch
        else:
            initial_opp_game = env_state.game._replace(
                player=(1 - env_state.learner_player).astype(jnp.int32)
            )
            initial_opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                initial_opp_game
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
