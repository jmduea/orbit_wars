from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.jax.decoder_carry import (
    decoder_carry_enabled,
    empty_decoder_hidden,
    reset_decoder_hidden_on_done,
)
from src.jax.ship_action import is_continuous_ship_mode
from src.jax.features import encode_turn
from src.jax.env import (
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
    batched_step_multi_player,
)
from src.jax.features import TurnBatch
from src.jax.ppo_update import gae_returns_and_advantages
from src.jax.rollout.types import JaxTrainState, JaxTransitionBatch
from src.opponents.jax_actions.builders import (
    _sample_shielded_sequence_with_params,
    build_action_from_factored_batch,
    owned_planet_ships,
)
from src.opponents.jax_actions.sampling import (
    _maybe_effective_single_family_id,
    _opponent_count_metrics,
    _single_stage_family_id,
)
from src.opponents.jax_actions.sampling import (
    _four_player_step_action,
    _sample_opponent_2p_action,
)
from src.opponents.pool import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    sample_opponent_type_ids_jax,
)
from src.training.curriculum import StageView, default_stage_view

from src.jax.normalization import ObservationNormState, normalize_turn_batch

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
    if cfg.opponents.mode.opponent not in {"self", "random"}:
        raise ValueError(
            "JAX training supports opponent='self' or opponent='random', "
            f"got {cfg.opponents.mode.opponent!r}."
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

    def scan_step(carry, _):
        key, state, batch, opp_batch_cache, decoder_hidden = carry
        policy_batch = _policy_turn_batch(batch, norm_state, cfg)
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
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
                sample.decoder_hidden_out,
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
            **{
                key: (
                    historical_fallback_slots
                    if key == "opponent_historical_fallback_latest_slots"
                    else family_counts[key]
                )
                for key in OPPONENT_SLOT_METRIC_KEYS
            },
            "trajectory_shield_blocked_count": sample.diagnostics.blocked_count,
            "trajectory_shield_blocked_sun_count": sample.diagnostics.blocked_sun_count,
            "trajectory_shield_blocked_bounds_count": sample.diagnostics.blocked_bounds_count,
            "trajectory_shield_blocked_unintended_hit_count": sample.diagnostics.blocked_unintended_hit_count,
            "trajectory_shield_blocked_horizon_count": sample.diagnostics.blocked_horizon_count,
            "trajectory_shield_fallback_noop_count": sample.diagnostics.fallback_noop_count,
            "trajectory_shield_legal_non_noop_count": sample.diagnostics.legal_non_noop_count,
            "trajectory_shield_original_non_noop_count": sample.diagnostics.original_non_noop_count,
        }
        if carry_enabled:
            transition["decoder_hidden"] = decoder_hidden
        if is_continuous_ship_mode(cfg):
            transition["ship_fraction"] = sample.ship_fraction
        next_carry_decoder_hidden = next_state.decoder_hidden if carry_enabled else None
        return (
            key,
            next_state,
            next_batch,
            next_opp_batch_cache,
            next_carry_decoder_hidden,
        ), transition

    if cfg.task.player_count == 2:
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
        "returns": returns,
        "advantages": advantages,
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
    return key, env_state, turn_batch, transitions, metrics
