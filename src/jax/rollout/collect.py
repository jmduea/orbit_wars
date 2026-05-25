from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.game.trajectory_shield import apply_trajectory_shield_to_turn_batch
from src.opponents.jax_actions.builders import (
    build_action_from_batch,
    build_random_action_from_batch,
    _sample_shielded_sequence_with_params,
)
from src.opponents.jax_actions.sampling import (
    _four_player_step_action,
    _maybe_effective_single_family_id,
    _opponent_count_metrics,
    _sample_mixed_opponent_2p_action,
    _sample_single_family_2p_action,
    _single_stage_family_id,
)
from src.opponents.pool import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    sample_opponent_type_ids_jax,
)
from src.training.curriculum import StageView, default_stage_view

from ..env import (
    JaxAction,
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
    batched_step_multi_player,
)
from ..features import JaxTurnBatch, encode_turn
from ..ppo_update import discounted_returns
from ..rollout.metrics import _rollout_diagnostics, _rollout_diagnostics_lean
from ..rollout.types import JaxTrainState, JaxTransitionBatch


def collect_rollout_jax(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None = None,
    stage_view: StageView | None = None,
    historical_params_pool: dict | None = None,
    update: int = 0,
    env_index_offset: int | jax.Array = 0,
) -> tuple[
    jax.Array, JaxEnvState, JaxTurnBatch, JaxTransitionBatch, dict[str, jax.Array]
]:
    """Collect one fixed-length rollout entirely in JAX.

    The function is designed to be wrapped in ``jax.jit`` by the training loop.
    It samples learner actions, generates the configured opponent actions,
    advances the vectorized JAX environment, resets completed episodes, and
    returns PPO transitions plus rollout metrics.
    """

    env_indices = jnp.arange(turn_batch.self_features.shape[0], dtype=jnp.int32) + jnp.asarray(
        env_index_offset, dtype=jnp.int32
    )
    active_stage_view = default_stage_view(cfg) if stage_view is None else stage_view

    def scan_step(carry, _):
        key, state, batch, opp_batch_cache = carry
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
        sample = _sample_shielded_sequence_with_params(
            learner_key,
            state.game,
            batch,
            train_state.params,
            policy,
            cfg,
            deterministic=False,
        )
        target = sample.target_index
        bucket = sample.ship_bucket
        log_prob = sample.log_prob
        learner_action = build_action_from_batch(batch, target, bucket, cfg)

        env_count = state.game.step.shape[0]
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
            slot_type = jnp.take_along_axis(
                effective_type_ids,
                (1 - state.learner_player).astype(jnp.int32)[:, None],
                axis=1,
            ).squeeze(axis=1)
            if cfg.opponents.mode.opponent == "self":

                def single_opponent_branch(_: None) -> JaxAction:
                    return _sample_single_family_2p_action(
                        opp_key,
                        effective_single_family_id,
                        opp_game,
                        opp_batch_cache,
                        train_state,
                        policy,
                        cfg,
                        active_stage_view,
                        historical_params_pool,
                    )

                def mixed_opponent_branch(_: None) -> JaxAction:
                    return _sample_mixed_opponent_2p_action(
                        opp_key,
                        opp_game,
                        opp_batch_cache,
                        train_state,
                        policy,
                        cfg,
                        slot_type,
                        active_stage_view,
                        historical_params_pool,
                    )

                opponent_action = jax.lax.cond(
                    single_family,
                    single_opponent_branch,
                    mixed_opponent_branch,
                    None,
                )
            elif cfg.opponents.mode.opponent == "random":
                opp_shielded = jax.vmap(
                    lambda game, turn: apply_trajectory_shield_to_turn_batch(
                        game, turn, cfg.task
                    )
                )(opp_game, opp_batch_cache)
                opponent_action = build_random_action_from_batch(
                    opp_key, opp_shielded.batch, cfg, opp_shielded.ship_bucket_mask
                )
            else:
                raise ValueError(
                    "JAX training supports opponent='self' or opponent='random', "
                    f"got {cfg.opponents.mode.opponent!r}."
                )

            next_state, result = batched_step(
                state, learner_action, opponent_action, cfg.task, cfg.reward
            )
        elif cfg.task.player_count == 4:
            player_actions = []
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
        else:
            raise ValueError(
                "JAX PPO rollout supports env.player_count of 2 or 4, "
                f"got {cfg.task.player_count}."
            )

        def maybe_reset(new, old):
            cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
            return jnp.where(cond, new, old)

        def reset_branch(_):
            reset_keys = jax.random.split(reset_key, batch.self_features.shape[0])
            reset_states, reset_batches = batched_reset(reset_keys, cfg.task)
            reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
            reset_states, reset_batches = assign_learner_players(
                reset_states,
                env_indices,
                reset_episode_counts,
                cfg.task,
                cfg.opponents.mode.alternate_player_sides,
            )
            merged_state = jax.tree.map(maybe_reset, reset_states, next_state)
            merged_batch = jax.tree.map(maybe_reset, reset_batches, result.batch)
            return merged_state, merged_batch

        def no_reset_branch(_):
            return next_state, result.batch

        next_state, next_batch = jax.lax.cond(
            jnp.any(result.done), reset_branch, no_reset_branch, operand=None
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
            "self_features": batch.self_features,
            "candidate_features": batch.candidate_features,
            "global_features": batch.global_features,
            "candidate_mask": batch.candidate_mask,
            "player_count": jnp.full(
                batch.decision_mask.shape, cfg.task.player_count, dtype=jnp.int32
            ),
            "ship_bucket_mask": sample.ship_bucket_mask.reshape(
                batch.decision_mask.shape
                + (
                    target.shape[-1],
                    cfg.task.candidate_count,
                    cfg.task.ship_bucket_count,
                )
            ),
            "decision_mask": jnp.broadcast_to(
                batch.decision_mask[..., None],
                batch.decision_mask.shape + (target.shape[-1],),
            ),
            "target_index": target.reshape(
                batch.decision_mask.shape + (target.shape[-1],)
            ),
            "ship_bucket": bucket.reshape(
                batch.decision_mask.shape + (bucket.shape[-1],)
            ),
            "log_prob": log_prob.reshape(
                batch.decision_mask.shape + (log_prob.shape[-1],)
            ),
            "value": sample.value.reshape(batch.decision_mask.shape),
            "reward": result.reward,
            "done": result.done,
            "terminal_is_first": result.terminal_is_first,
            "terminal_placement": result.terminal_placement,
            "terminal_score_share": result.terminal_score_share,
            "terminal_survival_time": result.terminal_survival_time,
        }
        if not cfg.training.lean_rollout_metrics:
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
                    "trajectory_shield_legal_non_noop_rate": sample.diagnostics.legal_non_noop_rate,
                    "opponent_slots_total": family_counts["opponent_slots_total"],
                    "opponent_slots_latest": family_counts["opponent_slots_latest"],
                    "opponent_slots_historical": family_counts[
                        "opponent_slots_historical"
                    ],
                    "opponent_slots_random": family_counts["opponent_slots_random"],
                    "opponent_slots_noop": family_counts["opponent_slots_noop"],
                    "opponent_slots_nearest_sniper": family_counts[
                        "opponent_slots_nearest_sniper"
                    ],
                    "opponent_slots_turtle": family_counts["opponent_slots_turtle"],
                    "opponent_slots_opportunistic": family_counts[
                        "opponent_slots_opportunistic"
                    ],
                    "opponent_historical_fallback_latest_slots": historical_fallback_slots,
                }
            )
        return (key, next_state, next_batch, next_opp_batch_cache), transition

    if cfg.task.player_count == 2:
        initial_opp_game = env_state.game._replace(
            player=(1 - env_state.learner_player).astype(jnp.int32)
        )
        initial_opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
            initial_opp_game
        )
    else:
        initial_opp_batch_cache = turn_batch

    (key, env_state, turn_batch, _), data = jax.lax.scan(
        scan_step,
        (key, env_state, turn_batch, initial_opp_batch_cache),
        None,
        length=cfg.training.rollout_steps,
    )
    returns_step = discounted_returns(data["reward"], data["done"], cfg.training.gamma)
    returns = jnp.broadcast_to(
        returns_step[..., None, None], data["target_index"].shape
    )
    advantages = returns - data["value"][..., None]
    transitions = JaxTransitionBatch(
        self_features=data["self_features"],
        candidate_features=data["candidate_features"],
        global_features=data["global_features"],
        candidate_mask=data["candidate_mask"],
        player_count=data["player_count"],
        ship_bucket_mask=data["ship_bucket_mask"],
        decision_mask=data["decision_mask"],
        target_index=data["target_index"],
        ship_bucket=data["ship_bucket"],
        log_prob=data["log_prob"],
        returns=returns,
        advantages=advantages,
    )
    opponent_slots = jnp.array(
        cfg.training.rollout_steps
        * turn_batch.self_features.shape[0]
        * max(cfg.task.player_count - 1, 0),
        dtype=jnp.float32,
    )
    mode = (
        cfg.opponents.mode.multi_opponent_mode.strip().lower()
        if cfg.opponents.self_play.enabled
        else "shared_current"
    )
    snapshot_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if (
            cfg.opponents.mode.opponent == "self"
            and mode == "sampled_pool"
            and opponent_params_by_player is not None
        )
        else jnp.array(0.0, dtype=jnp.float32)
    )
    current_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if (
            cfg.opponents.mode.opponent == "self"
            and (
                mode == "shared_current"
                or (mode == "sampled_pool" and opponent_params_by_player is None)
            )
        )
        else (
            jnp.array(
                min(max(cfg.opponents.mix.weights.get("latest", 0.0), 0.0), 1.0),
                dtype=jnp.float32,
            )
            if cfg.opponents.mode.opponent == "self" and mode == "mixed"
            else jnp.array(0.0, dtype=jnp.float32)
        )
    )
    random_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if cfg.opponents.mode.opponent == "random"
        else (
            (1.0 - current_share)
            if cfg.opponents.mode.opponent == "self" and mode == "mixed"
            else jnp.array(0.0, dtype=jnp.float32)
        )
    )
    metrics = (
        _rollout_diagnostics_lean(
            data=data,
            transitions=transitions,
            turn_batch=turn_batch,
            cfg=cfg,
        )
        if cfg.training.lean_rollout_metrics
        else _rollout_diagnostics(
            data=data,
            transitions=transitions,
            turn_batch=turn_batch,
            cfg=cfg,
            opponent_slots=opponent_slots,
            snapshot_share=snapshot_share,
            current_share=current_share,
            random_share=random_share,
        )
    )
    return key, env_state, turn_batch, transitions, metrics
