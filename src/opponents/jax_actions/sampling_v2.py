from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.game.trajectory_shield import apply_trajectory_shield_to_turn_batch_v2
from src.jax.env import JaxAction
from src.jax.features_v2 import JaxTurnBatchV2
from src.jax.policy_v2 import edge_action_count
from src.jax.rollout.types import JaxTrainState
from src.opponents.jax_actions.builders_v2 import (
    _sample_policy_action_v2,
    _sample_policy_action_v2_with_params,
    build_noop_action_from_edge_batch,
    build_random_action_from_edge_batch,
)
from src.opponents.jax_actions.sampling import (
    _gather_action_by_env,
    _maybe_effective_single_family_id,
    _opponent_count_metrics,
    _opponent_params_for_player,
    _select_env_action,
    _single_stage_family_id,
    _slice_env_axis,
)
from src.opponents.pool import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_NEAREST_SNIPER,
    OPPONENT_NOOP,
    OPPONENT_OPPORTUNISTIC,
    OPPONENT_RANDOM,
    OPPONENT_TURTLE,
)
from src.training.curriculum import StageView


def _edge_bucket_mask(
    shielded,
    cfg: TrainConfig,
    *,
    env_count: int,
) -> jax.Array:
    edge_count = edge_action_count(cfg.task)
    return shielded.ship_bucket_mask.reshape(
        env_count, edge_count, cfg.task.ship_bucket_count
    )


def _shielded_random_edge_action(
    key: jax.Array,
    game,
    batch: JaxTurnBatchV2,
    cfg: TrainConfig,
) -> JaxAction:
    env_count = batch.planet_features.shape[0]
    shielded = jax.vmap(
        lambda game_row, batch_row: apply_trajectory_shield_to_turn_batch_v2(
            game_row, batch_row, cfg.task
        )
    )(game, batch)
    return build_random_action_from_edge_batch(
        key,
        game,
        shielded.batch,
        cfg,
        _edge_bucket_mask(shielded, cfg, env_count=env_count),
    )


def _sample_historical_action_v2(
    key: jax.Array,
    game,
    batch: JaxTurnBatchV2,
    historical_params_pool: dict | None,
    stage_view: StageView,
    current_action: JaxAction,
    policy: object,
    cfg: TrainConfig,
) -> tuple[JaxAction, jax.Array]:
    env_count = batch.planet_features.shape[0]
    has_snapshot = jnp.any(stage_view.snapshot_valid_mask)
    if historical_params_pool is None:
        return current_action, jnp.zeros((env_count,), dtype=jnp.int32)
    logits = jnp.where(
        stage_view.snapshot_valid_mask,
        jnp.log(jnp.maximum(stage_view.historical_selection_probs, 1e-12)),
        jnp.asarray(-1e9, dtype=jnp.float32),
    )
    selected = jax.random.categorical(key, logits, shape=(env_count,))
    pool_size = stage_view.snapshot_valid_mask.shape[0]
    pool_actions = jax.vmap(
        lambda idx, params: _sample_policy_action_v2_with_params(
            jax.random.fold_in(key, idx),
            game,
            batch,
            params,
            policy,
            cfg,
            deterministic=cfg.opponents.snapshot.deterministic,
        )
    )(jnp.arange(pool_size, dtype=jnp.int32), historical_params_pool)
    historical_action = _gather_action_by_env(pool_actions, selected)
    fallback = jnp.logical_not(has_snapshot)
    action = jax.tree.map(
        lambda hist, cur: jnp.where(fallback, cur, hist),
        historical_action,
        current_action,
    )
    fallback_count = jnp.where(
        fallback,
        jnp.ones((env_count,), dtype=jnp.int32),
        jnp.zeros((env_count,), dtype=jnp.int32),
    )
    return action, fallback_count


def _sample_single_family_2p_action_v2(
    key: jax.Array,
    family_id: jax.Array,
    game,
    batch: JaxTurnBatchV2,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    def latest_branch(_: None) -> JaxAction:
        return _sample_policy_action_v2(
            key,
            game,
            batch,
            train_state,
            policy,
            cfg,
            deterministic=cfg.opponents.self_play.deterministic,
        )

    def historical_branch(_: None) -> JaxAction:
        current_action = latest_branch(None)
        historical_action, _fallback = _sample_historical_action_v2(
            jax.random.fold_in(key, 71),
            game,
            batch,
            historical_params_pool,
            stage_view,
            current_action,
            policy,
            cfg,
        )
        return historical_action

    def random_branch(_: None) -> JaxAction:
        return _shielded_random_edge_action(key, game, batch, cfg)

    def scripted_fallback_branch(_: None) -> JaxAction:
        return _shielded_random_edge_action(jax.random.fold_in(key, 19), game, batch, cfg)

    def noop_branch(_: None) -> JaxAction:
        return build_noop_action_from_edge_batch(game, batch, cfg)

    return jax.lax.switch(
        jnp.clip(family_id, 0, OPPONENT_NOOP),
        (
            latest_branch,
            historical_branch,
            scripted_fallback_branch,
            scripted_fallback_branch,
            scripted_fallback_branch,
            random_branch,
            noop_branch,
        ),
        None,
    )


def _sample_mixed_opponent_2p_action_v2(
    opp_key: jax.Array,
    opp_game,
    opp_batch_cache: JaxTurnBatchV2,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    slot_type: jax.Array,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    env_count = int(slot_type.shape[0])
    env_indices = jnp.arange(env_count, dtype=jnp.int32)
    per_env = jax.vmap(
        lambda env_index: _sample_single_family_2p_action_v2(
            jax.random.fold_in(opp_key, env_index),
            jnp.asarray(slot_type[env_index], dtype=jnp.int32),
            _slice_env_axis(opp_game, env_index),
            _slice_env_axis(opp_batch_cache, env_index),
            train_state,
            policy,
            cfg,
            stage_view,
            historical_params_pool,
        )
    )(env_indices)
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=1), per_env)


def _sample_opponent_2p_action_v2(
    opp_key: jax.Array,
    opp_game,
    opp_batch_cache: JaxTurnBatchV2,
    *,
    effective_type_ids: jax.Array,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    slot_type = jnp.take_along_axis(
        effective_type_ids,
        (1 - opp_game.player).astype(jnp.int32)[:, None],
        axis=1,
    ).squeeze(axis=1)
    if cfg.opponents.mode.opponent == "self":

        def single_opponent_branch(_: None) -> JaxAction:
            return _sample_single_family_2p_action_v2(
                opp_key,
                effective_single_family_id,
                opp_game,
                opp_batch_cache,
                train_state,
                policy,
                cfg,
                stage_view,
                historical_params_pool,
            )

        def mixed_opponent_branch(_: None) -> JaxAction:
            return _sample_mixed_opponent_2p_action_v2(
                opp_key,
                opp_game,
                opp_batch_cache,
                train_state,
                policy,
                cfg,
                slot_type,
                stage_view,
                historical_params_pool,
            )

        return jax.lax.cond(
            single_family,
            single_opponent_branch,
            mixed_opponent_branch,
            None,
        )
    if cfg.opponents.mode.opponent == "random":
        return _shielded_random_edge_action(opp_key, opp_game, opp_batch_cache, cfg)
    raise ValueError(
        "JAX training supports opponent='self' or opponent='random', "
        f"got {cfg.opponents.mode.opponent!r}."
    )


def _sample_single_family_4p_action_v2(
    key: jax.Array,
    family_id: jax.Array,
    player_id: jax.Array,
    game,
    batch: JaxTurnBatchV2,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    opponent_params = _opponent_params_for_player(
        player_id,
        train_state,
        opponent_params_by_player,
        player_count=int(cfg.task.player_count),
    )

    def latest_branch(_: None) -> JaxAction:
        return _sample_policy_action_v2_with_params(
            key,
            game,
            batch,
            opponent_params,
            policy,
            cfg,
            deterministic=cfg.opponents.self_play.deterministic,
        )

    def historical_branch(_: None) -> JaxAction:
        current_action = latest_branch(None)
        historical_action, _fallback = _sample_historical_action_v2(
            jax.random.fold_in(key, 71),
            game,
            batch,
            historical_params_pool,
            stage_view,
            current_action,
            policy,
            cfg,
        )
        return historical_action

    def random_branch(_: None) -> JaxAction:
        return _shielded_random_edge_action(
            jax.random.fold_in(key, cfg.task.player_count), game, batch, cfg
        )

    def scripted_fallback_branch(_: None) -> JaxAction:
        return _shielded_random_edge_action(jax.random.fold_in(key, 29), game, batch, cfg)

    def noop_branch(_: None) -> JaxAction:
        return build_noop_action_from_edge_batch(game, batch, cfg)

    return jax.lax.switch(
        jnp.clip(family_id, 0, OPPONENT_NOOP),
        (
            latest_branch,
            historical_branch,
            scripted_fallback_branch,
            scripted_fallback_branch,
            scripted_fallback_branch,
            random_branch,
            noop_branch,
        ),
        None,
    )


def _sample_mixed_player_4p_action_v2(
    player_key: jax.Array,
    player_id: jax.Array,
    player_game,
    player_batch: JaxTurnBatchV2,
    slot_type: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    env_count = int(slot_type.shape[0])
    env_indices = jnp.arange(env_count, dtype=jnp.int32)
    per_env = jax.vmap(
        lambda env_index: _sample_single_family_4p_action_v2(
            jax.random.fold_in(player_key, env_index),
            jnp.asarray(slot_type[env_index], dtype=jnp.int32),
            player_id,
            _slice_env_axis(player_game, env_index),
            _slice_env_axis(player_batch, env_index),
            train_state,
            policy,
            cfg,
            opponent_params_by_player,
            stage_view,
            historical_params_pool,
        )
    )(env_indices)
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=1), per_env)


def _four_player_step_action_v2(
    player_id: jax.Array,
    *,
    opp_key: jax.Array,
    player_games,
    player_batches,
    effective_type_ids: jax.Array,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    learner_action: JaxAction,
    learner_player: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None,
    active_stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    player_batch = jax.tree.map(
        lambda x: jnp.take(x, player_id, axis=0), player_batches
    )
    player_game = jax.tree.map(lambda x: jnp.take(x, player_id, axis=0), player_games)
    player_key = jax.random.fold_in(opp_key, player_id)
    slot_type = effective_type_ids[:, player_id]
    if cfg.opponents.mode.opponent == "self":

        def single_player_branch(_: None) -> JaxAction:
            return _sample_single_family_4p_action_v2(
                player_key,
                effective_single_family_id,
                player_id,
                player_game,
                player_batch,
                train_state,
                policy,
                cfg,
                opponent_params_by_player,
                active_stage_view,
                historical_params_pool,
            )

        def mixed_player_branch(_: None) -> JaxAction:
            return _sample_mixed_player_4p_action_v2(
                player_key,
                player_id,
                player_game,
                player_batch,
                slot_type,
                train_state,
                policy,
                cfg,
                opponent_params_by_player,
                active_stage_view,
                historical_params_pool,
            )

        opponent_action = jax.lax.cond(
            single_family,
            single_player_branch,
            mixed_player_branch,
            None,
        )
    elif cfg.opponents.mode.opponent == "random":
        opponent_action = _shielded_random_edge_action(
            player_key, player_game, player_batch, cfg
        )
    else:
        raise ValueError(
            "JAX training supports opponent='self' or opponent='random', "
            f"got {cfg.opponents.mode.opponent!r}."
        )
    is_learner_player = learner_player == player_id
    return _select_env_action(is_learner_player, learner_action, opponent_action)

