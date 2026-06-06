from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.action_sampling import (
    _sample_opponent_policy_action,
    _sample_opponent_policy_action_with_params,
)
from src.jax.env import JaxAction
from src.jax.features import TurnBatch
from src.jax.policy import edge_action_count
from src.jax.rollout.types import JaxTrainState
from src.jax.shield import apply_trajectory_shield_to_turn_batch_v2
from src.opponents.constants import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_NEAREST_SNIPER,
    OPPONENT_NOOP,
    OPPONENT_OPPORTUNISTIC,
    OPPONENT_RANDOM,
    OPPONENT_TURTLE,
    is_noop_jax_training_opponent_mode,
    validate_jax_training_opponent_mode,
)
from src.opponents.jax_actions.builders import (
    build_noop_action_from_edge_batch,
    build_opportunistic_action_from_edge_batch,
    build_random_action_from_edge_batch,
    build_sniper_action_from_edge_batch,
    build_turtle_action_from_edge_batch,
)
from src.training.curriculum import StageView


def _select_env_action(
    condition: jax.Array,
    true_action: JaxAction,
    false_action: JaxAction,
) -> JaxAction:
    """Select between two batched actions independently for each environment."""

    return jax.tree.map(
        lambda true, false: jnp.where(
            condition.reshape((condition.shape[0],) + (1,) * (true.ndim - 1)),
            true,
            false,
        ),
        true_action,
        false_action,
    )


def _slice_env_axis(tree: object, env_index: jax.Array) -> object:
    """Slice leading env axis to a single row (keeps batch dim size 1)."""

    def slice_leaf(value):
        if isinstance(value, jax.Array) and value.ndim > 0:
            return jax.lax.dynamic_index_in_dim(value, env_index, axis=0, keepdims=True)
        return value

    return jax.tree.map(slice_leaf, tree)


def _stack_player_actions(player_actions: tuple[JaxAction, ...]) -> JaxAction:
    """Stack per-player batched actions into batched_step_multi_player layout."""

    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *player_actions)


def _gather_action_by_env(pool_action: JaxAction, indices: jax.Array) -> JaxAction:
    env_indices = jnp.arange(indices.shape[0], dtype=jnp.int32)
    return jax.tree.map(lambda field: field[indices, env_indices], pool_action)


def _opponent_count_metrics(
    effective_type_ids: jax.Array,
    learner_player: jax.Array,
) -> dict[str, jax.Array]:
    player_ids = jnp.arange(effective_type_ids.shape[1], dtype=jnp.int32)
    slot_mask = player_ids[None, :] != learner_player[:, None]
    slot_values = slot_mask.astype(jnp.float32)
    return {
        "opponent_slots_total": slot_values.sum(),
        "opponent_slots_latest": ((effective_type_ids == OPPONENT_LATEST) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_historical": (
            (effective_type_ids == OPPONENT_HISTORICAL) & slot_mask
        )
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_random": ((effective_type_ids == OPPONENT_RANDOM) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_noop": ((effective_type_ids == OPPONENT_NOOP) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_nearest_sniper": (
            (effective_type_ids == OPPONENT_NEAREST_SNIPER) & slot_mask
        )
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_turtle": ((effective_type_ids == OPPONENT_TURTLE) & slot_mask)
        .astype(jnp.float32)
        .sum(),
        "opponent_slots_opportunistic": (
            (effective_type_ids == OPPONENT_OPPORTUNISTIC) & slot_mask
        )
        .astype(jnp.float32)
        .sum(),
    }


OPPONENT_SLOT_COUNT_KEYS: tuple[str, ...] = (
    "opponent_slots_total",
    "opponent_slots_latest",
    "opponent_slots_historical",
    "opponent_slots_random",
    "opponent_slots_noop",
    "opponent_slots_nearest_sniper",
    "opponent_slots_turtle",
    "opponent_slots_opportunistic",
)


def _single_stage_family_id(stage_view: StageView) -> jax.Array:
    """Return the sole configured opponent family id, or -1 for true mixtures."""

    active = stage_view.family_probs > 0.0
    single = active.astype(jnp.int32).sum() == 1
    family_index = jnp.argmax(active.astype(jnp.int32))
    family_id = stage_view.family_ids[family_index]
    return jnp.where(single, family_id, jnp.asarray(-1, dtype=jnp.int32))


def _maybe_effective_single_family_id(
    family_id: jax.Array, stage_view: StageView
) -> jax.Array:
    has_historical = jnp.any(stage_view.snapshot_valid_mask)
    return jnp.where(
        (family_id == OPPONENT_HISTORICAL) & jnp.logical_not(has_historical),
        stage_view.fallback_family_id,
        family_id,
    )


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
    batch: TurnBatch,
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


def _shielded_scripted_edge_action(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    builder,
) -> JaxAction:
    env_count = batch.planet_features.shape[0]
    edge_count = edge_action_count(cfg.task)
    shielded = jax.vmap(
        lambda game_row, batch_row: apply_trajectory_shield_to_turn_batch_v2(
            game_row, batch_row, cfg.task
        )
    )(game, batch)
    bucket_mask = shielded.ship_bucket_mask.reshape(
        env_count, edge_count, cfg.task.ship_bucket_count
    )
    return builder(game, shielded.batch, cfg, bucket_mask)


def _sample_historical_action(
    key: jax.Array,
    game,
    batch: TurnBatch,
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
        lambda idx, params: _sample_opponent_policy_action_with_params(
            jax.random.fold_in(key, idx),
            game,
            batch,
            params,
            policy,
            cfg,
            deterministic=cfg.opponents.snapshot.deterministic,
        )[0]
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


def _sample_single_family_2p_action(
    key: jax.Array,
    family_id: jax.Array,
    game,
    batch: TurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    def latest_branch(_: None) -> JaxAction:
        action, _decoder_hidden = _sample_opponent_policy_action(
            key,
            game,
            batch,
            train_state,
            policy,
            cfg,
            deterministic=cfg.opponents.self_play.deterministic,
        )
        return action

    def historical_branch(_: None) -> JaxAction:
        current_action = latest_branch(None)
        historical_action, _fallback = _sample_historical_action(
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

    def nearest_branch(_: None) -> JaxAction:
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_sniper_action_from_edge_batch
        )

    def turtle_branch(_: None) -> JaxAction:
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_turtle_action_from_edge_batch
        )

    def opportunistic_branch(_: None) -> JaxAction:
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_opportunistic_action_from_edge_batch
        )

    def noop_branch(_: None) -> JaxAction:
        return build_noop_action_from_edge_batch(game, batch, cfg)

    return jax.lax.switch(
        jnp.clip(family_id, 0, OPPONENT_NOOP),
        (
            latest_branch,
            historical_branch,
            nearest_branch,
            turtle_branch,
            opportunistic_branch,
            random_branch,
            noop_branch,
        ),
        None,
    )


def _sample_mixed_opponent_2p_action(
    opp_key: jax.Array,
    opp_game,
    opp_batch_cache: TurnBatch,
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
        lambda env_index: _sample_single_family_2p_action(
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


def _sample_opponent_2p_action(
    opp_key: jax.Array,
    opp_game,
    opp_batch_cache: TurnBatch,
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
    if is_noop_jax_training_opponent_mode(cfg.opponents.mode.opponent):
        return build_noop_action_from_edge_batch(opp_game, opp_batch_cache, cfg)
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
                stage_view,
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
    validate_jax_training_opponent_mode(cfg.opponents.mode.opponent)
    raise AssertionError("unreachable")


def _opponent_params_for_player(
    player_id: jax.Array,
    train_state: JaxTrainState,
    opponent_params_by_player: tuple[dict, ...] | None,
    *,
    player_count: int,
) -> dict:
    if opponent_params_by_player is None:
        return train_state.params
    return jax.lax.switch(
        jnp.asarray(player_id, dtype=jnp.int32),
        tuple(opponent_params_by_player[index] for index in range(player_count)),
    )


def _sample_single_family_4p_action(
    key: jax.Array,
    family_id: jax.Array,
    player_id: jax.Array,
    game,
    batch: TurnBatch,
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
        action, _decoder_hidden = _sample_opponent_policy_action_with_params(
            key,
            game,
            batch,
            opponent_params,
            policy,
            cfg,
            deterministic=cfg.opponents.self_play.deterministic,
        )
        return action

    def historical_branch(_: None) -> JaxAction:
        current_action = latest_branch(None)
        historical_action, _fallback = _sample_historical_action(
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

    def nearest_branch(_: None) -> JaxAction:
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_sniper_action_from_edge_batch
        )

    def turtle_branch(_: None) -> JaxAction:
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_turtle_action_from_edge_batch
        )

    def opportunistic_branch(_: None) -> JaxAction:
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_opportunistic_action_from_edge_batch
        )

    def noop_branch(_: None) -> JaxAction:
        return build_noop_action_from_edge_batch(game, batch, cfg)

    return jax.lax.switch(
        jnp.clip(family_id, 0, OPPONENT_NOOP),
        (
            latest_branch,
            historical_branch,
            nearest_branch,
            turtle_branch,
            opportunistic_branch,
            random_branch,
            noop_branch,
        ),
        None,
    )


def _sample_mixed_player_4p_action(
    player_key: jax.Array,
    player_id: jax.Array,
    player_game,
    player_batch: TurnBatch,
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
        lambda env_index: _sample_single_family_4p_action(
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


def _four_player_step_action(
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
    if is_noop_jax_training_opponent_mode(cfg.opponents.mode.opponent):
        opponent_action = build_noop_action_from_edge_batch(
            player_game, player_batch, cfg
        )
    elif cfg.opponents.mode.opponent == "self":

        def single_player_branch(_: None) -> JaxAction:
            return _sample_single_family_4p_action(
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
            return _sample_mixed_player_4p_action(
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
        validate_jax_training_opponent_mode(cfg.opponents.mode.opponent)
        raise AssertionError("unreachable")
    is_learner_player = learner_player == player_id
    return _select_env_action(is_learner_player, learner_action, opponent_action)
