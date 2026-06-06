from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

import jax
from src.config import TaskConfig, TrainConfig
from src.jax.action_sampling import (
    _sample_policy_action,
    _sample_policy_action_with_params,
)
from src.jax.env import JaxAction
from src.jax.features import TurnBatch, encode_turn
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


def _encode_opponent_turn_batch_2p(
    game,
    learner_player: jax.Array,
    task: TaskConfig,
) -> TurnBatch:
    """Encode turn batches from each env's non-learner player perspective (2p)."""

    opp_game = game._replace(player=(1 - learner_player).astype(jnp.int32))
    return jax.vmap(lambda encoded_game: encode_turn(encoded_game, task))(opp_game)


def _select_opp_batch_cache_2p(
    *,
    skip_refresh: jax.Array,
    cached: TurnBatch,
    env_state,
    task: TaskConfig,
) -> TurnBatch:
    """Pick cached or freshly encoded opponent batch (JIT-safe skip predicate)."""

    return jax.lax.cond(
        skip_refresh,
        lambda _: cached,
        lambda _: _encode_opponent_turn_batch_2p(
            env_state.game, env_state.learner_player, task
        ),
        operand=None,
    )


def _encode_four_player_turn_batches(
    state,
    task: TaskConfig,
    env_count: int,
) -> tuple[object, object]:
    """Build per-player games and encoded turn batches for 4p opponent phase."""

    player_ids = jnp.arange(task.player_count, dtype=jnp.int32)
    player_games = jax.vmap(
        lambda player_id: state.game._replace(
            player=jnp.full_like(state.game.step, player_id, dtype=jnp.int32)
        )
    )(player_ids)
    flat_player_games = jax.tree.map(
        lambda x: x.reshape((task.player_count * env_count,) + x.shape[2:]),
        player_games,
    )
    flat_player_batch = jax.vmap(lambda encoded_game: encode_turn(encoded_game, task))(
        flat_player_games
    )
    player_batches = jax.tree.map(
        lambda x: x.reshape((task.player_count, env_count) + x.shape[1:]),
        flat_player_batch,
    )
    return player_games, player_batches


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


def _masked_env_sort_order(mask: jax.Array) -> jax.Array:
    """Sort env axis so masked rows are leading (JIT-safe, no dynamic slice sizes)."""

    sort_key = jnp.where(mask, 0, 1)
    return jnp.argsort(sort_key)


def _reorder_env_axis(tree: object, order: jax.Array, env_count: int) -> object:
    """Reorder leading env axis; static ``env_count`` keeps slice sizes concrete."""

    def reorder_leaf(value):
        if (
            isinstance(value, jax.Array)
            and value.ndim > 0
            and value.shape[0] == env_count
        ):
            return jnp.take(value, order, axis=0)
        return value

    return jax.tree.map(reorder_leaf, tree)


def _merge_reordered_family_action(
    full_action: JaxAction,
    partial_action: JaxAction,
    mask: jax.Array,
    order: jax.Array,
) -> JaxAction:
    """Merge family sampler output back into the original env order."""

    inv_order = jnp.argsort(order)

    def merge_leaf(full, partial):
        if isinstance(full, jax.Array) and full.ndim > 0:
            restored = jnp.take(partial, inv_order, axis=0)
            expanded_mask = mask.reshape((mask.shape[0],) + (1,) * (full.ndim - 1))
            return jnp.where(expanded_mask, restored, full)
        return full

    return jax.tree.map(merge_leaf, full_action, partial_action)


def _gather_action_by_env(
    pool_action: JaxAction,
    snapshot_indices: jax.Array,
    pool_row_indices: jax.Array,
) -> JaxAction:
    """Gather per-env actions from a snapshot×env pool.

    ``pool_action[s, p]`` was sampled with batch row ``p``. For original env ``e``,
    pass ``pool_row_indices[e] = p`` (use ``jnp.argsort(order)`` after reorder).
    """

    return jax.tree.map(
        lambda field: field[snapshot_indices, pool_row_indices],
        pool_action,
    )


_OPPONENT_SLOT_COUNT_SPECS: tuple[tuple[str, int], ...] = (
    ("opponent_slots_latest", OPPONENT_LATEST),
    ("opponent_slots_historical", OPPONENT_HISTORICAL),
    ("opponent_slots_random", OPPONENT_RANDOM),
    ("opponent_slots_noop", OPPONENT_NOOP),
    ("opponent_slots_nearest_sniper", OPPONENT_NEAREST_SNIPER),
    ("opponent_slots_turtle", OPPONENT_TURTLE),
    ("opponent_slots_opportunistic", OPPONENT_OPPORTUNISTIC),
)


def _opponent_count_metrics(
    effective_type_ids: jax.Array,
    learner_player: jax.Array,
) -> dict[str, jax.Array]:
    player_ids = jnp.arange(effective_type_ids.shape[1], dtype=jnp.int32)
    slot_mask = player_ids[None, :] != learner_player[:, None]
    metrics: dict[str, jax.Array] = {
        "opponent_slots_total": slot_mask.astype(jnp.float32).sum(),
    }
    for key, family_id in _OPPONENT_SLOT_COUNT_SPECS:
        metrics[key] = (
            ((effective_type_ids == family_id) & slot_mask).astype(jnp.float32).sum()
        )
    return metrics


def is_single_family_noop_stage_view(stage_view: StageView) -> bool:
    """Host-side mirror of single-family noop detection used in rollout scan."""

    probs = [float(value) for value in list(stage_view.family_probs)]
    family_ids = [int(value) for value in list(stage_view.family_ids)]
    active_indices = [index for index, prob in enumerate(probs) if prob > 0.0]
    if len(active_indices) != 1:
        return False
    family_id = family_ids[active_indices[0]]
    has_historical = any(bool(value) for value in list(stage_view.snapshot_valid_mask))
    if family_id == OPPONENT_HISTORICAL and not has_historical:
        family_id = int(stage_view.fallback_family_id)
    return family_id == OPPONENT_NOOP


def should_skip_opponent_batch_refresh_2p(
    cfg: TrainConfig,
    stage_view: StageView,
) -> jax.Array:
    """Skip 2p opponent re-encode when opponents ignore edge semantics (noop paths)."""

    if cfg.task.player_count != 2:
        return jnp.asarray(False)
    if is_noop_jax_training_opponent_mode(cfg.opponents.mode.opponent):
        return jnp.asarray(True)
    single_family_id = _single_stage_family_id(stage_view)
    effective_id = _maybe_effective_single_family_id(single_family_id, stage_view)
    return (single_family_id >= 0) & (effective_id == OPPONENT_NOOP)


def is_single_family_noop_stage_view(stage_view: StageView) -> bool:
    """Host-side mirror of single-family noop detection used in rollout scan."""

    probs = [float(value) for value in list(stage_view.family_probs)]
    family_ids = [int(value) for value in list(stage_view.family_ids)]
    active_indices = [index for index, prob in enumerate(probs) if prob > 0.0]
    if len(active_indices) != 1:
        return False
    family_id = family_ids[active_indices[0]]
    has_historical = any(bool(value) for value in list(stage_view.snapshot_valid_mask))
    if family_id == OPPONENT_HISTORICAL and not has_historical:
        family_id = int(stage_view.fallback_family_id)
    return family_id == OPPONENT_NOOP


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
    shielded = jax.vmap(
        lambda game_row, batch_row: apply_trajectory_shield_to_turn_batch_v2(
            game_row, batch_row, cfg.task
        )
    )(game, batch)
    return builder(
        game,
        shielded.batch,
        cfg,
        _edge_bucket_mask(shielded, cfg, env_count=env_count),
    )


def _sample_historical_action(
    key: jax.Array,
    game,
    batch: TurnBatch,
    historical_params_pool: dict | None,
    stage_view: StageView,
    current_action: JaxAction,
    policy: object,
    cfg: TrainConfig,
    pool_row_indices: jax.Array,
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
        lambda idx, params: _sample_policy_action_with_params(
            jax.random.fold_in(key, idx),
            game,
            batch,
            params,
            policy,
            cfg,
            deterministic=cfg.opponents.snapshot.deterministic,
        )[0]
    )(jnp.arange(pool_size, dtype=jnp.int32), historical_params_pool)
    historical_action = _gather_action_by_env(pool_actions, selected, pool_row_indices)
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
    pool_row_indices: jax.Array,
) -> JaxAction:
    def latest_branch(_: None) -> JaxAction:
        action, _decoder_hidden = _sample_policy_action(
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
            pool_row_indices,
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


def _sample_mixed_by_family_batched(
    *,
    slot_type: jax.Array,
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    base_key: jax.Array,
    sample_single_family: Callable[
        [jax.Array, jax.Array, object, TurnBatch, jax.Array], JaxAction
    ],
) -> JaxAction:
    env_count = int(slot_type.shape[0])
    merged = build_noop_action_from_edge_batch(game, batch, cfg)
    for family_id in range(OPPONENT_NOOP + 1):
        mask = slot_type == family_id
        if family_id == OPPONENT_NOOP:
            continue

        def sample_branch(_: None) -> JaxAction:
            order = _masked_env_sort_order(mask)
            pool_row_indices = jnp.argsort(order)
            sub_action = sample_single_family(
                jax.random.fold_in(base_key, family_id),
                jnp.asarray(family_id, dtype=jnp.int32),
                _reorder_env_axis(game, order, env_count),
                _reorder_env_axis(batch, order, env_count),
                pool_row_indices,
            )
            return _merge_reordered_family_action(merged, sub_action, mask, order)

        merged = jax.lax.cond(jnp.any(mask), sample_branch, lambda _: merged, None)
    return merged


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
    def sample_single_family(
        key, family_id, reordered_game, reordered_batch, pool_row_indices
    ):
        return _sample_single_family_2p_action(
            key,
            family_id,
            reordered_game,
            reordered_batch,
            train_state,
            policy,
            cfg,
            stage_view,
            historical_params_pool,
            pool_row_indices,
        )

    return _sample_mixed_by_family_batched(
        slot_type=slot_type,
        game=opp_game,
        batch=opp_batch_cache,
        cfg=cfg,
        base_key=opp_key,
        sample_single_family=sample_single_family,
    )


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
            env_count = opp_batch_cache.planet_features.shape[0]
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
                jnp.arange(env_count, dtype=jnp.int32),
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
    pool_row_indices: jax.Array,
) -> JaxAction:
    opponent_params = _opponent_params_for_player(
        player_id,
        train_state,
        opponent_params_by_player,
        player_count=int(cfg.task.player_count),
    )

    def latest_branch(_: None) -> JaxAction:
        action, _decoder_hidden = _sample_policy_action_with_params(
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
            pool_row_indices,
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
    def sample_single_family(
        key, family_id, reordered_game, reordered_batch, pool_row_indices
    ):
        return _sample_single_family_4p_action(
            key,
            family_id,
            player_id,
            reordered_game,
            reordered_batch,
            train_state,
            policy,
            cfg,
            opponent_params_by_player,
            stage_view,
            historical_params_pool,
            pool_row_indices,
        )

    return _sample_mixed_by_family_batched(
        slot_type=slot_type,
        game=player_game,
        batch=player_batch,
        cfg=cfg,
        base_key=player_key,
        sample_single_family=sample_single_family,
    )


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
            env_count = player_batch.planet_features.shape[0]
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
                jnp.arange(env_count, dtype=jnp.int32),
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
