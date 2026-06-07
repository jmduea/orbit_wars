from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
OPPONENT_SLOT_COUNT_KEYS: tuple[str, ...] = (
    "opponent_slots_total",
    *tuple(key for key, _ in _OPPONENT_SLOT_COUNT_SPECS),
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


def is_single_family_latest_stage_view(stage_view: StageView) -> bool:
    """Host-side mirror of single-family latest detection used in rollout scan."""

    probs = [float(value) for value in list(stage_view.family_probs)]
    family_ids = [int(value) for value in list(stage_view.family_ids)]
    active_indices = [index for index, prob in enumerate(probs) if prob > 0.0]
    if len(active_indices) != 1:
        return False
    family_id = family_ids[active_indices[0]]
    has_historical = any(bool(value) for value in list(stage_view.snapshot_valid_mask))
    if family_id == OPPONENT_HISTORICAL and not has_historical:
        family_id = int(stage_view.fallback_family_id)
    return family_id == OPPONENT_LATEST


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


def _switch_sample_opponent_family(
    key: jax.Array,
    family_id: jax.Array,
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    stage_view: StageView,
    historical_params_pool: dict | None,
    policy: object,
    pool_row_indices: jax.Array,
    sample_latest: Callable[[], JaxAction],
    random_key: jax.Array,
) -> JaxAction:
    """Dispatch one opponent family via the shared seven-way lax.switch."""

    def historical_branch(_: None) -> JaxAction:
        current_action = sample_latest(None)
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
        return _shielded_random_edge_action(random_key, game, batch, cfg)

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
            sample_latest,
            historical_branch,
            nearest_branch,
            turtle_branch,
            opportunistic_branch,
            random_branch,
            noop_branch,
        ),
        None,
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


def _sample_single_family_action(
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
    *,
    player_id: jax.Array,
    opponent_params_by_player: tuple[dict, ...] | None,
) -> JaxAction:
    player_count = int(cfg.task.player_count)

    def sample_latest(_: None) -> JaxAction:
        if player_count == 4:
            opponent_params = _opponent_params_for_player(
                player_id,
                train_state,
                opponent_params_by_player,
                player_count=player_count,
            )
            action, _decoder_hidden = _sample_policy_action_with_params(
                key,
                game,
                batch,
                opponent_params,
                policy,
                cfg,
                deterministic=cfg.opponents.self_play.deterministic,
            )
        else:
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

    random_key = jax.random.fold_in(key, player_count) if player_count == 4 else key
    return _switch_sample_opponent_family(
        key,
        family_id,
        game,
        batch,
        cfg,
        stage_view=stage_view,
        historical_params_pool=historical_params_pool,
        policy=policy,
        pool_row_indices=pool_row_indices,
        sample_latest=sample_latest,
        random_key=random_key,
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


@dataclass(frozen=True, slots=True)
class _OpponentStepContext:
    """Per-player game/batch view and slot family id for one opponent sample."""

    game: object
    batch: TurnBatch
    sample_key: jax.Array
    resolved_player_id: jax.Array
    slot_type: jax.Array


def _opponent_step_context(
    opp_key: jax.Array,
    *,
    effective_type_ids: jax.Array,
    player_count: int,
    opp_game,
    opp_batch: TurnBatch | None,
    player_id: jax.Array | None,
    player_games,
    player_batches,
) -> _OpponentStepContext:
    if player_count == 2:
        game = opp_game
        batch = opp_batch
        sample_key = opp_key
        resolved_player_id = jnp.asarray(0, dtype=jnp.int32)
        slot_type = jnp.take_along_axis(
            effective_type_ids,
            (1 - game.player).astype(jnp.int32)[:, None],
            axis=1,
        ).squeeze(axis=1)
    else:
        game = jax.tree.map(lambda x: jnp.take(x, player_id, axis=0), player_games)
        batch = jax.tree.map(lambda x: jnp.take(x, player_id, axis=0), player_batches)
        sample_key = jax.random.fold_in(opp_key, player_id)
        resolved_player_id = player_id
        slot_type = effective_type_ids[:, player_id]
    return _OpponentStepContext(
        game=game,
        batch=batch,
        sample_key=sample_key,
        resolved_player_id=resolved_player_id,
        slot_type=slot_type,
    )


def _flatten_player_major(tree: object, player_count: int, env_count: int) -> object:
    """Flatten a ``(player, env, ...)`` pytree into ``(player * env, ...)``."""

    return jax.tree.map(
        lambda value: value.reshape((player_count * env_count,) + value.shape[2:]),
        tree,
    )


def _unflatten_player_major_action(
    action: JaxAction,
    player_count: int,
    env_count: int,
) -> JaxAction:
    """Restore flat player-major actions to ``batched_step_multi_player`` layout."""

    return jax.tree.map(
        lambda value: jnp.moveaxis(
            value.reshape((player_count, env_count) + value.shape[1:]),
            0,
            1,
        ),
        action,
    )


def _flatten_four_player_turn_batches(
    state,
    task: TaskConfig,
    env_count: int,
) -> tuple[object, TurnBatch]:
    """Build 4p player views as one flat player-major batch."""

    player_games, player_batches = _encode_four_player_turn_batches(
        state, task, env_count
    )
    player_count = int(task.player_count)
    return (
        _flatten_player_major(player_games, player_count, env_count),
        _flatten_player_major(player_batches, player_count, env_count),
    )


def _broadcast_learner_action_player_major(
    learner_action: JaxAction,
    player_count: int,
) -> JaxAction:
    """Broadcast ``(env, ...)`` learner actions into flat player-major slots."""

    return jax.tree.map(
        lambda value: jnp.broadcast_to(
            value[None, ...],
            (player_count,) + value.shape,
        ).reshape((player_count * value.shape[0],) + value.shape[1:]),
        learner_action,
    )


def _flat_slot_metadata(
    *,
    effective_type_ids: jax.Array,
    learner_player: jax.Array,
    player_count: int,
    env_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Return flat family ids and learner-slot mask for player-major views."""

    player_ids = jnp.arange(player_count, dtype=jnp.int32)
    family_pe = jnp.swapaxes(effective_type_ids, 0, 1)
    learner_pe = player_ids[:, None] == learner_player[None, :]
    return (
        family_pe.reshape((player_count * env_count,)),
        learner_pe.reshape((player_count * env_count,)),
    )


def _sample_flat_self_play_action(
    key: jax.Array,
    *,
    flat_family: jax.Array,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    flat_game,
    flat_batch: TurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Sample self-play over one flat player-space batch."""

    flat_count = flat_batch.planet_features.shape[0]

    def sample_single_family(
        key_in,
        family_id,
        game_in,
        batch_in,
        pool_row_indices,
    ):
        return _sample_single_family_action(
            key_in,
            family_id,
            game_in,
            batch_in,
            train_state,
            policy,
            cfg,
            stage_view,
            historical_params_pool,
            pool_row_indices,
            player_id=jnp.asarray(0, dtype=jnp.int32),
            opponent_params_by_player=None,
        )

    def single_family_branch(_: None) -> JaxAction:
        return sample_single_family(
            key,
            effective_single_family_id,
            flat_game,
            flat_batch,
            jnp.arange(flat_count, dtype=jnp.int32),
        )

    def mixed_family_branch(_: None) -> JaxAction:
        return _sample_mixed_by_family_batched(
            slot_type=flat_family,
            game=flat_game,
            batch=flat_batch,
            cfg=cfg,
            base_key=key,
            sample_single_family=sample_single_family,
        )

    return jax.lax.cond(
        single_family,
        single_family_branch,
        mixed_family_branch,
        None,
    )


def _sample_flat_four_player_actions(
    key: jax.Array,
    *,
    flat_game,
    flat_batch: TurnBatch,
    learner_action: JaxAction,
    learner_player: jax.Array,
    effective_type_ids: jax.Array,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
) -> JaxAction:
    """Sample all 4p player slots as a single flat opponent batch."""

    player_count = int(cfg.task.player_count)
    env_count = int(learner_player.shape[0])
    flat_family, flat_is_learner = _flat_slot_metadata(
        effective_type_ids=effective_type_ids,
        learner_player=learner_player,
        player_count=player_count,
        env_count=env_count,
    )
    learner_flat = _broadcast_learner_action_player_major(
        learner_action, player_count
    )

    if is_noop_jax_training_opponent_mode(cfg.opponents.mode.opponent):
        flat_action = build_noop_action_from_edge_batch(flat_game, flat_batch, cfg)
    elif cfg.opponents.mode.opponent == "random":
        flat_action = _shielded_random_edge_action(key, flat_game, flat_batch, cfg)
    elif cfg.opponents.mode.opponent == "self":
        flat_action = _sample_flat_self_play_action(
            key,
            flat_family=flat_family,
            single_family=single_family,
            effective_single_family_id=effective_single_family_id,
            flat_game=flat_game,
            flat_batch=flat_batch,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=stage_view,
            historical_params_pool=historical_params_pool,
        )
    else:
        validate_jax_training_opponent_mode(cfg.opponents.mode.opponent)
        raise AssertionError("unreachable")

    flat_action = _select_env_action(flat_is_learner, learner_flat, flat_action)
    return _unflatten_player_major_action(flat_action, player_count, env_count)


def _sample_self_play_opponent_action(
    ctx: _OpponentStepContext,
    *,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
    opponent_params_by_player: tuple[dict, ...] | None,
) -> JaxAction:
    """Sample one opponent action under self-play (single-family or mixed batched)."""

    def single_family_branch(_: None) -> JaxAction:
        env_count = ctx.batch.planet_features.shape[0]
        return _sample_single_family_action(
            ctx.sample_key,
            effective_single_family_id,
            ctx.game,
            ctx.batch,
            train_state,
            policy,
            cfg,
            stage_view,
            historical_params_pool,
            jnp.arange(env_count, dtype=jnp.int32),
            player_id=ctx.resolved_player_id,
            opponent_params_by_player=opponent_params_by_player,
        )

    def mixed_family_branch(_: None) -> JaxAction:
        def sample_single_family(
            key, family_id, reordered_game, reordered_batch, pool_row_indices
        ):
            return _sample_single_family_action(
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
                player_id=ctx.resolved_player_id,
                opponent_params_by_player=opponent_params_by_player,
            )

        return _sample_mixed_by_family_batched(
            slot_type=ctx.slot_type,
            game=ctx.game,
            batch=ctx.batch,
            cfg=cfg,
            base_key=ctx.sample_key,
            sample_single_family=sample_single_family,
        )

    return jax.lax.cond(
        single_family,
        single_family_branch,
        mixed_family_branch,
        None,
    )


def _sample_opponent_player_action(
    opp_key: jax.Array,
    *,
    effective_type_ids: jax.Array,
    single_family: jax.Array,
    effective_single_family_id: jax.Array,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    stage_view: StageView,
    historical_params_pool: dict | None,
    opp_game=None,
    opp_batch: TurnBatch | None = None,
    player_id: jax.Array | None = None,
    player_games=None,
    player_batches=None,
    opponent_params_by_player: tuple[dict, ...] | None = None,
    learner_action: JaxAction | None = None,
    learner_player: jax.Array | None = None,
) -> JaxAction:
    """Shared 2p/4p opponent dispatch; 4p overlays learner actions per env."""

    player_count = int(cfg.task.player_count)
    ctx = _opponent_step_context(
        opp_key,
        effective_type_ids=effective_type_ids,
        player_count=player_count,
        opp_game=opp_game,
        opp_batch=opp_batch,
        player_id=player_id,
        player_games=player_games,
        player_batches=player_batches,
    )

    if is_noop_jax_training_opponent_mode(cfg.opponents.mode.opponent):
        opponent_action = build_noop_action_from_edge_batch(ctx.game, ctx.batch, cfg)
    elif cfg.opponents.mode.opponent == "self":
        opponent_action = _sample_self_play_opponent_action(
            ctx,
            single_family=single_family,
            effective_single_family_id=effective_single_family_id,
            train_state=train_state,
            policy=policy,
            cfg=cfg,
            stage_view=stage_view,
            historical_params_pool=historical_params_pool,
            opponent_params_by_player=opponent_params_by_player,
        )
    elif cfg.opponents.mode.opponent == "random":
        opponent_action = _shielded_random_edge_action(
            ctx.sample_key, ctx.game, ctx.batch, cfg
        )
    else:
        validate_jax_training_opponent_mode(cfg.opponents.mode.opponent)
        raise AssertionError("unreachable")

    if player_count == 4:
        is_learner_player = learner_player == player_id
        return _select_env_action(is_learner_player, learner_action, opponent_action)
    return opponent_action
