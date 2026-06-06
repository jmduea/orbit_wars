"""Family-batched mixed opponent sampling."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig
from src.jax.env import JaxAction, batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state
from src.jax.train.snapshots import (
    add_historical_snapshot,
    init_historical_snapshot_pool,
)
from src.opponents.constants import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_NOOP,
    OPPONENT_RANDOM,
)
from src.opponents.jax_actions.sampling import (
    _gather_action_by_env,
    _masked_env_sort_order,
    _merge_reordered_family_action,
    _reorder_env_axis,
    _sample_mixed_opponent_2p_action,
    _single_stage_family_id,
    is_single_family_noop_stage_view,
)
from src.training.curriculum import CurriculumController


def _empty_snapshot_kwargs() -> dict[str, jax.Array]:
    return {
        "snapshot_ids": jnp.zeros((1,), dtype=jnp.int32),
        "snapshot_valid_mask": jnp.zeros((1,), dtype=bool),
        "snapshot_updates": jnp.zeros((1,), dtype=jnp.int32),
    }


def _historical_mix_stage_view(cfg: TrainConfig):
    controller = CurriculumController(
        type(
            "HistoricalMixCurriculum",
            (),
            {
                "enabled": True,
                "stages": [
                    {
                        "id": "hist_mix",
                        "opponent_families": {
                            "historical": 0.5,
                            "random": 0.5,
                        },
                    }
                ],
            },
        )(),
        cfg.opponents.snapshot,
    )
    return controller.stage_view(
        0,
        snapshot_ids=jnp.array([42], dtype=jnp.int32),
        snapshot_valid_mask=jnp.array([True], dtype=bool),
        snapshot_updates=jnp.array([10], dtype=jnp.int32),
    )


def _latest_only_stage_view(cfg: TrainConfig):
    controller = CurriculumController(
        type(
            "LatestOnlyCurriculum",
            (),
            {
                "enabled": True,
                "stages": [
                    {
                        "id": "latest_only",
                        "opponent_families": {"latest": 1.0},
                    }
                ],
            },
        )(),
        cfg.opponents.snapshot,
    )
    return controller.stage_view(0, **_empty_snapshot_kwargs())


def _mixed_stage_view(cfg: TrainConfig):
    controller = CurriculumController(
        type(
            "MixedCurriculum",
            (),
            {
                "enabled": True,
                "stages": [
                    {
                        "id": "bootstrap_mix",
                        "opponent_families": {
                            "latest": 0.3,
                            "random": 0.5,
                            "noop": 0.2,
                        },
                    }
                ],
            },
        )(),
        cfg.opponents.snapshot,
    )
    return controller.stage_view(0, **_empty_snapshot_kwargs())


def test_reorder_merge_helpers_restore_masked_rows() -> None:
    env_count = 4
    values = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    mask = jnp.array([True, False, True, False])
    order = _masked_env_sort_order(mask)
    reordered = _reorder_env_axis(values, order, env_count)
    assert int(mask.sum()) == 2
    assert reordered[0] == 10.0
    assert reordered[1] == 30.0

    partial = jnp.array([111.0, 333.0], dtype=jnp.float32)
    full = jnp.zeros((env_count,), dtype=jnp.float32)
    expanded_partial = jnp.zeros((env_count,), dtype=jnp.float32)
    expanded_partial = expanded_partial.at[jnp.arange(2)].set(partial)
    merged = _merge_reordered_family_action(
        full,
        expanded_partial,
        mask,
        order,
    )
    assert merged.tolist() == [111.0, 0.0, 333.0, 0.0]


def test_gather_action_by_env_uses_batch_row_indices() -> None:
    pool_source = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=jnp.float32,
    )
    pool = JaxAction(
        source_id=jnp.zeros((2, 4), dtype=jnp.int32),
        angle=pool_source,
        ships=jnp.zeros((2, 4), dtype=jnp.int32),
        valid=jnp.ones((2, 4), dtype=jnp.bool_),
    )
    snapshot_indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    row_indices = jnp.array([1, 2, 0, 3], dtype=jnp.int32)
    gathered = _gather_action_by_env(pool, snapshot_indices, row_indices)
    assert gathered.angle.tolist() == [2.0, 7.0, 1.0, 8.0]


@pytest.mark.jax
def test_mixed_opponent_sampling_finite_actions() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.candidate_count = 3
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 4
    cfg.training.rollout_steps = 2
    cfg.opponents.mode.opponent = "self"
    stage_view = _mixed_stage_view(cfg)
    assert not is_single_family_noop_stage_view(stage_view)

    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)

    opp_game = env_state.game._replace(
        player=(1 - env_state.learner_player).astype(jnp.int32)
    )
    slot_type = jnp.array(
        [OPPONENT_LATEST, OPPONENT_RANDOM, OPPONENT_NOOP, OPPONENT_RANDOM],
        dtype=jnp.int32,
    )
    action = _sample_mixed_opponent_2p_action(
        jax.random.PRNGKey(2),
        opp_game,
        turn_batch,
        train_state,
        policy,
        cfg,
        slot_type,
        stage_view,
        None,
    )
    assert action.source_id.shape[0] == cfg.training.num_envs
    assert jnp.all(jnp.isfinite(action.angle))
    assert jnp.all(jnp.isfinite(action.ships.astype(jnp.float32)))


@pytest.mark.jax
def test_collect_rollout_mixed_curriculum_4p_finite() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.player_count = 4
    cfg.task.candidate_count = 3
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 2
    cfg.opponents.mode.opponent = "self"
    stage_view = _mixed_stage_view(cfg)

    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)

    _, _, _, transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=stage_view,
    )
    assert (
        float(metrics["env_steps"])
        == cfg.training.rollout_steps * cfg.training.num_envs
    )
    assert jnp.all(jnp.isfinite(transitions.returns))


@pytest.mark.jax
def test_mixed_historical_family_sampling_finite() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.candidate_count = 3
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 4
    cfg.opponents.mode.opponent = "self"
    stage_view = _historical_mix_stage_view(cfg)

    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    pool = init_historical_snapshot_pool(train_state.params, 1)
    pool, _event = add_historical_snapshot(pool, train_state.params, update=1)

    opp_game = env_state.game._replace(
        player=(1 - env_state.learner_player).astype(jnp.int32)
    )
    slot_type = jnp.array(
        [OPPONENT_HISTORICAL, OPPONENT_RANDOM, OPPONENT_HISTORICAL, OPPONENT_RANDOM],
        dtype=jnp.int32,
    )
    action = _sample_mixed_opponent_2p_action(
        jax.random.PRNGKey(2),
        opp_game,
        turn_batch,
        train_state,
        policy,
        cfg,
        slot_type,
        stage_view,
        pool.params,
    )
    assert action.source_id.shape[0] == cfg.training.num_envs
    assert jnp.all(jnp.isfinite(action.angle))
    assert jnp.all(jnp.isfinite(action.ships.astype(jnp.float32)))


@pytest.mark.jax
def test_mixed_opponent_sampling_is_deterministic_for_fixed_key() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.candidate_count = 3
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 4
    cfg.opponents.mode.opponent = "self"
    stage_view = _mixed_stage_view(cfg)

    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    opp_game = env_state.game._replace(
        player=(1 - env_state.learner_player).astype(jnp.int32)
    )
    slot_type = jnp.array(
        [OPPONENT_LATEST, OPPONENT_RANDOM, OPPONENT_NOOP, OPPONENT_RANDOM],
        dtype=jnp.int32,
    )
    kwargs = dict(
        opp_game=opp_game,
        opp_batch_cache=turn_batch,
        train_state=train_state,
        policy=policy,
        cfg=cfg,
        slot_type=slot_type,
        stage_view=stage_view,
        historical_params_pool=None,
    )
    first = _sample_mixed_opponent_2p_action(jax.random.PRNGKey(99), **kwargs)
    second = _sample_mixed_opponent_2p_action(jax.random.PRNGKey(99), **kwargs)
    assert jnp.all(first.angle == second.angle)
    assert jnp.all(first.ships == second.ships)


def test_single_family_latest_stage_resolves_to_one_family() -> None:
    cfg = TrainConfig()
    cfg.opponents.snapshot.pool_size = 2
    stage_view = _latest_only_stage_view(cfg)
    family_id = int(jax.device_get(_single_stage_family_id(stage_view)))
    assert family_id == OPPONENT_LATEST
    assert not is_single_family_noop_stage_view(stage_view)


@pytest.mark.jax
def test_collect_rollout_mixed_curriculum_finite() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.candidate_count = 3
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 2
    cfg.opponents.mode.opponent = "self"
    stage_view = _mixed_stage_view(cfg)

    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)

    _, _, _, transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=stage_view,
    )
    assert (
        float(metrics["env_steps"])
        == cfg.training.rollout_steps * cfg.training.num_envs
    )
    assert jnp.all(jnp.isfinite(transitions.returns))
