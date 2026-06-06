"""Family-batched mixed opponent sampling."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state
from src.opponents.constants import OPPONENT_LATEST, OPPONENT_NOOP, OPPONENT_RANDOM
from src.opponents.jax_actions.sampling import (
    _sample_mixed_opponent_2p_action,
    is_single_family_noop_stage_view,
)
from src.training.curriculum import CurriculumController


def _empty_snapshot_kwargs() -> dict[str, jax.Array]:
    return {
        "snapshot_ids": jnp.zeros((1,), dtype=jnp.int32),
        "snapshot_valid_mask": jnp.zeros((1,), dtype=bool),
        "snapshot_updates": jnp.zeros((1,), dtype=jnp.int32),
    }


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


def test_compress_and_scatter_env_action_round_trip() -> None:
    full = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    partial = jnp.array([10.0, 30.0], dtype=jnp.float32)
    indices = jnp.array([0, 2], dtype=jnp.int32)
    scattered = full.at[indices].set(partial)
    assert scattered.tolist() == [10.0, 2.0, 30.0, 4.0]


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
    assert float(metrics["env_steps"]) == cfg.training.rollout_steps * cfg.training.num_envs
    assert jnp.all(jnp.isfinite(transitions.returns))
