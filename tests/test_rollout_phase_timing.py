"""Fast smoke for profile rollout groups (timed collect path)."""

import pytest

import jax
from src.config import TrainConfig
from src.jax.policy import build_jax_policy
from src.jax.rollout.phase_timing import ROLLOUT_PHASE_TIMING_KEYS
from src.jax.train import init_train_state
from src.jax.train.rollout_groups import init_profile_rollout_groups


@pytest.mark.jax
def test_profile_rollout_groups_emit_phase_metrics() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.planet_transformer_layers = 1
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 2
    cfg.opponents.mode.opponent = "random"

    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    _, groups = init_profile_rollout_groups(jax.random.PRNGKey(2), cfg, policy)
    group = groups[0]

    _, _, _, _, metrics = group.collect_fn(
        jax.random.PRNGKey(3),
        group.env_state,
        group.turn_batch,
        train_state,
    )

    for key in ROLLOUT_PHASE_TIMING_KEYS:
        assert key in metrics
        assert float(metrics[key]) >= 0.0


@pytest.mark.jax
def test_profile_rollout_groups_ignore_training_microbatch_for_host_timing() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.player_count = 4
    cfg.task.candidate_count = 3
    cfg.task.max_fleets = 16
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.planet_transformer_layers = 1
    cfg.model.max_moves_k = 1
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.training.rollout_microbatch_envs = 1
    cfg.opponents.mode.opponent = "noop"

    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)
    _, groups = init_profile_rollout_groups(jax.random.PRNGKey(12), cfg, policy)
    group = groups[0]

    _, _, _, transitions, metrics = group.collect_fn(
        jax.random.PRNGKey(13),
        group.env_state,
        group.turn_batch,
        train_state,
    )

    assert transitions.planet_features.shape[1] == cfg.training.num_envs
    assert float(metrics["rollout_phase_opponent_sample_seconds"]) >= 0.0
