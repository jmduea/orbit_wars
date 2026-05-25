import jax
import pytest

from src.config import TrainConfig
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import ppo_update_jax
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.rollout.types import JaxTransitionBatchV2
from src.jax.train_state import init_train_state


def _v2_smoke_cfg(*, rollout_steps: int) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "gnn_pointer_v2"
    cfg.task.encoding_version = "v2"
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.model.hidden_size = 16
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = rollout_steps
    cfg.training.update_chunk_rows_min = 1
    cfg.training.minibatch_size = 2
    cfg.opponents.mode.opponent = "random"
    return cfg


@pytest.mark.jax
def test_v2_rollout_and_ppo_update_smoke():
    cfg = _v2_smoke_cfg(rollout_steps=1)
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    _key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(2), env_state, turn_batch, train_state, policy, cfg
    )
    assert isinstance(transitions, JaxTransitionBatchV2)
    next_train_state, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(rollout_metrics["env_steps"]) == cfg.training.rollout_steps * cfg.training.num_envs
    assert "total_loss" in metrics
    assert float(metrics["loss_sample_count_2p"]) > 0.0
    assert all(bool(jax.numpy.isfinite(value)) for value in metrics.values())
    assert next_train_state.params is not train_state.params


@pytest.mark.jax
def test_v2_ten_update_training_smoke():
    """Phase 2 exit: 10 collect+ppo cycles with encoding_version=v2."""
    cfg = _v2_smoke_cfg(rollout_steps=2)
    reset_keys = jax.random.split(jax.random.PRNGKey(10), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)
    key = jax.random.PRNGKey(12)

    for _ in range(10):
        key, rollout_key = jax.random.split(key)
        key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
            rollout_key, env_state, turn_batch, train_state, policy, cfg
        )
        assert isinstance(transitions, JaxTransitionBatchV2)
        train_state, metrics = ppo_update_jax(train_state, policy, transitions, cfg)
        assert float(rollout_metrics["env_steps"]) == (
            cfg.training.rollout_steps * cfg.training.num_envs
        )
        assert float(metrics["loss_sample_count_2p"]) > 0.0
        assert all(bool(jax.numpy.isfinite(value)) for value in metrics.values())


@pytest.mark.jax
def test_v2_four_player_random_rollout_smoke():
    cfg = _v2_smoke_cfg(rollout_steps=1)
    cfg.task.player_count = 4
    reset_keys = jax.random.split(jax.random.PRNGKey(20), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(21), policy, cfg)
    _key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(22), env_state, turn_batch, train_state, policy, cfg
    )
    assert isinstance(transitions, JaxTransitionBatchV2)
    next_train_state, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(rollout_metrics["env_steps"]) == cfg.training.rollout_steps * cfg.training.num_envs
    assert float(rollout_metrics["episodes_4p"]) >= 0.0
    assert float(metrics["loss_sample_count_4p"]) > 0.0
    assert float(metrics["loss_sample_count_2p"]) == 0.0
    assert all(bool(jax.numpy.isfinite(value)) for value in metrics.values())
    assert next_train_state.params is not train_state.params
