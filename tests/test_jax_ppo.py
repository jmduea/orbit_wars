import jax
import pytest

from src.config import TrainConfig
from src.jax_env import batched_reset
from src.jax_policy import build_jax_policy
from src.jax_ppo import collect_rollout_jax, init_train_state, ppo_update_jax


@pytest.mark.parametrize("architecture", ["mlp", "attention", "transformer"])
def test_end_to_end_jax_rollout_and_update_smoke(architecture: str):
    cfg = TrainConfig()
    cfg.model.architecture = architecture
    cfg.env.max_planets = 8
    cfg.env.max_fleets = 16
    cfg.env.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.ppo.num_envs = 2
    cfg.ppo.rollout_steps = 1
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.ppo.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.env)
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
        attention_heads=cfg.model.attention_heads,
    )
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    _key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(2), env_state, turn_batch, train_state, policy, cfg
    )
    next_train_state, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert transitions.self_features.shape[:3] == (
        cfg.ppo.rollout_steps,
        cfg.ppo.num_envs,
        cfg.env.max_planets,
    )
    assert (
        float(rollout_metrics["env_steps"]) == cfg.ppo.rollout_steps * cfg.ppo.num_envs
    )
    assert "total_loss" in metrics
    assert next_train_state.params is not train_state.params


def test_jax_action_builder_allows_fewer_fleet_slots_than_planets():
    from src.jax_ppo import build_action_from_batch, build_random_action_from_batch

    cfg = TrainConfig()
    cfg.env.max_planets = 8
    cfg.env.max_fleets = 4
    cfg.env.candidate_count = 4
    cfg.ppo.num_envs = 2
    _env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(42), cfg.ppo.num_envs), cfg.env
    )
    target = jax.numpy.zeros(
        (cfg.ppo.num_envs * cfg.env.max_planets,), dtype=jax.numpy.int32
    )
    bucket = jax.numpy.zeros_like(target)

    action = build_action_from_batch(turn_batch, target, bucket, cfg)

    assert action.source_id.shape == (cfg.ppo.num_envs, cfg.env.max_fleets)
    assert action.valid.shape == (cfg.ppo.num_envs, cfg.env.max_fleets)

    random_action = build_random_action_from_batch(
        jax.random.PRNGKey(7), turn_batch, cfg
    )

    assert random_action.source_id.shape == (cfg.ppo.num_envs, cfg.env.max_fleets)
    assert random_action.valid.shape == (cfg.ppo.num_envs, cfg.env.max_fleets)
