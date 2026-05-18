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


def test_jax_checkpoint_roundtrip_restores_resume_metadata(tmp_path):
    from src.jax_train import load_jax_checkpoint, save_jax_checkpoint

    cfg = TrainConfig()
    cfg.env.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.ppo.rollout_steps = 3
    cfg.ppo.num_envs = 2
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
        attention_heads=cfg.model.attention_heads,
    )
    train_state = init_train_state(jax.random.PRNGKey(5), policy, cfg)

    save_jax_checkpoint(
        tmp_path,
        "roundtrip",
        7,
        train_state,
        cfg,
        key=jax.random.PRNGKey(9),
        total_env_steps=42,
        completed_episodes=3,
    )
    loaded_state, key, start_update, total_env_steps, completed_episodes = (
        load_jax_checkpoint(
            str(tmp_path / "roundtrip" / "jax_ckpt_000007.pkl"), train_state, cfg
        )
    )

    assert start_update == 8
    assert total_env_steps == 42
    assert completed_episodes == 3
    assert jax.numpy.array_equal(key, jax.random.PRNGKey(9))
    assert loaded_state.opt_state is not None


def test_collect_rollout_jax_supports_four_player_multi_player_step():
    cfg = TrainConfig()
    cfg.env.player_count = 4
    cfg.env.max_planets = 8
    cfg.env.max_fleets = 16
    cfg.env.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.ppo.num_envs = 2
    cfg.ppo.rollout_steps = 1
    cfg.opponent = "random"
    reset_keys = jax.random.split(jax.random.PRNGKey(10), cfg.ppo.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.env)
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
        attention_heads=cfg.model.attention_heads,
    )
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)

    _key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(12), env_state, turn_batch, train_state, policy, cfg
    )

    assert transitions.self_features.shape[:3] == (
        cfg.ppo.rollout_steps,
        cfg.ppo.num_envs,
        cfg.env.max_planets,
    )
    assert transitions.decision_mask.shape == (
        cfg.ppo.rollout_steps,
        cfg.ppo.num_envs,
        cfg.env.max_planets,
    )
    assert (
        float(rollout_metrics["env_steps"]) == cfg.ppo.rollout_steps * cfg.ppo.num_envs
    )


def test_assign_learner_players_uses_env_index_and_episode_count():
    from src.jax_env import assign_learner_players

    cfg = TrainConfig()
    cfg.env.player_count = 4
    cfg.env.max_planets = 8
    cfg.env.max_fleets = 16
    cfg.ppo.num_envs = 5
    reset_keys = jax.random.split(jax.random.PRNGKey(20), cfg.ppo.num_envs)
    env_state, _turn_batch = batched_reset(reset_keys, cfg.env)

    env_indices = jax.numpy.arange(cfg.ppo.num_envs, dtype=jax.numpy.int32)
    episode_counts = jax.numpy.array([0, 0, 1, 2, 3], dtype=jax.numpy.int32)
    env_state, turn_batch = assign_learner_players(
        env_state, env_indices, episode_counts, cfg.env, True
    )

    expected = (env_indices + episode_counts) % cfg.env.player_count
    assert jax.numpy.array_equal(env_state.learner_player, expected)
    assert jax.numpy.array_equal(env_state.episode_count, episode_counts)
    assert turn_batch.self_features.shape[:2] == (cfg.ppo.num_envs, cfg.env.max_planets)


def test_collect_rollout_jax_rotates_learner_after_reset_done():
    from src.jax_env import assign_learner_players

    cfg = TrainConfig()
    cfg.env.player_count = 4
    cfg.env.episode_steps = 2
    cfg.env.max_planets = 8
    cfg.env.max_fleets = 16
    cfg.env.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.ppo.num_envs = 4
    cfg.ppo.rollout_steps = 1
    cfg.opponent = "random"
    reset_keys = jax.random.split(jax.random.PRNGKey(30), cfg.ppo.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.env)
    env_indices = jax.numpy.arange(cfg.ppo.num_envs, dtype=jax.numpy.int32)
    episode_counts = jax.numpy.zeros((cfg.ppo.num_envs,), dtype=jax.numpy.int32)
    env_state, turn_batch = assign_learner_players(
        env_state, env_indices, episode_counts, cfg.env, cfg.alternate_player_sides
    )
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
        attention_heads=cfg.model.attention_heads,
    )
    train_state = init_train_state(jax.random.PRNGKey(31), policy, cfg)

    _key, env_state, _turn_batch, _transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(32), env_state, turn_batch, train_state, policy, cfg
    )

    expected_episode_counts = jax.numpy.ones((cfg.ppo.num_envs,), dtype=jax.numpy.int32)
    expected_players = (env_indices + expected_episode_counts) % cfg.env.player_count
    assert float(rollout_metrics["episode_done"]) == cfg.ppo.num_envs
    assert jax.numpy.array_equal(env_state.episode_count, expected_episode_counts)
    assert jax.numpy.array_equal(env_state.learner_player, expected_players)
