import jax

from src.config import TrainConfig
from src.jax_env import batched_reset
from src.jax_policy import build_jax_policy, sample_actions
from src.jax_ppo import collect_rollout_jax, init_train_state, ppo_update_jax


def test_end_to_end_jax_rollout_and_update_smoke():
    cfg = TrainConfig()
    cfg.env.max_planets = 8
    cfg.env.max_fleets = 16
    cfg.env.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.ppo.num_envs = 2
    cfg.ppo.rollout_steps = 1
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.ppo.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.env)
    policy = build_jax_policy(
        architecture=cfg.model.architecture,
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
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


def test_jax_attention_policy_shapes_and_sampling():
    import jax.numpy as jnp

    cfg = TrainConfig()
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    for architecture in ("attention", "transformer"):
        policy = build_jax_policy(
            architecture=architecture,
            candidate_count=4,
            ship_bucket_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
            attention_heads=cfg.model.attention_heads,
        )
        self_features = jnp.zeros((3, 11), dtype=jnp.float32)
        candidate_features = jnp.zeros((3, 4, 14), dtype=jnp.float32)
        global_features = jnp.zeros((3, 8), dtype=jnp.float32)
        candidate_mask = jnp.asarray([[True, True, False, False]] * 3)
        params = policy.init(
            jax.random.PRNGKey(0),
            self_features,
            candidate_features,
            global_features,
            candidate_mask,
        )
        output = policy.apply(
            params, self_features, candidate_features, global_features, candidate_mask
        )
        target_index, ship_bucket, log_prob, entropy = sample_actions(
            jax.random.PRNGKey(1), output
        )

        assert output.target_logits.shape == (3, 4)
        assert output.ship_logits.shape == (3, 4, cfg.env.ship_bucket_count)
        assert output.value.shape == (3,)
        assert target_index.shape == (3,)
        assert ship_bucket.shape == (3,)
        assert log_prob.shape == (3,)
        assert entropy.shape == (3,)
