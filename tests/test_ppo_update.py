"""CPU-light unit tests for PPO return math, loss wiring, and Hydra hyperparameters."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig, compose_hydra_train_config
from src.game.constants import MAX_PLANETS
from src.jax.policy import build_jax_policy, make_synthetic_turn_batch
from src.jax.ppo_update import (
    discounted_returns,
    gae_returns_and_advantages,
    masked_mean,
    ppo_update_jax,
)
from src.jax.rollout.types import FactorizedActionReplay, JaxTransitionBatch
from src.jax.train import init_train_state


def test_discounted_returns_resets_at_terminal_steps() -> None:
    rewards = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    done = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    returns = discounted_returns(rewards, done, gamma=0.5)

    assert float(returns[2]) == pytest.approx(3.0)
    assert float(returns[1]) == pytest.approx(2.0 + 0.5 * 3.0)
    assert float(returns[0]) == pytest.approx(1.0 + 0.5 * (2.0 + 0.5 * 3.0))


def test_masked_mean_respects_zero_mask_entries() -> None:
    values = jnp.array([[1.0, 2.0, 99.0]], dtype=jnp.float32)
    mask = jnp.array([[1.0, 1.0, 0.0]], dtype=jnp.float32)

    assert float(masked_mean(values, mask)) == pytest.approx(1.5)


def test_masked_mean_ignores_nan_when_mask_zero() -> None:
    values = jnp.array([1.0, jnp.nan, 3.0], dtype=jnp.float32)
    mask = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)

    assert float(masked_mean(values, mask)) == pytest.approx(2.0)


def test_clipped_policy_objective_caps_negative_advantage_large_ratio() -> None:
    from src.jax.ppo_update import _clipped_policy_objective

    advantages = jnp.array([-2.5], dtype=jnp.float32)
    ratio = jnp.array([2980.0], dtype=jnp.float32)
    clipped_ratio = jnp.array([1.2], dtype=jnp.float32)
    legacy_objective = jnp.minimum(advantages * ratio, advantages * clipped_ratio)
    fixed_objective = _clipped_policy_objective(advantages, ratio, clipped_ratio)

    assert float(-legacy_objective[0]) == pytest.approx(7450.0, rel=1e-4)
    assert float(-fixed_objective[0]) == pytest.approx(3.0, rel=1e-4)
    assert float(-fixed_objective[0]) < 100.0


def test_aggregate_ppo_metrics_ignores_empty_minibatches() -> None:
    from src.jax.ppo_update import _aggregate_ppo_metrics

    metrics_by_minibatch = {
        "sample_count": jnp.array([4.0, 0.0], dtype=jnp.float32),
        "total_loss": jnp.array([1.0, jnp.nan], dtype=jnp.float32),
        "policy_loss": jnp.array([0.5, jnp.nan], dtype=jnp.float32),
        "value_loss": jnp.array([0.25, jnp.nan], dtype=jnp.float32),
        "entropy": jnp.array([2.0, jnp.nan], dtype=jnp.float32),
        "approx_kl": jnp.array([0.01, jnp.nan], dtype=jnp.float32),
        "loss_sample_count_2p": jnp.array([4.0, 0.0], dtype=jnp.float32),
        "loss_sample_count_4p": jnp.array([0.0, 0.0], dtype=jnp.float32),
        "total_loss_2p": jnp.array([1.0, jnp.nan], dtype=jnp.float32),
        "policy_loss_2p": jnp.array([0.5, jnp.nan], dtype=jnp.float32),
        "value_loss_2p": jnp.array([0.25, jnp.nan], dtype=jnp.float32),
        "entropy_2p": jnp.array([2.0, jnp.nan], dtype=jnp.float32),
        "approx_kl_2p": jnp.array([0.01, jnp.nan], dtype=jnp.float32),
        "approx_kl_v2_2p": jnp.array([0.02, jnp.nan], dtype=jnp.float32),
        "total_loss_4p": jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
        "policy_loss_4p": jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
        "value_loss_4p": jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
        "entropy_4p": jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
        "approx_kl_4p": jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
        "approx_kl_v2_4p": jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
    }
    metrics = _aggregate_ppo_metrics(metrics_by_minibatch, minibatch_count=2)

    assert float(metrics["total_loss"]) == pytest.approx(1.0)
    assert float(metrics["total_loss_2p"]) == pytest.approx(1.0)
    assert float(metrics["total_loss_4p"]) == pytest.approx(0.0)
    assert jnp.isfinite(jnp.array(list(metrics.values()))).all()


@pytest.mark.parametrize(
    ("overrides", "attr", "expected"),
    [
        (["training.gamma=0.95"], "gamma", 0.95),
        (["training.gae_lambda=0.9"], "gae_lambda", 0.9),
        (["training.gae_lambda=1.0"], "gae_lambda", 1.0),
        (["training.clip_coef=0.1"], "clip_coef", 0.1),
        (["training.ent_coef=0.02"], "ent_coef", 0.02),
        (["training.vf_coef=0.25"], "vf_coef", 0.25),
        (["training.lr=0.001"], "lr", 0.001),
        (["training.max_grad_norm=1.0"], "max_grad_norm", 1.0),
        (["training.update_chunk_rows=256"], "update_chunk_rows", 256),
        (["training.epochs=3"], "epochs", 3),
        (["training.update_chunk_rows=2048"], "update_chunk_rows", 2048),
    ],
)
def test_training_ppo_hyperparameters_compose_from_hydra(
    overrides: list[str], attr: str, expected: float
) -> None:
    cfg = compose_hydra_train_config(overrides)
    assert getattr(cfg.training, attr) == expected


def test_default_training_config_uses_canonical_gae_lambda() -> None:
    cfg = compose_hydra_train_config()
    assert cfg.training.gae_lambda == pytest.approx(0.95)


@pytest.mark.jax
def test_ppo_update_chunk_rows_drives_minibatch_count() -> None:
    import math

    cfg = _small_factorized_cfg()
    cfg.training.update_chunk_rows = 3
    num_envs = 7
    key = jax.random.PRNGKey(31)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    transitions, _output = _build_factorized_on_policy_transitions(
        cfg,
        train_state,
        policy,
        turn_batch,
        rollout_steps=1,
        reward=jnp.zeros((1, num_envs), dtype=jnp.float32),
    )
    _, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(metrics["minibatches"]) == pytest.approx(math.ceil(num_envs / 3))

    cfg.training.update_chunk_rows = 100
    _, metrics_one = ppo_update_jax(train_state, policy, transitions, cfg)
    assert float(metrics_one["minibatches"]) == pytest.approx(1.0)


def test_gae_lambda_one_matches_monte_carlo_path() -> None:
    rewards = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    values = jnp.array([0.5, 0.25, 0.1], dtype=jnp.float32)
    done = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    mc_returns = discounted_returns(rewards, done, gamma=0.99)
    gae_returns, gae_advantages = gae_returns_and_advantages(
        rewards,
        values,
        done,
        gamma=0.99,
        gae_lambda=1.0,
    )

    assert jnp.allclose(gae_returns, mc_returns, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(gae_advantages, mc_returns - values, rtol=1e-5, atol=1e-5)


def test_gae_lambda_below_one_differs_from_monte_carlo() -> None:
    rewards = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    values = jnp.array([0.5, 0.25, 0.1], dtype=jnp.float32)
    done = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    mc_returns = discounted_returns(rewards, done, gamma=0.99)
    gae_returns, _ = gae_returns_and_advantages(
        rewards,
        values,
        done,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert not jnp.allclose(gae_returns, mc_returns, rtol=1e-5, atol=1e-5)


def test_invalid_gae_lambda_rejected_at_compose() -> None:
    with pytest.raises(ValueError, match="gae_lambda"):
        compose_hydra_train_config(["training.gae_lambda=1.5"])


def _factorized_ship_bucket_mask(
    cfg: TrainConfig, *, rollout_steps: int, num_envs: int, sequence_k: int
) -> jax.Array:
    from src.features.registry import edge_k
    
    k_slots = edge_k(cfg.task)
    return jnp.ones(
        (
            rollout_steps,
            num_envs,
            sequence_k,
            MAX_PLANETS,
            k_slots,
            cfg.task.ship_bucket_count,
        ),
        dtype=bool,
    )


def _sparse_factorized_ship_bucket_mask(
    cfg: TrainConfig, *, rollout_steps: int, num_envs: int, sequence_k: int
) -> jax.Array:
    """Per-source bucket legality that differs across planets (rollout-realistic)."""

    from src.features.registry import edge_k
    
    k_slots = edge_k(cfg.task)
    mask = jnp.zeros(
        (
            rollout_steps,
            num_envs,
            sequence_k,
            MAX_PLANETS,
            k_slots,
            cfg.task.ship_bucket_count,
        ),
        dtype=bool,
    )
    # Planet 0: only slot 0 / bucket 0 legal; planet 1: only slot 1 / bucket 1.
    mask = mask.at[..., 0, 0, 0].set(True)
    mask = mask.at[..., 1, 1, 1].set(True)
    return mask


def _factorized_behavior_log_prob(
    cfg: TrainConfig,
    params: dict,
    policy: object,
    turn_batch,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    ship_bucket_mask: jax.Array,
) -> jax.Array:
    from src.jax.factored_sequence_scan import replay_factored_sequence_logprob

    num_envs = int(turn_batch.planet_features.shape[0])
    player_count = jnp.full((num_envs,), cfg.task.player_count, dtype=jnp.int32)
    per_env_bucket_mask = ship_bucket_mask[0]
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        turn_batch,
        cfg,
        player_count=player_count,
        source_index=source_index,
        target_slot=target_slot,
        ship_bucket=ship_bucket,
        stop_flag=stop_flag,
        step_mask=step_mask,
        ship_bucket_mask=per_env_bucket_mask,
    )
    return replay.log_prob


def _factorized_transition_batch(
    cfg: TrainConfig,
    turn_batch,
    *,
    rollout_steps: int,
    target_index: jax.Array,
    ship_bucket: jax.Array,
    log_prob: jax.Array,
    value: jax.Array,
    reward: jax.Array,
) -> JaxTransitionBatch:
    num_envs = int(turn_batch.planet_features.shape[0])
    sequence_k = int(target_index.shape[1])
    done = jnp.zeros((rollout_steps, num_envs), dtype=jnp.float32)
    rewards = reward[None, :] if reward.ndim == 1 else reward
    values = value[None, :] if value.ndim == 1 else value
    if values.shape[0] < rollout_steps:
        values = jnp.broadcast_to(values, (rollout_steps, num_envs))
    if rewards.shape[0] < rollout_steps:
        rewards = jnp.broadcast_to(rewards, (rollout_steps, num_envs))
    returns_step, advantages_step = gae_returns_and_advantages(
        rewards,
        values,
        done,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
    )
    returns = returns_step
    advantages = advantages_step
    ship_bucket_mask = _factorized_ship_bucket_mask(
        cfg, rollout_steps=rollout_steps, num_envs=num_envs, sequence_k=sequence_k
    )
    player_count = jnp.full(
        (rollout_steps, num_envs), cfg.task.player_count, dtype=jnp.int32
    )
    from src.features.registry import edge_k

    k_slots = edge_k(cfg.task)
    flat = target_index[None, ...]
    source_index = flat // k_slots
    target_slot = flat % k_slots

    def expand(field: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            field[None, ...], (rollout_steps, num_envs, *field.shape[1:])
        )

    return JaxTransitionBatch(
        planet_features=expand(turn_batch.planet_features),
        planet_mask=expand(turn_batch.planet_mask),
        edge_features=expand(turn_batch.edge_features),
        edge_mask=expand(turn_batch.edge_mask),
        edge_src_ids=expand(turn_batch.edge_src_ids),
        edge_tgt_ids=expand(turn_batch.edge_tgt_ids),
        global_features=expand(turn_batch.global_features),
        theta_ref=expand(turn_batch.theta_ref),
        player_count=player_count,
        returns=returns,
        advantages=advantages,
        action_replay=FactorizedActionReplay(
            ship_bucket_mask=ship_bucket_mask,
            target_index=flat,
            ship_bucket=ship_bucket[None, ...],
            log_prob=log_prob[None, ...],
            source_index=source_index,
            target_slot=target_slot,
            stop_flag=jnp.zeros((rollout_steps, num_envs, sequence_k), dtype=jnp.int32),
            step_mask=jnp.ones((rollout_steps, num_envs, sequence_k), dtype=jnp.float32),
        ),
    )


def _build_factorized_on_policy_transitions(
    cfg: TrainConfig,
    train_state,
    policy,
    turn_batch,
    *,
    rollout_steps: int,
    reward: jax.Array,
    ship_bucket_mask_fn=_factorized_ship_bucket_mask,
):
    num_envs = int(turn_batch.planet_features.shape[0])
    player_count = jnp.full((num_envs,), cfg.task.player_count, dtype=jnp.int32)
    source_sequence = jnp.zeros((num_envs, cfg.model.max_moves_k), dtype=jnp.int32)
    target_slot_sequence = jnp.zeros((num_envs, cfg.model.max_moves_k), dtype=jnp.int32)
    output = policy.apply(
        train_state.params,
        turn_batch,
        player_count=player_count,
        source_sequence=source_sequence,
        target_slot_sequence=target_slot_sequence,
    )
    ship_bucket = jnp.zeros((num_envs, cfg.model.max_moves_k), dtype=jnp.int32)
    stop_flag = jnp.zeros((num_envs, cfg.model.max_moves_k), dtype=jnp.float32)
    step_mask = jnp.ones((num_envs, cfg.model.max_moves_k), dtype=jnp.float32)
    source_index = output.decoded_source_sequence
    target_slot = output.decoded_target_slot_sequence
    ship_bucket_mask = ship_bucket_mask_fn(
        cfg, rollout_steps=rollout_steps, num_envs=num_envs, sequence_k=cfg.model.max_moves_k
    )
    log_prob = _factorized_behavior_log_prob(
        cfg,
        train_state.params,
        policy,
        turn_batch,
        source_index,
        target_slot,
        ship_bucket,
        stop_flag,
        step_mask,
        ship_bucket_mask,
    )
    transitions = _factorized_transition_batch(
        cfg,
        turn_batch,
        rollout_steps=rollout_steps,
        target_index=jnp.zeros((num_envs, cfg.model.max_moves_k), dtype=jnp.int32),
        ship_bucket=ship_bucket,
        log_prob=log_prob,
        value=output.value,
        reward=reward,
    )
    replay = transitions.action_replay
    assert isinstance(replay, FactorizedActionReplay)
    return transitions._replace(
        action_replay=replay._replace(
            source_index=source_index[None, ...],
            target_slot=target_slot[None, ...],
            stop_flag=stop_flag[None, ...].astype(jnp.int32),
            step_mask=step_mask[None, ...],
            ship_bucket_mask=ship_bucket_mask,
        )
    ), output


def _small_factorized_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.planet_transformer_layers = 1
    cfg.model.max_moves_k = 2
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.training.update_chunk_rows = 8
    return cfg


@pytest.mark.jax
def test_ppo_vf_and_ent_coefs_scale_reported_total_loss() -> None:
    cfg = _small_factorized_cfg()
    cfg.training.vf_coef = 0.0
    cfg.training.ent_coef = 0.0
    num_envs = 2
    key = jax.random.PRNGKey(0)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    transitions, _output = _build_factorized_on_policy_transitions(
        cfg,
        train_state,
        policy,
        turn_batch,
        rollout_steps=1,
        reward=jnp.array([[0.5, -0.25]], dtype=jnp.float32),
    )
    _, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(metrics["total_loss"]) == pytest.approx(
        float(metrics["policy_loss"]), rel=1e-5, abs=1e-5
    )
    assert float(metrics["value_loss"]) > 0.0
    assert float(metrics["entropy"]) > 0.0

@pytest.mark.jax
def test_ppo_update_factorized_path_matches_on_policy_kl() -> None:
    cfg = _small_factorized_cfg()
    cfg.training.debug_replay_parity = True
    num_envs = 2
    key = jax.random.PRNGKey(20)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    transitions, _output = _build_factorized_on_policy_transitions(
        cfg,
        train_state,
        policy,
        turn_batch,
        rollout_steps=1,
        reward=jnp.zeros((1, num_envs), dtype=jnp.float32),
    )
    _, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(metrics["approx_kl"]) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics["approx_kl_first_minibatch"]) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics["parity_logprob_delta_abs_mean"]) == pytest.approx(
        0.0, abs=1e-5
    )
    assert float(metrics["loss_sample_count_2p"]) > 0.0


def _small_planet_flow_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "planet_flow_target_heatmap"
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.planet_transformer_layers = 1
    cfg.model.max_moves_k = 1
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.training.update_chunk_rows = 8
    cfg.opponents.mode.opponent = "random"
    return cfg


@pytest.mark.jax
def test_ppo_update_planet_flow_path_matches_on_policy_kl() -> None:
    from src.jax.env import batched_reset
    from src.jax.rollout.collect import collect_rollout_jax

    cfg = _small_planet_flow_cfg()
    key = jax.random.PRNGKey(24)
    reset_keys = jax.random.split(jax.random.fold_in(key, 0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)

    _key, _env_state, _turn_batch, transitions, _rollout_metrics = collect_rollout_jax(
        jax.random.fold_in(key, 2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
    )
    _, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(metrics["approx_kl"]) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics["value_loss"]) >= 0.0
    assert float(metrics["entropy"]) > 0.0
    assert jnp.isfinite(jnp.array(list(metrics.values()))).all()


@pytest.mark.jax
def test_ppo_update_factorized_source_aware_masks_match_on_policy_kl() -> None:
    """Sparse per-source bucket masks must match rollout masking (not planet-aggregated)."""

    cfg = _small_factorized_cfg()
    num_envs = 2
    key = jax.random.PRNGKey(21)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    transitions, _output = _build_factorized_on_policy_transitions(
        cfg,
        train_state,
        policy,
        turn_batch,
        rollout_steps=1,
        reward=jnp.zeros((1, num_envs), dtype=jnp.float32),
        ship_bucket_mask_fn=_sparse_factorized_ship_bucket_mask,
    )
    _, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(metrics["approx_kl"]) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics["approx_kl_first_minibatch"]) == pytest.approx(0.0, abs=1e-5)
    assert jnp.isfinite(jnp.array(list(metrics.values()))).all()


@pytest.mark.jax
def test_ppo_last_minibatch_kl_exceeds_first_with_high_lr() -> None:
    cfg = _small_factorized_cfg()
    cfg.training.debug_replay_parity = True
    cfg.training.lr = 0.05
    cfg.training.update_chunk_rows = 4
    num_envs = 8
    key = jax.random.PRNGKey(22)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    transitions, _output = _build_factorized_on_policy_transitions(
        cfg,
        train_state,
        policy,
        turn_batch,
        rollout_steps=1,
        reward=jnp.array(
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            dtype=jnp.float32,
        ),
    )
    _, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert float(metrics["approx_kl_first_minibatch"]) == pytest.approx(0.0, abs=1e-4)
    assert float(metrics["parity_logprob_delta_abs_max"]) == pytest.approx(
        0.0, abs=1e-4
    )
    assert float(metrics["minibatches"]) > 1.0
    assert abs(float(metrics["approx_kl_last_minibatch"])) > abs(
        float(metrics["approx_kl_first_minibatch"])
    )
    assert float(metrics["approx_kl_v2"]) >= float(
        metrics["approx_kl_v2_first_minibatch"]
    )


@pytest.mark.jax
def test_ppo_update_changes_params_after_optimizer_step() -> None:
    cfg = _small_factorized_cfg()
    num_envs = 2
    cfg.training.lr = 0.05
    key = jax.random.PRNGKey(30)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    transitions, _output = _build_factorized_on_policy_transitions(
        cfg,
        train_state,
        policy,
        turn_batch,
        rollout_steps=1,
        reward=jnp.array([[1.0, -1.0]], dtype=jnp.float32),
    )
    next_state, _metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    flat_before = jax.tree.leaves(train_state.params)
    flat_after = jax.tree.leaves(next_state.params)
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(flat_before, flat_after, strict=True)
    )

@pytest.mark.jax
@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradient_checkpointing_encoder_init_apply_smoke(checkpointing: bool) -> None:
    cfg = _small_factorized_cfg()
    cfg.model.attention_heads = 2
    cfg.model.planet_transformer_layers = 1
    cfg.training.enable_gradient_checkpointing = checkpointing
    num_envs = 2
    key = jax.random.PRNGKey(40)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.fold_in(key, 1), policy, cfg)
    turn_batch = make_synthetic_turn_batch(
        num_envs, cfg.task, key=jax.random.fold_in(key, 2)
    )
    player_count = jnp.full((num_envs,), cfg.task.player_count, dtype=jnp.int32)
    output = policy.apply(
        train_state.params,
        turn_batch,
        player_count=player_count,
        deterministic=True,
    )
    assert output.value.shape == (num_envs,)
    assert jnp.all(jnp.isfinite(output.value))
