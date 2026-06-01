from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.action_sampling import _sample_shielded_factored_sequence_with_params
from src.jax.env import batched_reset
from src.jax.factored_sequence_scan import (
    owned_planet_ships_from_turn_batch,
    replay_factored_sequence_logprob,
)
from src.jax.policy import build_planet_graph_transformer_policy


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(**kwargs) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 32
    cfg.model.max_moves_k = 2
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.task = _task_cfg(**kwargs.pop("task", {}))
    for key, value in kwargs.pop("model", {}).items():
        setattr(cfg.model, key, value)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.jax
def test_rollout_replay_logprob_parity_distributional_value_head() -> None:
    cfg = _train_cfg(
        task={"trajectory_shield_mode": "cheap"},
        model={"value_head": "distributional", "value_bins": 51},
    )
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(51), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(52), batch)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(53),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
    )
    assert replay.value_logits is not None
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.jax
def test_rollout_replay_logprob_parity_with_stepwise_scan() -> None:
    cfg = _train_cfg(task={"trajectory_shield_mode": "off"})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(1), batch)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(2),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
    )
    delta = replay.log_prob - sample.log_prob
    assert jnp.all(jnp.isfinite(delta))
    assert float(jnp.mean(jnp.abs(delta))) < 1e-4




@pytest.mark.jax
def test_continuous_ship_logprob_depends_on_policy_loc() -> None:
    from src.jax.ship_action import continuous_fraction_log_prob_at_action

    policy_loc = jnp.array([0.0, 1.5], dtype=jnp.float32)
    fraction = jnp.array([0.5, 0.7], dtype=jnp.float32)

    def log_prob_sum(loc: jax.Array) -> jax.Array:
        return continuous_fraction_log_prob_at_action(loc, fraction).sum()

    grad = jax.grad(log_prob_sum)(policy_loc)
    assert jnp.any(jnp.abs(grad) > 1e-6)


@pytest.mark.jax
def test_rollout_replay_logprob_parity_continuous_fraction() -> None:
    cfg = _train_cfg(task={"ship_action_mode": "continuous_fraction"})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(11), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(12), batch)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(13),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=False,
        deterministic_eval=False,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
    )
    active = sample.step_mask > 0.0
    delta = (replay.log_prob - sample.log_prob) * active
    assert jnp.all(jnp.isfinite(delta))
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.jax
def test_rollout_replay_logprob_parity_with_decoder_carry() -> None:
    cfg = _train_cfg(task={"trajectory_shield_mode": "off"})
    cfg.model.decoder_carry = True
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(3), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(4), batch)
    carry_in = jnp.full((1, cfg.model.hidden_size), 0.5, dtype=jnp.float32)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(5),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
        decoder_hidden_in=carry_in,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=jnp.full((1,), cfg.task.player_count, dtype=jnp.int32),
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
        decoder_hidden=carry_in,
    )
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.jax
@pytest.mark.parametrize("decoder_carry", [False, True])
def test_replay_logprob_matches_prefix_forward_per_step(decoder_carry: bool) -> None:
    """PPO replay must score each sub-step with the same prefix forward as sampling."""

    cfg = _train_cfg(
        task={"trajectory_shield_mode": "off"},
        model={"max_moves_k": 3, "decoder_carry": decoder_carry},
    )
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(21), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(22), batch)
    carry_in = (
        jnp.full((1, cfg.model.hidden_size), 0.25, dtype=jnp.float32)
        if decoder_carry
        else None
    )
    player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)
    initial_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(23),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
        decoder_hidden_in=carry_in,
    )
    assert int(jnp.sum(sample.step_mask > 0.0)) >= 2

    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
        decoder_hidden=carry_in,
        initial_remaining_ships=initial_ships,
    )
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )

    from src.jax.factored_sequence_scan import (
        build_shield_prefix_teacher_sequences,
        factored_step_logprob_at_index,
        forward_factored_policy,
        remaining_ships_before_step,
    )

    sequence_k = sample.source_index.shape[1]
    for step_idx in range(sequence_k):
        if not bool(jnp.any(sample.step_mask[:, step_idx] > 0.0)):
            continue
        source_prefix, target_prefix = build_shield_prefix_teacher_sequences(
            sample.source_index, sample.target_slot, step_idx
        )
        prefix_out = forward_factored_policy(
            params,
            policy,
            batch,
            cfg,
            player_count=player_count,
            source_sequence=source_prefix,
            target_slot_sequence=target_prefix,
            decoder_hidden=carry_in,
            deterministic=True,
        )
        ships_before = remaining_ships_before_step(
            cfg,
            initial_remaining_ships=initial_ships,
            source_index=sample.source_index,
            target_slot=sample.target_slot,
            ship_bucket=sample.ship_bucket,
            stop_flag=sample.stop_flag.astype(jnp.float32),
            step_mask=sample.step_mask,
            ship_fraction=sample.ship_fraction,
            step_idx=step_idx,
        )
        lp_prefix = factored_step_logprob_at_index(
            prefix_out,
            cfg,
            step_idx,
            source_index=sample.source_index,
            target_slot=sample.target_slot,
            ship_bucket=sample.ship_bucket,
            stop_flag=sample.stop_flag.astype(jnp.float32),
            ship_bucket_mask=sample.ship_bucket_mask,
            remaining_ships=ships_before,
            ship_fraction=sample.ship_fraction,
            batch=batch,
            step_mask=sample.step_mask,
        )
        np.testing.assert_allclose(
            np.asarray(replay.log_prob[:, step_idx]),
            np.asarray(lp_prefix),
            rtol=1e-5,
            atol=1e-4,
        )


@pytest.mark.jax
def test_zero_teacher_forward_mismatches_prefix_at_step_one() -> None:
    """Document why zeroed replay forward was removed for multi-step sequences."""

    from src.jax.factored_sequence_scan import (
        build_shield_prefix_teacher_sequences,
        factored_step_logprob_at_index,
        forward_factored_policy,
        forward_factored_replay_policy,
        remaining_ships_before_step,
    )

    cfg = _train_cfg(task={"trajectory_shield_mode": "off"}, model={"max_moves_k": 3})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(41), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(42), batch)
    player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(43),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
        deterministic_eval=True,
    )
    step_idx = 1
    if not bool(jnp.any(sample.step_mask[:, step_idx] > 0.0)):
        pytest.skip("fixture did not activate sub-step 1")

    initial_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)
    ships_before = remaining_ships_before_step(
        cfg,
        initial_remaining_ships=initial_ships,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_fraction=sample.ship_fraction,
        step_idx=step_idx,
    )
    source_prefix, target_prefix = build_shield_prefix_teacher_sequences(
        sample.source_index, sample.target_slot, step_idx
    )
    prefix_out = forward_factored_policy(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_sequence=source_prefix,
        target_slot_sequence=target_prefix,
        deterministic=True,
    )
    zero_out = forward_factored_replay_policy(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        sequence_k=sample.source_index.shape[1],
    )
    lp_prefix = factored_step_logprob_at_index(
        prefix_out,
        cfg,
        step_idx,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        ship_bucket_mask=sample.ship_bucket_mask,
        remaining_ships=ships_before,
        ship_fraction=sample.ship_fraction,
    )
    lp_zero = factored_step_logprob_at_index(
        zero_out,
        cfg,
        step_idx,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        ship_bucket_mask=sample.ship_bucket_mask,
        remaining_ships=ships_before,
        ship_fraction=sample.ship_fraction,
    )
    assert float(jnp.max(jnp.abs(lp_prefix - lp_zero))) > 1e-3


@pytest.mark.jax
def test_prefix_replay_logprob_parity_stochastic_multistep() -> None:
    cfg = _train_cfg(
        task={"trajectory_shield_mode": "cheap"},
        model={"max_moves_k": 4, "decoder_carry": True},
    )
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(31), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(32), batch)
    carry_in = jnp.zeros((1, cfg.model.hidden_size), dtype=jnp.float32)
    player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(33),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=False,
        decoder_hidden_in=carry_in,
    )
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
        decoder_hidden=carry_in,
    )
    active = sample.step_mask > 0.0
    delta = (replay.log_prob - sample.log_prob) * active
    assert jnp.all(jnp.isfinite(delta))
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.jax
def test_hygiene_duplicate_prefix_replay_logprob_parity() -> None:
    """Replay recomputes hygiene from prefix; must match sampling when duplicate masked."""
    from src.jax.factored_sequence_scan import (
        build_shield_prefix_teacher_sequences,
        factored_step_logprob_at_index,
        forward_factored_policy,
        owned_planet_ships_from_turn_batch,
        remaining_ships_before_step,
    )

    cfg = _train_cfg(task={"trajectory_shield_mode": "off"}, model={"max_moves_k": 2})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(61), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(62), batch)
    player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(63),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
    )
    if int(jnp.sum(sample.step_mask)) < 2:
        pytest.skip("fixture did not activate two sub-steps")

    initial_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
        initial_remaining_ships=initial_ships,
    )
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )

    step_idx = 1
    ships_before = remaining_ships_before_step(
        cfg,
        initial_remaining_ships=initial_ships,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_fraction=sample.ship_fraction,
        step_idx=step_idx,
    )
    source_prefix, target_prefix = build_shield_prefix_teacher_sequences(
        sample.source_index, sample.target_slot, step_idx
    )
    prefix_out = forward_factored_policy(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_sequence=source_prefix,
        target_slot_sequence=target_prefix,
        deterministic=True,
    )
    lp_with_hygiene = factored_step_logprob_at_index(
        prefix_out,
        cfg,
        step_idx,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        ship_bucket_mask=sample.ship_bucket_mask,
        remaining_ships=ships_before,
        ship_fraction=sample.ship_fraction,
        batch=batch,
        step_mask=sample.step_mask,
    )
    np.testing.assert_allclose(
        np.asarray(lp_with_hygiene),
        np.asarray(replay.log_prob[:, step_idx]),
        rtol=1e-5,
        atol=1e-4,
    )
