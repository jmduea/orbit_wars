from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax
from src.config import TaskConfig, TrainConfig
from src.features.registry import edge_k
from src.jax.action_codec import (
    FactoredPolicyOutput,
    factored_action_log_prob_and_entropy,
)
from src.jax.env import batched_reset
from src.opponents.jax_actions.builders import build_action_from_factored_batch


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def test_build_action_merges_identical_launches() -> None:
    cfg = TrainConfig()
    cfg.task = _task_cfg()
    cfg.model.max_moves_k = 2
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(7), 1), cfg.task)
    game = state.game
    src_row = int(np.asarray(jnp.argmax(batch.planet_mask[0].astype(jnp.int32))))
    slot = 0
    src_id = int(np.asarray(batch.edge_src_ids[0, src_row]))
    bucket = 1

    action = build_action_from_factored_batch(
        game,
        batch,
        source_index=jnp.array([[src_row, src_row]], dtype=jnp.int32),
        target_slot=jnp.array([[slot, slot]], dtype=jnp.int32),
        ship_bucket=jnp.array([[bucket, bucket]], dtype=jnp.int32),
        stop_flag=jnp.zeros((1, 2), dtype=jnp.int32),
        step_mask=jnp.ones((1, 2), dtype=jnp.float32),
        cfg=cfg,
    )
    valid = np.asarray(action.valid[0])
    assert int(valid.sum()) == 1
    assert int(np.asarray(action.source_id[0, 0])) == src_id
    assert float(np.asarray(action.ships[0, 0])) > 0.0


def test_build_action_from_factored_batch_stop_emits_no_valid_launches() -> None:
    cfg = TrainConfig()
    cfg.task = _task_cfg()
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), cfg.task)
    action = build_action_from_factored_batch(
        state.game,
        batch,
        source_index=jnp.zeros((1, 2), dtype=jnp.int32),
        target_slot=jnp.zeros((1, 2), dtype=jnp.int32),
        ship_bucket=jnp.zeros((1, 2), dtype=jnp.int32),
        stop_flag=jnp.ones((1, 2), dtype=jnp.int32),
        step_mask=jnp.ones((1, 2), dtype=jnp.float32),
        cfg=cfg,
    )
    assert not bool(np.asarray(action.valid).any())


def test_factored_action_log_prob_respects_step_mask() -> None:
    cfg = _task_cfg()
    k = edge_k(cfg)
    batch_size = 2
    seq = 3
    buckets = cfg.ship_bucket_count
    output = FactoredPolicyOutput(
        source_logits=jnp.zeros((batch_size, seq, 4)),
        target_logits=jnp.zeros((batch_size, seq, k)),
        stop_logits=jnp.zeros((batch_size, seq)),
        ship_logits=jnp.zeros((batch_size, seq, k, buckets)),
        value=jnp.zeros((batch_size,)),
        decoded_source_sequence=jnp.zeros((batch_size, seq), dtype=jnp.int32),
        decoded_target_slot_sequence=jnp.zeros((batch_size, seq), dtype=jnp.int32),
        decoded_stop_sequence=jnp.zeros((batch_size, seq), dtype=jnp.int32),
    )
    source = jnp.zeros((batch_size, seq), dtype=jnp.int32)
    target_slot = jnp.zeros((batch_size, seq), dtype=jnp.int32)
    ship_bucket = jnp.zeros((batch_size, seq), dtype=jnp.int32)
    stop_flag = jnp.array([[1, 0, 0], [0, 0, 0]], dtype=jnp.int32)
    step_mask = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)

    log_prob, entropy = factored_action_log_prob_and_entropy(
        output,
        source,
        target_slot,
        ship_bucket,
        stop_flag,
        step_mask,
    )
    assert log_prob.shape == (batch_size, seq)
    assert entropy.shape == (batch_size, seq)
    assert np.isfinite(float(np.asarray(log_prob).mean()))
    assert np.isfinite(float(np.asarray(entropy).mean()))
