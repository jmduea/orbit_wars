from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax
from src.artifacts.checkpoint_compat import (
    POINTER_DECODER_FACTORIZED_TOPK,
    pointer_decoder_for_model,
)
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.policy import build_planet_graph_transformer_policy, make_synthetic_turn_batch


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=8, max_fleets=16)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, pointer_decoder: str = "factorized_topk") -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = pointer_decoder
    cfg.model.hidden_size = 64
    cfg.model.max_moves_k = 3
    cfg.model.attention_heads = 4
    cfg.model.planet_transformer_layers = 1
    cfg.task = _task_cfg()
    return cfg


def test_factorized_pointer_decoder_forward_shapes() -> None:
    cfg = _train_cfg(pointer_decoder="factorized_topk")
    assert pointer_decoder_for_model(cfg.model) == POINTER_DECODER_FACTORIZED_TOPK

    policy = build_planet_graph_transformer_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(0))
    params = policy.init(jax.random.PRNGKey(1), batch)
    output = policy.apply(params, batch, deterministic=True)

    k_slots = edge_k(cfg.task)
    assert output.source_logits.shape == (2, cfg.model.max_moves_k, MAX_PLANETS)
    assert output.target_logits.shape == (2, cfg.model.max_moves_k, k_slots)
    assert output.stop_logits.shape == (2, cfg.model.max_moves_k)
    assert output.ship_logits.shape == (
        2,
        cfg.model.max_moves_k,
        k_slots,
        cfg.task.ship_bucket_count,
    )
    assert output.value.shape == (2,)
    assert output.decoded_source_sequence.shape == (2, cfg.model.max_moves_k)
    assert output.decoded_target_slot_sequence.shape == (2, cfg.model.max_moves_k)
    assert output.decoded_stop_sequence.shape == (2, cfg.model.max_moves_k)


def test_factorized_pointer_masks_illegal_sources() -> None:
    cfg = _train_cfg(pointer_decoder="factorized_topk")
    policy = build_planet_graph_transformer_policy(cfg)
    batch = make_synthetic_turn_batch(1, cfg.task, key=jax.random.PRNGKey(2))
    batch = batch._replace(
        edge_mask=jnp.zeros_like(batch.edge_mask),
        planet_mask=jnp.ones_like(batch.planet_mask),
    )
    params = policy.init(jax.random.PRNGKey(3), batch)
    output = policy.apply(params, batch, deterministic=True)
    illegal = jnp.finfo(jnp.float32).min
    np.testing.assert_array_equal(
        np.asarray(output.source_logits),
        np.full(np.asarray(output.source_logits).shape, illegal),
    )
