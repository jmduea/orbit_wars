from __future__ import annotations

import jax.numpy as jnp
import pytest

from src.config import TrainConfig
from src.jax.action_sampling import _sample_factored_step_from_logits
from src.jax.submission_runtime import apply_feature_metadata_to_model_config


def _sample_stop(*, deterministic: bool, stop_logit: float, seed: int) -> int:
    source_logits = jnp.array([[0.0, -5.0]], dtype=jnp.float32)
    target_logits = jnp.array([[0.0, -5.0]], dtype=jnp.float32)
    stop_logits = jnp.array([stop_logit], dtype=jnp.float32)
    ship_logits = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    source_mask = jnp.array([[True, False]], dtype=bool)
    ship_bucket_mask = jnp.array([[[False, True], [False, False]]], dtype=bool)

    _, _, _, stop, _, _, _ = _sample_factored_step_from_logits(
        jnp.array([seed, seed + 1], dtype=jnp.uint32),
        source_logits,
        target_logits,
        stop_logits,
        ship_logits,
        source_mask,
        ship_bucket_mask,
        deterministic=deterministic,
    )
    return int(stop[0])


def test_deterministic_respects_stop_when_can_launch() -> None:
    assert _sample_stop(deterministic=True, stop_logit=10.0, seed=0) == 1


def test_stochastic_can_sample_stop_when_can_launch() -> None:
    stops = {
        _sample_stop(deterministic=False, stop_logit=0.0, seed=seed)
        for seed in range(32)
    }
    assert 0 in stops and 1 in stops


def test_submission_runtime_rejects_planet_flow_checkpoint_metadata() -> None:
    cfg = TrainConfig()
    metadata = {
        "pointer_decoder": "planet_flow_target_heatmap",
        "action_layout_version": 3,
        "pressure_bucket_values": (0.0, 0.25, 0.5, 0.75, 1.0),
    }

    with pytest.raises(ValueError, match="Planet Flow checkpoints"):
        apply_feature_metadata_to_model_config(cfg, metadata)
