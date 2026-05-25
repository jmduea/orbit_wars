from __future__ import annotations

import jax.numpy as jnp

from src.opponents.jax_actions.builders import (
    _pick_eval_deterministic_bucket,
    _sample_step_from_logits,
)


def test_eval_deterministic_masks_noop_when_launch_available() -> None:
    target_logits = jnp.array([[0.1, -0.5, -0.6, 0.2]], dtype=jnp.float32)
    ship_logits = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    ship_bucket_mask = jnp.array(
        [
            [
                [False, True],
                [False, True],
                [False, False],
                [True, False],
            ]
        ],
        dtype=bool,
    )

    target, bucket, _, _ = _sample_step_from_logits(
        key=jnp.array([0, 0], dtype=jnp.uint32),
        target_logits=target_logits,
        ship_logits=ship_logits,
        ship_bucket_mask=ship_bucket_mask,
        deterministic=True,
        deterministic_eval=True,
    )

    assert int(target[0]) == 0
    assert int(bucket[0]) == 1


def test_eval_deterministic_falls_back_to_noop_without_launch() -> None:
    target_logits = jnp.array([[0.1, -0.5, -0.6, 0.2]], dtype=jnp.float32)
    ship_logits = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    ship_bucket_mask = jnp.array(
        [
            [
                [False, False],
                [False, False],
                [False, False],
                [True, False],
            ]
        ],
        dtype=bool,
    )

    target, bucket, _, _ = _sample_step_from_logits(
        key=jnp.array([0, 0], dtype=jnp.uint32),
        target_logits=target_logits,
        ship_logits=ship_logits,
        ship_bucket_mask=ship_bucket_mask,
        deterministic=True,
        deterministic_eval=True,
    )

    assert int(target[0]) == 3
    assert int(bucket[0]) == 0


def test_pick_eval_deterministic_bucket_prefers_nonzero() -> None:
    target = jnp.array([1], dtype=jnp.int32)
    selected_bucket_mask = jnp.array([[False, True, True]], dtype=bool)
    bucket = jnp.array([0], dtype=jnp.int32)

    picked = _pick_eval_deterministic_bucket(
        target,
        selected_bucket_mask,
        bucket,
        noop_idx=3,
    )

    assert int(picked[0]) == 1
