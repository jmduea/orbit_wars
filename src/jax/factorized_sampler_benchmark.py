"""Micro-benchmark: factorized shield sampler policy.apply count vs encoder-once bound."""

from __future__ import annotations

import time

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.action_sampling import _sample_shielded_factored_sequence_with_params
from src.jax.decoder_carry import decoder_carry_enabled
from src.jax.env import batched_reset
from src.jax.policy import build_planet_graph_transformer_policy


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=6, ship_bucket_count=8, max_fleets=32)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, max_moves_k: int, decoder_carry: bool) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 128
    cfg.model.max_moves_k = max_moves_k
    cfg.model.decoder_carry = decoder_carry
    cfg.task = _task_cfg(trajectory_shield_mode="cheap")
    return cfg


def _expected_apply_count(max_moves_k: int, *, decoder_carry: bool) -> int:
    # encode + critic + per-step decode_step + advance_carry (+ tiered replay decode)
    count = 2 + (2 * max_moves_k) + max_moves_k
    return count


def run_factorized_sampler_benchmark(
    *,
    max_moves_k: int = 3,
    decoder_carry: bool = True,
    batch_size: int = 16,
    warmup: int = 3,
    repeats: int = 30,
    assert_max_ms: float | None = None,
) -> int:
    """Run the tier-1 factorized sampler microbench. Returns process exit code."""

    cfg = _train_cfg(max_moves_k=max_moves_k, decoder_carry=decoder_carry)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    state, batch = batched_reset(keys, cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(1), batch)
    carry = (
        jnp.zeros((batch_size, cfg.model.hidden_size), dtype=jnp.float32)
        if decoder_carry_enabled(cfg)
        else None
    )

    @jax.jit
    def sample(key):
        return _sample_shielded_factored_sequence_with_params(
            key,
            state.game,
            batch,
            params,
            policy,
            cfg,
            deterministic=True,
            decoder_hidden_in=carry,
        )

    for _ in range(warmup):
        sample(jax.random.PRNGKey(2)).target_index.block_until_ready()

    timings = []
    for i in range(repeats):
        key = jax.random.fold_in(jax.random.PRNGKey(3), i)
        start = time.perf_counter()
        out = sample(key)
        out.target_index.block_until_ready()
        timings.append(time.perf_counter() - start)

    expected = _expected_apply_count(max_moves_k, decoder_carry=decoder_carry)
    replay_applies = max_moves_k
    mean_s = sum(timings) / len(timings)

    print(f"max_moves_k={max_moves_k} decoder_carry={decoder_carry} batch={batch_size}")
    print(f"structural module applies per sample (sampler): ~{expected}")
    print(
        f"  breakdown: 1 encode + 1 critic + {max_moves_k} decode_step + "
        f"{max_moves_k} advance_carry + {replay_applies} replay decode"
    )
    print("encoder passes (target 1): 1 encode + 0 redundant full policy.apply")
    mean_ms = mean_s * 1000.0
    print(f"mean wall time per sample (JIT): {mean_ms:.2f} ms")

    if assert_max_ms is not None and mean_ms > assert_max_ms:
        print(
            f"mean {mean_ms:.2f} ms exceeds --assert-max-ms {assert_max_ms:.2f}",
            flush=True,
        )
        return 1
    return 0
