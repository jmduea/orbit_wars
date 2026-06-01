#!/usr/bin/env python3
"""Micro-benchmark: factorized shield sampler policy.apply count vs encoder-once bound.

Run:
  uv run python scripts/benchmark_factorized_sampler.py
  uv run python scripts/benchmark_factorized_sampler.py --max-moves-k 8 --repeats 50
"""

from __future__ import annotations

import argparse
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
    """Encode once + critic + K shield decodes + K replay decodes + optional carry-out."""

    count = 2 + max_moves_k + max_moves_k
    if decoder_carry:
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-moves-k", type=int, default=3)
    parser.add_argument(
        "--decoder-carry", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    cfg = _train_cfg(max_moves_k=args.max_moves_k, decoder_carry=args.decoder_carry)
    keys = jax.random.split(jax.random.PRNGKey(0), args.batch_size)
    state, batch = batched_reset(keys, cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(1), batch)
    carry = (
        jnp.zeros((args.batch_size, cfg.model.hidden_size), dtype=jnp.float32)
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

    for _ in range(args.warmup):
        sample(jax.random.PRNGKey(2)).target_index.block_until_ready()

    timings = []
    for i in range(args.repeats):
        key = jax.random.fold_in(jax.random.PRNGKey(3), i)
        start = time.perf_counter()
        out = sample(key)
        out.target_index.block_until_ready()
        timings.append(time.perf_counter() - start)

    expected = _expected_apply_count(args.max_moves_k, decoder_carry=args.decoder_carry)
    replay_applies = args.max_moves_k
    mean_s = sum(timings) / len(timings)

    print(
        f"max_moves_k={args.max_moves_k} decoder_carry={args.decoder_carry} batch={args.batch_size}"
    )
    print(f"structural module applies per sample (sampler): ~{expected}")
    print(
        f"  breakdown: 1 encode + 1 critic + {args.max_moves_k} shield decode + "
        f"{replay_applies} replay decode"
    )
    if args.decoder_carry:
        print("  + 1 carry-out decode")
    print("encoder passes (target 1): 1 encode + 0 redundant full policy.apply")
    print(f"mean wall time per sample (JIT): {mean_s * 1000:.2f} ms")


if __name__ == "__main__":
    main()
