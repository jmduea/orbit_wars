"""``ow benchmark factorized-sampler`` microbenchmark."""

from __future__ import annotations

import argparse


def run_factorized_sampler_cli(args: argparse.Namespace) -> int:
    from src.jax.factorized_sampler_benchmark import run_factorized_sampler_benchmark

    return run_factorized_sampler_benchmark(
        max_moves_k=int(args.max_moves_k),
        decoder_carry=bool(args.decoder_carry),
        batch_size=int(args.batch_size),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        assert_max_ms=args.assert_max_ms,
    )

