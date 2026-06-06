"""``ow benchmark policy-path-profile`` microbenchmark."""

from __future__ import annotations

import argparse


def run_policy_path_profile_cli(args: argparse.Namespace) -> int:
    from src.benchmark.policy_path_profile import run_policy_path_profile_benchmark

    shield_modes: list[str] | None = None
    if args.shield_modes is not None:
        shield_modes = [str(mode) for mode in args.shield_modes]

    return run_policy_path_profile_benchmark(
        batch_size=int(args.batch_size),
        max_moves_k=int(args.max_moves_k),
        decoder_carry=bool(args.decoder_carry),
        candidate_count=int(args.candidate_count),
        shield_modes=shield_modes,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        out=args.out,
    )
