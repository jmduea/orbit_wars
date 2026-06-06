"""``ow benchmark encode-turn`` microbenchmark."""

from __future__ import annotations

import argparse


def run_encode_turn_cli(args: argparse.Namespace) -> int:
    from src.jax.encode_turn_benchmark import run_encode_turn_benchmark

    edge_rank_modes: list[str] | None = None
    if args.edge_rank_mode is not None:
        edge_rank_modes = [str(args.edge_rank_mode)]

    return run_encode_turn_benchmark(
        batch_size=int(args.batch_size),
        player_count=int(args.player_count),
        candidate_count=int(args.candidate_count),
        edge_rank_modes=edge_rank_modes,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        include_learner_turn=bool(args.include_learner_turn),
        include_4p_all_players=bool(args.include_4p_all_players),
        sweep_defaults=bool(args.sweep_defaults),
        out=args.out,
    )
