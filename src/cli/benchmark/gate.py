"""``ow benchmark gate`` thin wrapper."""

from __future__ import annotations

import argparse
import json
import sys


def _resolve_gate_id(args: argparse.Namespace) -> str | None:
    if args.list:
        return None
    tokens = [str(token) for token in args.tokens]
    if not tokens or tokens[0] == "list":
        return None
    if tokens[0] == "run":
        return tokens[1] if len(tokens) > 1 else None
    return tokens[0]


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


def run_gate_cli(args: argparse.Namespace) -> int:
    from src.cli.benchmark_gates import list_gate_recipes, run_gate_cli as run_gate

    gate_id = _resolve_gate_id(args)
    if gate_id is None:
        payload = {"gates": list_gate_recipes()}
        print(json.dumps(payload, indent=2))
        return 0
    if gate_id not in {"beat_noop", "beat_random", "curriculum_staged"}:
        print(
            f"Unknown gate id {gate_id!r}. Use: ow benchmark gate run beat_noop",
            file=sys.stderr,
        )
        return 2
    return run_gate(
        gate_id,
        model=args.model,
        output_root=args.output_root,
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        thresholds_path=args.thresholds_path,
        profiles_path=args.profile_path,
        train_overrides=tuple(args.train_overrides),
        out=args.out,
    )
