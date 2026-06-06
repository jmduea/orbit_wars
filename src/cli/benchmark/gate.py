"""``ow benchmark gate`` thin wrapper."""

from __future__ import annotations

import argparse
import json
import sys

KNOWN_GATE_IDS = frozenset(
    {"admission", "beat_noop", "beat_random", "curriculum_staged"}
)


def _resolve_gate_id(args: argparse.Namespace) -> str | None:
    if args.list:
        return None
    tokens = [str(token) for token in args.tokens]
    if not tokens or tokens[0] == "list":
        return None
    if tokens[0] == "run":
        return tokens[1] if len(tokens) > 1 else None
    return tokens[0]


def run_gate_cli(args: argparse.Namespace) -> int:
    from src.cli.benchmark_gates import list_gate_recipes
    from src.cli.benchmark_gates import run_gate_cli as run_gate

    gate_id = _resolve_gate_id(args)
    if gate_id is None:
        payload = {"gates": list_gate_recipes()}
        print(json.dumps(payload, indent=2))
        return 0
    if gate_id not in KNOWN_GATE_IDS:
        print(
            f"Unknown gate id {gate_id!r}. Use: ow benchmark gate run admission",
            file=sys.stderr,
        )
        return 2
    if args.also_throughput and args.out is None:
        print(
            "--also-throughput requires --out so the gate JSON path is known",
            file=sys.stderr,
        )
        return 2
    return run_gate(
        gate_id,
        model=args.model,
        output_root=args.output_root,
        repo_root=args.repo_root,
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        thresholds_path=args.thresholds_path,
        profiles_path=args.profile_path,
        train_overrides=tuple(args.train_overrides),
        out=args.out,
        include_throughput=bool(args.also_throughput or gate_id == "admission"),
        throughput_baseline=args.throughput_baseline,
        throughput_within_pct=args.throughput_within_pct,
    )
