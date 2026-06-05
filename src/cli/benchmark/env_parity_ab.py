"""``ow benchmark env-parity-ab`` — compare env_parity_mode throughput arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cli.benchmark.common import _git_head_sha, _init_benchmark_runtime


def run_env_parity_ab_cli(args: argparse.Namespace) -> int:
    from src.jax.benchmark_progress import emit_benchmark_progress
    from src.jax.env_parity_benchmark import run_env_parity_ab_benchmark

    _init_benchmark_runtime()
    modes = tuple(m.strip().lower() for m in args.modes.split(",") if m.strip())
    emit_benchmark_progress(
        f"env-parity-ab: modes={modes} batch={args.batch_size} "
        f"steps={args.steps} repeats={args.repeats}"
    )
    payload = run_env_parity_ab_benchmark(
        batch_size=int(args.batch_size),
        steps_per_episode=int(args.steps),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        modes=modes,
        commit_sha=_git_head_sha(),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)
    emit_benchmark_progress("env-parity-ab: done")
    return 0


def add_env_parity_ab_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "env-parity-ab",
        help=(
            "A/B JAX env throughput: legacy (no comets) vs train vs kaggle reference."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON results here (stdout always prints JSON).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Noop env steps per episode per repeat.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        default="legacy,train,kaggle",
        help="Comma-separated env_parity_mode arms.",
    )
