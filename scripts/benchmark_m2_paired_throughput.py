#!/usr/bin/env python3
"""Paired M2 throughput benchmark — one architecture at a time, never in parallel.

Runs GNN and planet_graph_transformer benchmarks sequentially with identical
warmup/update budgets, then reports per-arm medians and the H2 ratio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts/benchmark_jax_rl.py"
SUMMARY_PATH = REPO_ROOT / "artifacts/m2/benchmark_summary.json"

SHARED_OVERRIDES = [
    "format=mix_2p_4p_16env",
    "training=ablation_m2",
    "telemetry=ablation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=SUMMARY_PATH,
        help="JSON summary path (default: artifacts/m2/benchmark_summary.json).",
    )
    return parser.parse_args()


def _run_single(model: str, *, warmup: int, updates: int) -> dict:
    cmd = [
        "uv",
        "run",
        "python",
        str(BENCHMARK_SCRIPT),
        "--overrides",
        f"model={model}",
        *SHARED_OVERRIDES,
        "--warmup",
        str(warmup),
        "--updates",
        str(updates),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _median_metric(rows: list[dict], key: str) -> float:
    return float(statistics.median(row[key] for row in rows))


def main() -> None:
    args = parse_args()
    arms = [
        ("gnn_pointer", "gnn_pointer"),
        ("planet_graph_transformer", "planet_graph_transformer"),
    ]
    summary: dict = {
        "methodology": "sequential_only_one_process_at_a_time",
        "benchmark_config": {
            "shared_overrides": SHARED_OVERRIDES,
            "warmup_updates": args.warmup,
            "measured_updates": args.updates,
            "reps_per_arm": args.reps,
        },
        "runs": {},
    }

    for arm_key, model in arms:
        reps: list[dict] = []
        for rep_idx in range(args.reps):
            print(f"[benchmark] {model} rep {rep_idx + 1}/{args.reps}", flush=True)
            reps.append(_run_single(model, warmup=args.warmup, updates=args.updates))
        summary["runs"][arm_key] = reps
        summary[arm_key] = {
            "env_steps_per_sec_median": _median_metric(reps, "env_steps_per_sec"),
            "samples_per_sec_median": _median_metric(reps, "samples_per_sec"),
        }

    gnn_sps = summary["gnn_pointer"]["env_steps_per_sec_median"]
    tx_sps = summary["planet_graph_transformer"]["env_steps_per_sec_median"]
    summary["h2_ratio_transformer_over_gnn"] = tx_sps / max(gnn_sps, 1e-9)
    summary["h2_pass"] = summary["h2_ratio_transformer_over_gnn"] >= 0.90

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stdout)
        print(exc.stderr or "", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
