#!/usr/bin/env python3
"""CE-optimize measurement harness: opponent rollout phase fraction at worst ladder rung."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ce_optimize.opponent_ladder_rungs import (  # noqa: E402
    LADDER_RUNG_OVERRIDES,
    PROFILE_BASE_OVERRIDES,
)

TARGETED_TESTS = [
    "tests/test_rollout_noop_opponent.py",
    "tests/test_opponent_ladder_compose.py",
]

SCRATCH_DIR = (
    REPO_ROOT
    / ".context"
    / "compound-engineering"
    / "ce-optimize"
    / "opponent-rollout-throughput"
)


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _cold_cache_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env["ORBIT_WARS_PYTEST_JAX_CACHE"] = "0"
    return env


def _opponent_fraction_from_profile(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    phases = payload.get("phases")
    if isinstance(phases, dict) and isinstance(phases.get("opponent"), dict):
        return float(phases["opponent"]["fraction_mean"])
    from src.jax.rollout.phase_timing_report import (
        extract_rollout_phase_breakdown_from_input,
    )

    breakdown = extract_rollout_phase_breakdown_from_input(path)
    opponent = breakdown.get("phases", {}).get("opponent", {})
    return float(opponent["fraction_mean"])


def _finite_phase_metrics(fraction: float) -> int:
    import math

    return 1 if math.isfinite(fraction) and fraction >= 0.0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", required=True, choices=sorted(LADDER_RUNG_OVERRIDES))
    parser.add_argument(
        "--ladder-baseline",
        type=Path,
        default=None,
        help="Optional ladder-baseline.json for diagnostics (worst_rung pin).",
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    env = _cold_cache_env()
    test_proc = _run(["uv", "run", "pytest", *TARGETED_TESTS, "-q", "--tb=no"], env=env)
    tests_passed = 1 if test_proc.returncode == 0 else 0

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    rung_overrides = LADDER_RUNG_OVERRIDES[args.rung]
    fractions: list[float] = []
    last_error = ""

    for repeat in range(max(int(args.repeats), 1)):
        profile_out = SCRATCH_DIR / f"last_profile_{args.rung}_{repeat}.json"
        bench_cmd = [
            "uv",
            "run",
            "ow",
            "benchmark",
            "rollout-phase-profile",
            "--preset",
            "admission",
            "--updates",
            str(int(args.updates)),
            "--warmup",
            str(int(args.warmup)),
            "--out",
            str(profile_out),
            "--train-overrides",
            *PROFILE_BASE_OVERRIDES,
            *rung_overrides,
        ]
        bench_proc = _run(bench_cmd, env=env)
        if bench_proc.returncode != 0:
            last_error = (bench_proc.stderr or bench_proc.stdout or "")[-2000:]
            continue
        try:
            fractions.append(_opponent_fraction_from_profile(profile_out))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
            continue

    benchmark_passed = 1 if fractions else 0
    opponent_fraction = statistics.median(fractions) if fractions else 0.0
    finite_metrics = _finite_phase_metrics(opponent_fraction)

    worst_rung = args.rung
    if args.ladder_baseline is not None and args.ladder_baseline.is_file():
        try:
            baseline_doc = json.loads(args.ladder_baseline.read_text(encoding="utf-8"))
            worst_rung = str(baseline_doc.get("worst_rung", worst_rung))
        except (OSError, json.JSONDecodeError):
            pass

    payload = {
        "tests_passed": tests_passed,
        "benchmark_passed": benchmark_passed,
        "finite_phase_metrics": finite_metrics,
        "rollout_phase_opponent_fraction_worst_rung": opponent_fraction,
        "ladder_rung": args.rung,
        "worst_rung_pinned": worst_rung,
        "rollout_phase_opponent_fraction": opponent_fraction,
        "repeats_ok": len(fractions),
        "error": last_error,
    }
    print(json.dumps(payload))
    ok = tests_passed and benchmark_passed and finite_metrics
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
