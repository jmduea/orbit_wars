#!/usr/bin/env python3
"""Pre-loop ladder baseline: per-rung phase fractions + throughput at worst rung."""

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
    LADDER_RUNG_ORDER,
    LADDER_RUNG_OVERRIDES,
    PROFILE_BASE_OVERRIDES,
    THROUGHPUT_SHARED_OVERRIDES,
)

DEFAULT_OUT = (
    REPO_ROOT
    / ".context"
    / "compound-engineering"
    / "ce-optimize"
    / "opponent-rollout-throughput"
    / "ladder-baseline.json"
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


def _opponent_fraction(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["phases"]["opponent"]["fraction_mean"])


def _profile_rung(
    rung: str,
    *,
    env: dict[str, str],
    updates: int,
    warmup: int,
    repeats: int,
    scratch: Path,
) -> float:
    overrides = LADDER_RUNG_OVERRIDES[rung]
    fractions: list[float] = []
    for repeat in range(repeats):
        out = scratch / f"profile_{rung}_{repeat}.json"
        proc = _run(
            [
                "uv",
                "run",
                "ow",
                "benchmark",
                "rollout-phase-profile",
                "--preset",
                "admission",
                "--updates",
                str(updates),
                "--warmup",
                str(warmup),
                "--out",
                str(out),
                "--train-overrides",
                *PROFILE_BASE_OVERRIDES,
                *overrides,
            ],
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"profile failed for rung={rung}: {(proc.stderr or proc.stdout)[-1500:]}"
            )
        fractions.append(_opponent_fraction(out))
    return statistics.median(fractions)


def _throughput_at_rung(
    rung: str,
    *,
    env: dict[str, str],
    updates: int,
    warmup: int,
    repeats: int,
    scratch: Path,
) -> float:
    overrides = list(
        dict.fromkeys(THROUGHPUT_SHARED_OVERRIDES + LADDER_RUNG_OVERRIDES[rung])
    )
    rates: list[float] = []
    for repeat in range(repeats):
        out = scratch / f"training_{rung}_{repeat}.json"
        proc = _run(
            [
                "uv",
                "run",
                "ow",
                "benchmark",
                "training",
                "--label",
                "ladder_baseline",
                "--updates",
                str(updates),
                "--warmup",
                str(warmup),
                "--repeats",
                "1",
                "--detailed-timing",
                "--out",
                str(out),
                "--overrides",
                *overrides,
            ],
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"training benchmark failed for rung={rung}: "
                f"{(proc.stderr or proc.stdout)[-1500:]}"
            )
        bench = json.loads(out.read_text(encoding="utf-8"))
        rates.append(float(bench.get("env_steps_per_sec", 0.0)))
    return statistics.median(rates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--throughput-repeats", type=int, default=3)
    parser.add_argument("--profile-updates", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--training-updates", type=int, default=20)
    parser.add_argument("--training-warmup", type=int, default=2)
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Profile ladder only (skip training benchmark at worst rung).",
    )
    args = parser.parse_args()

    env = _cold_cache_env()
    scratch = args.out.parent / "capture_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    rung_rows: list[dict[str, object]] = []
    for rung in LADDER_RUNG_ORDER:
        print(f"capture: profiling rung={rung}", file=sys.stderr, flush=True)
        fraction = _profile_rung(
            rung,
            env=env,
            updates=int(args.profile_updates),
            warmup=int(args.profile_warmup),
            repeats=int(args.profile_repeats),
            scratch=scratch,
        )
        rung_rows.append(
            {
                "label": rung,
                "overrides": LADDER_RUNG_OVERRIDES[rung],
                "opponent_fraction_median": fraction,
            }
        )

    worst_row = max(rung_rows, key=lambda row: float(row["opponent_fraction_median"]))
    worst_rung = str(worst_row["label"])
    throughput_median = None
    if not args.skip_throughput:
        print(
            f"capture: throughput at worst_rung={worst_rung}",
            file=sys.stderr,
            flush=True,
        )
        throughput_median = _throughput_at_rung(
            worst_rung,
            env=env,
            updates=int(args.training_updates),
            warmup=int(args.training_warmup),
            repeats=int(args.throughput_repeats),
            scratch=scratch,
        )

    payload: dict[str, object] = {
        "rungs": rung_rows,
        "worst_rung": worst_rung,
        "throughput": (
            {"env_steps_per_sec_median": throughput_median}
            if throughput_median is not None
            else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
