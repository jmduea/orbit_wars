#!/usr/bin/env python3
"""Fail if rollout/training smokes are collectable under the fast pytest tier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAK_PATTERNS = ("collect_rollout_jax", "run_jax_training", "init_rollout_groups")


def _test_body(source: str, test_name: str) -> str:
    base = test_name.split("[", 1)[0]
    marker = f"def {base}"
    if marker not in source:
        return ""
    return source.split(marker, 1)[1].split("\ndef ", 1)[0]


def main() -> int:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--group",
            "dev",
            "pytest",
            "--collect-only",
            "-m",
            "not slow and not jax and not sweep",
            "--quiet",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    leaks: list[str] = []
    for line in proc.stdout.splitlines():
        nodeid = line.strip()
        if "::" not in nodeid or not nodeid.startswith("tests/"):
            continue
        rel, test_name = nodeid.split("::", 1)
        test_path = ROOT / rel
        if not test_path.is_file():
            continue
        body = _test_body(test_path.read_text(encoding="utf-8"), test_name)
        for pattern in LEAK_PATTERNS:
            if pattern in body:
                leaks.append(f"{nodeid} references {pattern}")
                break

    if leaks:
        print(
            "Fast-tier leaks (heavy JAX smokes in 'not slow and not jax'):",
            file=sys.stderr,
        )
        for item in sorted(leaks):
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("No rollout/training smokes in fast tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
