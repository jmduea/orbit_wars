"""``ow benchmark rollout-phase-profile`` — delegates to integration worktree."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTEGRATION_ROOT = REPO_ROOT.parent / "orbit_wars-integration"


def _integration_root(args: argparse.Namespace) -> Path:
    raw = args.repo_root or os.environ.get("ORBIT_WARS_INTEGRATION_ROOT")
    root = Path(raw) if raw else DEFAULT_INTEGRATION_ROOT
    if not (root / "src/jax/rollout/collect_timed.py").is_file():
        raise SystemExit(
            "rollout-phase-profile requires the integration worktree "
            f"(missing timed collect at {root}). "
            "Run from orbit_wars-integration or pass --repo-root.",
            file=sys.stderr,
        )
    return root


def run_rollout_phase_profile_cli(args: argparse.Namespace) -> int:
    root = _integration_root(args)
    cmd = [
        "uv",
        "run",
        "ow",
        "benchmark",
        "rollout-phase-profile",
        "--preset",
        args.preset,
        "--updates",
        str(args.updates),
        "--warmup",
        str(args.warmup),
        "--max-measured-update",
        str(args.max_measured_update),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.json:
        cmd.append("--json")
    if args.full_geometry:
        cmd.append("--full-geometry")
    if args.train_overrides:
        cmd.append("--train-overrides")
        cmd.extend(args.train_overrides)
    if args.out is not None:
        cmd.extend(["--out", str(args.out)])
    proc = subprocess.run(cmd, cwd=root, check=False)
    return int(proc.returncode)
