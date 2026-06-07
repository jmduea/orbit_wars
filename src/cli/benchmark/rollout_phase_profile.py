"""``ow benchmark rollout-phase-profile`` — offline admission-shaped phase profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.cli.benchmark.common import REPO_ROOT

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


def _run_profile_in_process(args: argparse.Namespace) -> int:
    from src.benchmark.rollout_phase_profile import (
        compose_profile_config,
        format_profile_report,
        profile_result_payload,
        resolve_profile_overrides,
        run_rollout_phase_profile,
    )
    from src.cli.benchmark.common import _init_benchmark_runtime
    from src.jax.rollout.phase_timing_report import PhaseTimingWindow

    _init_benchmark_runtime()
    quick = not bool(args.full_geometry)
    overrides = resolve_profile_overrides(
        preset=args.preset,
        extra_overrides=tuple(args.train_overrides),
        updates=int(args.updates),
        model=args.model,
        quick=quick,
    )
    cfg = compose_profile_config(
        preset=args.preset,
        extra_overrides=tuple(args.train_overrides),
        updates=int(args.updates),
        model=args.model,
        quick=quick,
    )
    window = PhaseTimingWindow(
        warmup=int(args.warmup),
        max_measured_update=int(args.max_measured_update),
    )
    result = run_rollout_phase_profile(
        cfg,
        warmup=int(args.warmup),
        updates=int(args.updates),
        window=window,
    )
    payload = profile_result_payload(
        result,
        overrides=overrides,
        preset=args.preset,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_profile_report(payload))
    return 0


def _run_profile_from_json(payload_json: str) -> int:
    payload = json.loads(payload_json)
    out_path = payload.pop("out", None)
    args = argparse.Namespace(
        preset=payload.get("preset", "admission"),
        updates=int(payload.get("updates", 5)),
        warmup=int(payload.get("warmup", 2)),
        max_measured_update=int(payload.get("max_measured_update", 22)),
        model=payload.get("model"),
        json=bool(payload.get("json")),
        full_geometry=bool(payload.get("full_geometry")),
        train_overrides=list(payload.get("train_overrides") or []),
        repo_root=None,
        out=Path(out_path) if out_path else None,
    )
    return _run_profile_in_process(args)


def run_rollout_phase_profile_cli(args: argparse.Namespace) -> int:
    root = _integration_root(args)
    if root.resolve() == REPO_ROOT.resolve():
        return _run_profile_in_process(args)

    payload = {
        "preset": args.preset,
        "updates": args.updates,
        "warmup": args.warmup,
        "max_measured_update": args.max_measured_update,
        "model": args.model,
        "json": bool(args.json),
        "full_geometry": bool(args.full_geometry),
        "train_overrides": list(args.train_overrides),
        "out": str(args.out) if args.out is not None else None,
    }
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "import sys; "
                "from src.cli.benchmark.rollout_phase_profile import _run_profile_from_json; "
                "raise SystemExit(_run_profile_from_json(sys.argv[1]))"
            ),
            json.dumps(payload),
        ],
        cwd=root,
        check=False,
    )
    return int(proc.returncode)
