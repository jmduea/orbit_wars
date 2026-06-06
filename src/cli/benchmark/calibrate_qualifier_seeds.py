"""``ow benchmark calibrate-qualifier-seeds`` — SSOT qualifier floor calibration."""

from __future__ import annotations

import argparse
import json
import sys

from src.benchmark.calibration.qualifier_floors import (
    default_qualifier_calibration_stub,
)
from src.cli.benchmark.common import REPO_ROOT, _git_head_sha, _init_benchmark_runtime


def run_calibrate_qualifier_seeds_cli(args: argparse.Namespace) -> int:
    _init_benchmark_runtime()
    started = __import__("time").perf_counter()

    if args.write_stub:
        payload = default_qualifier_calibration_stub(enforcement=bool(args.enforcement))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        report = {
            "ok": True,
            "mode": "write_stub",
            "written_calibration_path": str(args.out),
            "git_head": _git_head_sha(REPO_ROOT),
            "seconds_total": __import__("time").perf_counter() - started,
        }
        print(json.dumps(report, indent=2))
        return 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "dry_run",
                    "out": str(args.out),
                    "note": "JAX qualifier calibration campaign not yet automated; use --write-stub.",
                },
                indent=2,
            )
        )
        return 0

    if args.analyze_only:
        print(
            "analyze-only: discover ssot qualifier calibration campaigns under "
            f"{args.output_root} (not implemented)",
            file=sys.stderr,
        )
        return 1

    print(
        "calibrate-qualifier-seeds: run a JAX eval campaign on fixed checkpoints, "
        "then re-run with --analyze-only. For now use --write-stub.",
        file=sys.stderr,
    )
    return 1
