"""SSOT preflight benchmark subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_shortlist_ssot_preflight_sweep_cli(args: argparse.Namespace) -> int:
    from src.jax.ssot_preflight_shortlist import (
        build_ssot_shortlist_report,
        fetch_finished_sweep_runs,
        write_shortlist_report,
    )

    runs = fetch_finished_sweep_runs(
        entity=args.entity,
        project=args.project,
        sweep_id=args.sweep_id,
    )
    report = build_ssot_shortlist_report(
        runs,
        sweep_id=args.sweep_id,
        limit=args.limit,
    )
    write_shortlist_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0
