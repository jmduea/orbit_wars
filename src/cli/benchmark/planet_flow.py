"""Planet Flow benchmark subcommands."""

from __future__ import annotations

import argparse
import json

from src.cli.benchmark.common import REPO_ROOT


def run_shortlist_planet_flow_sweep_cli(args: argparse.Namespace) -> int:
    from src.jax.planet_flow_shortlist import (
        PLANET_FLOW_MAX_APPROX_KL,
        PLANET_FLOW_MIN_ENTROPY,
        build_shortlist_report,
        fetch_finished_sweep_runs,
        write_shortlist_report,
    )

    max_kl = (
        float(args.max_kl) if args.max_kl is not None else PLANET_FLOW_MAX_APPROX_KL
    )
    min_entropy = (
        float(args.min_entropy)
        if args.min_entropy is not None
        else PLANET_FLOW_MIN_ENTROPY
    )
    runs = fetch_finished_sweep_runs(
        entity=args.entity,
        project=args.project,
        sweep_id=args.sweep_id,
    )
    report = build_shortlist_report(
        runs,
        sweep_id=args.sweep_id,
        max_kl=max_kl,
        min_entropy=min_entropy,
        limit=args.limit,
    )
    write_shortlist_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def run_planet_flow_noop_smoke_cli(args: argparse.Namespace) -> int:
    from src.jax.planet_flow_smoke import run_planet_flow_noop_smoke, write_smoke_report

    report = run_planet_flow_noop_smoke(
        args.shortlist,
        top_k=int(args.top_k),
        output_root=args.output_root,
        repo_root=REPO_ROOT,
        thresholds_path=args.thresholds_path,
        dry_run=bool(args.dry_run),
    )
    write_smoke_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("any_passed") else 1
