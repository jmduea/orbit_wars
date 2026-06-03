"""``ow benchmark calibrate`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cli.benchmark.common import REPO_ROOT, _git_head_sha, _init_benchmark_runtime

def run_calibrate_cli(args: argparse.Namespace) -> int:
    from src.jax.preflight_calibration import (
        analyze_jsonl_path,
        build_calibration_report,
        default_calibration_json_path,
        derive_thresholds,
        discover_calibration_snapshots,
        git_head_sha,
        refresh_agents_md_thresholds,
        run_calibration_sweep,
        summarize_calibration,
        write_calibration_report,
    )

    started = __import__("time").perf_counter()
    opponents = tuple(
        part.strip()
        for part in args.opponents.split(",")
        if part.strip() in {"noop_only", "random_only"}
    )
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    update_counts = tuple(
        int(part.strip()) for part in args.updates.split(",") if part.strip()
    )
    snapshots = []
    for spec in args.analyze_jsonl:
        path_text, opponent, seed_text, updates_text = (spec.split(":") + ["", "", ""])[
            :4
        ]
        if not path_text:
            continue
        snapshots.append(
            analyze_jsonl_path(
                Path(path_text),
                opponent=opponent,  # type: ignore[arg-type]
                seed=int(seed_text),
                total_updates=int(updates_text),
                model=args.model,
            )
        )
    if args.analyze_campaigns is not None:
        snapshots.extend(
            discover_calibration_snapshots(
                args.output_root,
                campaign_glob=args.analyze_campaigns,
                model=args.model,
            )
        )
    if not args.analyze_only:
        snapshots.extend(
            run_calibration_sweep(
                opponents=opponents,
                seeds=seeds,
                update_counts=update_counts,
                model=args.model,
                output_root=args.output_root,
                repo_root=REPO_ROOT,
                dry_run=args.dry_run,
                profiles_path=args.profile_path,
            )
        )
    summary = summarize_calibration(snapshots)
    thresholds = derive_thresholds(summary)
    report = build_calibration_report(
        snapshots,
        commit_sha=git_head_sha(REPO_ROOT),
        thresholds=thresholds,
        seconds_total=__import__("time").perf_counter() - started,
        analyze_only=bool(args.analyze_only),
    )
    write_calibration_report(args.out, report)
    if not args.dry_run and args.out.resolve() == default_calibration_json_path(REPO_ROOT).resolve():
        refresh_agents_md_thresholds(REPO_ROOT, report)
    print(json.dumps(report, indent=2))
    return 0

