"""``ow benchmark calibrate unified`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cli.benchmark.common import REPO_ROOT, _git_head_sha, _init_benchmark_runtime

def run_calibrate_unified_tournament_cli(args: argparse.Namespace) -> int:
    from src.jax.unified_tournament_calibration import (
        DEFAULT_CALIBRATION_CHECKPOINT,
        UnifiedCalibrationPlan,
        build_unified_calibration_report,
        default_unified_tournament_stub,
        discover_unified_cal_snapshots,
        load_unified_section_from_calibration,
        merge_unified_section_into_calibration,
        run_unified_calibration_sweep,
        write_unified_calibration_artifact,
    )

    started = __import__("time").perf_counter()
    games_candidates = tuple(
        int(part.strip()) for part in args.games_per_pair.split(",") if part.strip()
    )
    checkpoints = tuple(args.checkpoint) or (DEFAULT_CALIBRATION_CHECKPOINT,)
    for checkpoint in checkpoints:
        if not checkpoint.is_file() and not args.dry_run and not args.write_stub:
            print(f"missing checkpoint: {checkpoint}", file=sys.stderr)
            return 1

    base_section = load_unified_section_from_calibration(args.out)
    plan = UnifiedCalibrationPlan(
        checkpoint_paths=checkpoints,
        games_per_pair_candidates=games_candidates or (4,),
        dry_run=bool(args.dry_run),
        output_root=args.output_root,
    )
    snapshots = []
    if args.write_stub:
        stub = default_unified_tournament_stub(enforcement=False)
        if base_section and base_section.get("incumbent_bootstrap_opponent"):
            stub["incumbent_bootstrap_opponent"] = base_section[
                "incumbent_bootstrap_opponent"
            ]
        merged = merge_unified_section_into_calibration(args.out, stub)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        report = build_unified_calibration_report(
            repo_root=REPO_ROOT,
            plan=plan,
            snapshots=[],
            analyze_only=True,
            seconds_total=__import__("time").perf_counter() - started,
            base_section=base_section,
            enable_enforcement=False,
        )
        report["written_calibration_path"] = str(args.out)
        write_unified_calibration_artifact(args.artifact_out, report)
        print(json.dumps(report, indent=2))
        return 0

    if not args.analyze_only:
        snapshots.extend(
            run_unified_calibration_sweep(
                plan=plan,
                repo_root=REPO_ROOT,
                base_section=base_section,
            )
        )
    snapshots.extend(
        discover_unified_cal_snapshots(
            plan.output_root,
            games_per_pair_candidates=plan.games_per_pair_candidates,
            checkpoint_paths=checkpoints,
        )
    )
    seen: set[tuple[str, int]] = set()
    deduped: list = []
    for snapshot in snapshots:
        key = (snapshot.checkpoint_path, snapshot.games_per_pair)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snapshot)

    report = build_unified_calibration_report(
        repo_root=REPO_ROOT,
        plan=plan,
        snapshots=deduped,
        analyze_only=bool(args.analyze_only),
        seconds_total=__import__("time").perf_counter() - started,
        base_section=base_section,
        enable_enforcement=True,
    )
    write_unified_calibration_artifact(args.artifact_out, report)
    unified_section = report.get("unified_tournament")
    if isinstance(unified_section, dict):
        merged = merge_unified_section_into_calibration(args.out, unified_section)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        report["written_preflight_calibration_path"] = str(args.out)
    report["written_artifact_path"] = str(args.artifact_out)
    print(json.dumps(report, indent=2))
    if plan.dry_run:
        print(
            "Dry run: no GPU calibration campaigns executed. "
            "Omit --dry-run to run Stage-1 unified ladder sweeps.",
            flush=True,
        )
    elif not deduped:
        print(
            "No calibration snapshots found; run without --analyze-only to execute campaigns.",
            file=sys.stderr,
        )
        return 1
    return 0

