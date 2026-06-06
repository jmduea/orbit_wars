"""``ow benchmark calibrate seed`` command."""

from __future__ import annotations

import argparse
import json

from src.cli.benchmark.common import REPO_ROOT


def run_calibrate_seed_scheduler_cli(args: argparse.Namespace) -> int:
    from src.benchmark.calibration.seed_scheduler import (
        DEFAULT_OPPONENTS,
        analyze_seed_sched_run,
        build_seed_scheduler_calibration_report,
        discover_seed_sched_runs,
        expand_reseed_intervals,
        run_seed_scheduler_sweep,
        write_seed_scheduler_calibration_report,
    )
    from src.jax.preflight_calibration import git_head_sha

    started = __import__("time").perf_counter()
    opponents = tuple(
        part.strip()
        for part in args.opponents.split(",")
        if part.strip() in DEFAULT_OPPONENTS
    )
    if not opponents:
        opponents = DEFAULT_OPPONENTS
    reseed_intervals = expand_reseed_intervals(
        tuple(
            int(part.strip())
            for part in args.reseed_intervals.split(",")
            if part.strip()
        ),
        total_updates=int(args.total_updates),
        include_total_fifth=not bool(args.no_include_total_fifth),
    )
    eval_seeds = tuple(
        int(part.strip()) for part in args.eval_seeds.split(",") if part.strip()
    )
    snapshots = []
    if not args.analyze_only:
        run_seed_scheduler_sweep(
            opponents=opponents,  # type: ignore[arg-type]
            reseed_intervals=reseed_intervals,
            total_updates=int(args.total_updates),
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            dry_run=bool(args.dry_run),
        )
    for opponent, reseed_interval, run_dir in discover_seed_sched_runs(
        args.output_root,
        total_updates=int(args.total_updates),
        train_seed=int(args.train_seed),
    ):
        if opponent not in opponents:
            continue
        if reseed_interval not in reseed_intervals:
            continue
        snapshots.append(
            analyze_seed_sched_run(
                opponent=opponent,
                reseed_interval=reseed_interval,
                total_updates=int(args.total_updates),
                train_seed=int(args.train_seed),
                run_dir=run_dir,
                eval_seeds=eval_seeds,
                repo_root=REPO_ROOT,
                output_root=args.output_root,
                baseline=str(args.baseline),
                games_per_pair=int(args.games_per_pair),
                dry_run=bool(args.dry_run),
                run_eval=not args.analyze_only or bool(args.eval_existing),
            )
        )
    snapshots = [item for item in snapshots if item.record_count > 0]
    report = build_seed_scheduler_calibration_report(
        snapshots,
        commit_sha=git_head_sha(REPO_ROOT),
        seconds_total=__import__("time").perf_counter() - started,
        analyze_only=bool(args.analyze_only),
        eval_seeds=eval_seeds,
        train_seed=int(args.train_seed),
        required_opponents=opponents,  # type: ignore[arg-type]
    )
    write_seed_scheduler_calibration_report(args.out, report)
    md_lines = [
        "# Seed scheduler calibration",
        "",
        f"Source JSON: `{args.out}`",
        "",
        "## Decision",
        "",
        f"```json\n{json.dumps(report.get('decision', {}), indent=2)}\n```",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run ow benchmark calibrate-seed-scheduler \\",
        f"  --opponents {','.join(opponents)} \\",
        f"  --reseed-intervals {','.join(str(v) for v in reseed_intervals)} \\",
        f"  --total-updates {int(args.total_updates)}",
        "```",
        "",
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0
