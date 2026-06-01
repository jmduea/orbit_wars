"""``ow benchmark`` CLI for stability benchmarks and pre-flight learning gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def print_benchmark_help() -> None:
    print(
        "ow benchmark — stability runs and preflight gates\n\n"
        "Subcommands:\n"
        "  training                 Short timed training benchmark\n"
        "  sanity                   Gate 1 reproducibility\n"
        "  learn-proof              Gates 2–5 learning proof ladder\n"
        "  calibrate                Derive preflight thresholds\n"
        "  calibrate-seed-scheduler Reseed-interval calibration\n\n"
        "Examples:\n"
        "  make preflight-sanity\n"
        "  make preflight-learn-proof\n"
        "  uv run ow benchmark calibrate --analyze-only --analyze-campaigns\n\n"
        "More: uv run ow benchmark learn-proof --help\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stability benchmarks and pre-flight learning gates (ow benchmark).",
    )
    subparsers = parser.add_subparsers(dest="command")

    training = subparsers.add_parser(
        "training",
        help="Gate stability: short production-path benchmark (self-play OK, not learnability).",
    )
    training.add_argument("--label", required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Hydra overrides; with --preset, merged after the preset bundle.",
    )
    training.add_argument(
        "--preset",
        choices=("validation",),
        default=None,
        help="Workstation stability bundle (--preset validation).",
    )
    training.add_argument(
        "--tier",
        default="micro",
        help="Label recorded in JSON (e.g. micro, workstation).",
    )
    training.add_argument("--updates", type=int, default=30)
    training.add_argument("--warmup", type=int, default=2)
    training.add_argument(
        "--snapshot-updates",
        nargs="*",
        type=int,
        default=[],
        help="Per-update metric snapshots at these global update indices.",
    )

    sanity = subparsers.add_parser(
        "sanity",
        help="Gate 1: reproducibility via paired training-benchmark snapshots.",
    )
    sanity.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/preflight/sanity_repro.json"),
    )
    sanity.add_argument("--updates", type=int, default=10)
    sanity.add_argument("--warmup", type=int, default=2)
    sanity.add_argument("--compare-update", type=int, default=10)
    sanity.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Hydra overrides (default: WORKSTATION_VALIDATION_OVERRIDES).",
    )

    learn_proof = subparsers.add_parser(
        "learn-proof",
        help="Gates 2–4: ow train scripted-opponent / curriculum runs with JSONL assertions.",
    )
    learn_proof.add_argument(
        "--gate",
        choices=("beat_noop", "beat_random", "curriculum_staged"),
        default=None,
        help="Run a single gate (default with --through: beat_noop → beat_random).",
    )
    learn_proof.add_argument(
        "--through",
        choices=("beat_noop", "beat_random", "curriculum_staged"),
        default=None,
        help="Run gates in order through this gate id (inclusive).",
    )
    learn_proof.add_argument(
        "--model",
        default="transformer_factorized_small",
        help="Model for beat_noop / beat_random gates.",
    )
    learn_proof.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/preflight/learn_proof_report.json"),
    )
    learn_proof.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    learn_proof.add_argument("--dry-run", action="store_true")
    learn_proof.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Gate 5: held-out tournament eval vs --baselines (requires checkpoint).",
    )
    learn_proof.add_argument(
        "--baselines",
        default="random",
        help="Comma-separated baselines for --eval-checkpoint (Gate 5).",
    )
    learn_proof.add_argument("--campaign", default="preflight_held_out")
    learn_proof.add_argument("--seeds", default="0,1,2,3,4")
    learn_proof.add_argument("--games-per-pair", type=int, default=4)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Short sweep to derive JAX learning-signal and tournament win-proof thresholds.",
    )
    calibrate.add_argument(
        "--out",
        type=Path,
        default=Path("docs/benchmarks/preflight-calibration.json"),
    )
    calibrate.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    calibrate.add_argument(
        "--model",
        default="transformer_factorized_small",
    )
    calibrate.add_argument("--seeds", default="42,43")
    calibrate.add_argument("--updates", default="200,500")
    calibrate.add_argument(
        "--opponents",
        default="noop_only,random_only",
        help="Comma-separated opponent profiles.",
    )
    calibrate.add_argument(
        "--analyze-jsonl",
        nargs="*",
        default=[],
        help="Existing *_jax.jsonl paths (path:opponent:seed:updates).",
    )
    calibrate.add_argument(
        "--analyze-campaigns",
        nargs="?",
        const="preflight_calibrate_*",
        default=None,
        metavar="GLOB",
        help=(
            "Discover completed runs under output-root/campaigns/GLOB "
            "(default: preflight_calibrate_*)."
        ),
    )
    calibrate.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip training; analyze --analyze-jsonl and/or --analyze-campaigns.",
    )
    calibrate.add_argument("--dry-run", action="store_true")

    seed_sched = subparsers.add_parser(
        "calibrate-seed-scheduler",
        help="Sweep reseed intervals and evaluate held-out seed generalization.",
    )
    seed_sched.add_argument(
        "--out",
        type=Path,
        default=Path("docs/benchmarks/seed-scheduler-calibration.json"),
    )
    seed_sched.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/benchmarks/seed-scheduler-calibration.md"),
    )
    seed_sched.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    seed_sched.add_argument(
        "--opponents",
        default="noop_only,random_only,self_play_only",
    )
    seed_sched.add_argument(
        "--reseed-intervals",
        default="0,25,50,100",
        help="Comma-separated training.reseed_every_updates values.",
    )
    seed_sched.add_argument(
        "--no-include-total-fifth",
        action="store_true",
        help="Do not append total_updates//5 to the interval grid.",
    )
    seed_sched.add_argument("--total-updates", type=int, default=500)
    seed_sched.add_argument("--train-seed", type=int, default=42)
    seed_sched.add_argument("--eval-seeds", default="0,1,2,3,4,43,44,45,46")
    seed_sched.add_argument("--baseline", default="noop")
    seed_sched.add_argument("--games-per-pair", type=int, default=4)
    seed_sched.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip training; analyze existing seed_sched_cal_* campaigns.",
    )
    seed_sched.add_argument(
        "--eval-existing",
        action="store_true",
        help="With --analyze-only, run tournament eval on discovered checkpoints.",
    )
    seed_sched.add_argument("--dry-run", action="store_true")

    return parser


def _init_benchmark_runtime() -> None:
    from src.jax.device import (
        configure_jax_runtime_for_host,
        ensure_jax_accelerator_backend,
    )

    configure_jax_runtime_for_host()
    ensure_jax_accelerator_backend()


def run_training_benchmark_cli(args: argparse.Namespace) -> int:
    import jax
    from src.jax.benchmark import rollout_group_summary
    from src.jax.training_benchmark import (
        compose_benchmark_config,
        format_profile_name,
        resolve_benchmark_overrides,
        run_training_benchmark,
        training_benchmark_payload,
    )

    _init_benchmark_runtime()
    overrides = resolve_benchmark_overrides(
        preset=args.preset,
        overrides=args.overrides,
    )
    cfg = compose_benchmark_config(overrides)
    group_specs = rollout_group_summary(cfg)
    result = run_training_benchmark(
        cfg,
        label=args.label,
        overrides=tuple(overrides),
        warmup=args.warmup,
        updates=args.updates,
        snapshot_updates=frozenset(args.snapshot_updates),
    )
    payload = training_benchmark_payload(result)
    payload.update(
        {
            "commit_sha": _git_head_sha(),
            "tier": args.tier,
            "jax_version": jax.__version__,
            "format": format_profile_name(overrides),
            "rollout_groups": [dict(group) for group in group_specs],
            "rollout_microbatch_envs": int(cfg.training.rollout_microbatch_envs),
            "gate": "stability",
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_sanity_cli(args: argparse.Namespace) -> int:
    import jax
    from src.jax.preflight import (
        PreflightVerdict,
        compare_repro_snapshots,
        write_report,
    )
    from src.jax.training_benchmark import (
        compose_benchmark_config,
        run_training_benchmark,
        training_benchmark_payload,
    )

    _init_benchmark_runtime()
    from src.jax.training_benchmark import WORKSTATION_VALIDATION_OVERRIDES

    overrides = (
        list(args.overrides)
        if args.overrides is not None
        else list(WORKSTATION_VALIDATION_OVERRIDES)
    )
    cfg = compose_benchmark_config(overrides)
    snapshot_updates = frozenset({args.compare_update})
    first = run_training_benchmark(
        cfg,
        label="sanity_repro_a",
        overrides=tuple(overrides),
        warmup=args.warmup,
        updates=args.updates,
        snapshot_updates=snapshot_updates,
    )
    second = run_training_benchmark(
        cfg,
        label="sanity_repro_b",
        overrides=tuple(overrides),
        warmup=args.warmup,
        updates=args.updates,
        snapshot_updates=snapshot_updates,
    )
    verdict, reasons = compare_repro_snapshots(
        training_benchmark_payload(first),
        training_benchmark_payload(second),
        update=args.compare_update,
    )
    report: dict[str, object] = {
        "gate": "sanity_repro",
        "commit_sha": _git_head_sha(),
        "jax_version": jax.__version__,
        "verdict": verdict.value,
        "reasons": list(reasons),
        "compare_update": args.compare_update,
        "run_a": training_benchmark_payload(first),
        "run_b": training_benchmark_payload(second),
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if verdict == PreflightVerdict.VERIFIED else 1


def _run_held_out_eval(args: argparse.Namespace) -> int:
    from src.jax.preflight import PreflightVerdict, write_report
    from src.jax.preflight_calibration import (
        default_calibration_json_path,
        load_thresholds,
    )

    baselines = [part.strip() for part in args.baselines.split(",") if part.strip()]
    baseline = baselines[0] if baselines else "random"
    thresholds = load_thresholds(default_calibration_json_path(REPO_ROOT))
    win_proof = thresholds.get("win_proof_tournament", {})
    if not isinstance(win_proof, dict):
        win_proof = {}
    min_win_rate_key = (
        "noop_min_win_rate"
        if baseline in {"noop", "noop_only"}
        else "random_min_win_rate"
    )
    min_win_rate = float(win_proof.get(min_win_rate_key, 0.45))
    games_per_pair = int(win_proof.get("games_per_pair", args.games_per_pair))
    seeds = str(win_proof.get("seeds", args.seeds))

    output_dir = (
        args.output_root
        / "campaigns"
        / args.campaign
        / "evaluations"
        / f"preflight_win_proof_{baseline}"
    )
    cmd = [
        "uv",
        "run",
        "ow",
        "eval",
        "tournament",
        "--checkpoint",
        str(args.eval_checkpoint),
        "--campaign",
        args.campaign,
        "--output-root",
        str(args.output_root),
        "--output-dir",
        str(output_dir),
        "--seeds",
        seeds,
        "--games-per-pair",
        str(games_per_pair),
        "--formats",
        "2p_vs_baseline",
        "--baselines",
        baseline,
    ]
    if args.dry_run:
        print(" ".join(cmd), flush=True)
        report = {
            "gate": "win_proof",
            "verdict": PreflightVerdict.INCONCLUSIVE.value,
            "baseline": baseline,
            "min_win_rate": min_win_rate,
            "dry_run": True,
        }
        write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 0

    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    leaderboard_path = output_dir / "leaderboard.json"
    if not leaderboard_path.is_file():
        print(f"missing leaderboard: {leaderboard_path}", file=sys.stderr)
        return 1
    rows = json.loads(leaderboard_path.read_text(encoding="utf-8")).get("rows", [])
    if not rows:
        print("leaderboard has no rows", file=sys.stderr)
        return 1
    observed = rows[0].get("win_rate_vs_baseline")
    if observed is None:
        observed = rows[0].get("win_rate_vs_sniper")
    baseline_name = rows[0].get("baseline_name") or baseline
    verdict = PreflightVerdict.VERIFIED
    reasons: list[str] = []
    if observed is None:
        verdict = PreflightVerdict.INCONCLUSIVE
        reasons.append("missing win_rate_vs_baseline in tournament leaderboard")
    elif float(observed) < min_win_rate:
        verdict = PreflightVerdict.NOT_VERIFIED
        reasons.append(
            f"tournament win rate {float(observed):.3f} < {min_win_rate:.3f} "
            f"vs {baseline_name}"
        )

    report = {
        "gate": "win_proof",
        "commit_sha": _git_head_sha(),
        "verdict": verdict.value,
        "reasons": reasons,
        "baseline_name": baseline_name,
        "min_win_rate": min_win_rate,
        "observed_win_rate": observed,
        "leaderboard_path": str(leaderboard_path),
        "checkpoint": str(args.eval_checkpoint),
        "evaluation_mode": "tournament",
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if verdict == PreflightVerdict.VERIFIED else 1


def run_calibrate_cli(args: argparse.Namespace) -> int:
    from src.jax.preflight_calibration import (
        analyze_jsonl_path,
        build_calibration_report,
        derive_thresholds,
        discover_calibration_snapshots,
        git_head_sha,
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
    print(json.dumps(report, indent=2))
    return 0


def run_calibrate_seed_scheduler_cli(args: argparse.Namespace) -> int:
    from src.jax.preflight_calibration import git_head_sha
    from src.jax.seed_scheduler_calibration import (
        DEFAULT_OPPONENTS,
        analyze_seed_sched_run,
        build_seed_scheduler_calibration_report,
        discover_seed_sched_runs,
        expand_reseed_intervals,
        run_seed_scheduler_sweep,
        write_seed_scheduler_calibration_report,
    )

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


def run_learn_proof_cli(args: argparse.Namespace) -> int:
    from src.jax.preflight import (
        GATE_ORDER,
        PreflightVerdict,
        gate_evaluation_to_dict,
        run_preflight_gate,
        run_preflight_ladder,
        write_report,
    )

    if args.eval_checkpoint is not None:
        return _run_held_out_eval(args)

    if args.gate is not None and args.through is not None:
        raise SystemExit("Use only one of --gate or --through.")

    started = __import__("time").perf_counter()
    if args.gate is not None:
        evaluation = run_preflight_gate(
            args.gate,
            model=args.model,
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            dry_run=args.dry_run,
        )
        overall = evaluation.verdict
        stages = [evaluation]
    else:
        through = args.through or "beat_random"
        overall, stages = run_preflight_ladder(
            through=through,
            model=args.model,
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            dry_run=args.dry_run,
        )

    report: dict[str, object] = {
        "gate": "learn_proof",
        "commit_sha": _git_head_sha(),
        "seconds_total": __import__("time").perf_counter() - started,
        "verdict": overall.value,
        "through": args.through or args.gate or "beat_random",
        "model": args.model,
        "gate_order": list(GATE_ORDER),
        "stages": [gate_evaluation_to_dict(item) for item in stages],
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if overall == PreflightVerdict.VERIFIED else 1


def main(argv: list[str] | None = None) -> int:
    if not argv:
        print_benchmark_help()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        print_benchmark_help()
        return 0
    match args.command:
        case "training":
            return run_training_benchmark_cli(args)
        case "sanity":
            return run_sanity_cli(args)
        case "learn-proof":
            return run_learn_proof_cli(args)
        case "calibrate":
            return run_calibrate_cli(args)
        case "calibrate-seed-scheduler":
            return run_calibrate_seed_scheduler_cli(args)
        case _:
            parser.error(f"unknown benchmark command: {args.command!r}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
