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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stability benchmarks and pre-flight learning gates (ow benchmark).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        choices=("validation", "primary", "admission", "planet_flow_p0"),
        default=None,
        help=(
            "Benchmark bundle: validation (workstation stability), primary "
            "(task=shield_cheap e2e throughput gate profile), admission "
            "(operator-locked beat_noop + admission.yaml recipe; append "
            "task=map_pool etc. via --overrides), or planet_flow_p0 (Planet Flow "
            "compiler-control proof)."
        ),
    )
    training.add_argument(
        "--tier",
        default="micro",
        help="Label recorded in JSON (e.g. micro, workstation).",
    )
    training.add_argument("--updates", type=int, default=None)
    training.add_argument("--warmup", type=int, default=2)
    training.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat measured runs; with --out writes aggregate baseline-style payload.",
    )
    training.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON for --assert-within-pct comparison.",
    )
    training.add_argument(
        "--assert-within-pct",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Exit non-zero when measured env_steps/s is below the baseline floor "
            "or seconds/update exceeds the ceiling (default band from baseline or 10%%). "
            "samples/s is recorded but not gated."
        ),
    )
    training.add_argument(
        "--device-check",
        choices=("warn", "strict"),
        default="warn",
        help="Compare JAX devices to baseline fingerprint (default: warn).",
    )
    training.add_argument(
        "--force",
        action="store_true",
        help="Compare throughput even when --device-check detects mismatch.",
    )
    training.add_argument(
        "--snapshot-updates",
        nargs="*",
        type=int,
        default=[],
        help="Per-update metric snapshots at these global update indices.",
    )
    training.add_argument(
        "--detailed-timing",
        action="store_true",
        help=(
            "Synchronize at rollout/PPO boundaries and emit timing buckets. "
            "Use for profiling only; it adds extra barriers."
        ),
    )

    factorized_sampler = subparsers.add_parser(
        "factorized-sampler",
        help=(
            "Tier-1 factorized shield sampler microbenchmark "
            "(in-process JAX via src/benchmark/factorized_sampler.py)."
        ),
    )
    factorized_sampler.add_argument("--max-moves-k", type=int, default=3)
    factorized_sampler.add_argument(
        "--decoder-carry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    factorized_sampler.add_argument("--batch-size", type=int, default=16)
    factorized_sampler.add_argument("--warmup", type=int, default=3)
    factorized_sampler.add_argument("--repeats", type=int, default=30)
    factorized_sampler.add_argument("--assert-max-ms", type=float, default=None)

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

    gate = subparsers.add_parser(
        "gate",
        help="Composable preflight gates (YAML in conf/benchmark/gates/).",
    )
    gate.add_argument(
        "tokens",
        nargs="*",
        default=[],
        help=(
            "`list`, `run <id>`, or legacy `<id>` "
            "(admission, beat_noop, beat_random, curriculum_staged)."
        ),
    )
    gate.add_argument(
        "--list",
        action="store_true",
        help="List gate YAML recipes (alias for `gate list`).",
    )
    gate.add_argument(
        "--model",
        default=None,
        help="Model override (default from gate YAML).",
    )
    gate.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    gate.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    gate.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Run ``ow train`` in another checkout (e.g. a git worktree) while gate "
            "recipes and thresholds load from this repo."
        ),
    )
    gate.add_argument("--dry-run", action="store_true")
    gate.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Extra stderr progress (train overrides, hints). Training subprocess "
            "output always streams on stderr."
        ),
    )
    gate.add_argument("--thresholds-path", type=Path, default=None)
    gate.add_argument("--profile-path", type=Path, default=None)
    gate.add_argument(
        "--train-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides appended after gate overrides.",
    )
    gate.add_argument(
        "--also-throughput",
        action="store_true",
        help=(
            "After the learning run, extract throughput from the gate JSONL "
            "(updates 3–22; 20 measured rows after warmup) and merge into --out JSON. "
            "Prefer `gate run admission`."
        ),
    )
    gate.add_argument(
        "--throughput-baseline",
        type=Path,
        default=None,
        help="Baseline JSON for --also-throughput comparison (with --throughput-within-pct).",
    )
    gate.add_argument(
        "--throughput-within-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Exit non-zero when extracted throughput is outside baseline band (default 10).",
    )

    rollout_phase_profile = subparsers.add_parser(
        "rollout-phase-profile",
        help=(
            "Offline admission-shaped rollout phase profile (short run; "
            "host-timed collect — do not use on gate spine)."
        ),
    )
    rollout_phase_profile.add_argument(
        "--preset",
        choices=("admission",),
        default="admission",
        help="Override bundle (default: operator-locked admission recipe).",
    )
    rollout_phase_profile.add_argument(
        "--full-geometry",
        action="store_true",
        help=(
            "Use full admission env geometry (32 envs × 256 steps). "
            "Default is --quick (4 envs × 16 steps) for interactive profiling."
        ),
    )
    rollout_phase_profile.add_argument(
        "--train-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides (e.g. task=map_pool).",
    )
    rollout_phase_profile.add_argument("--model", default=None)
    rollout_phase_profile.add_argument("--updates", type=int, default=5)
    rollout_phase_profile.add_argument("--warmup", type=int, default=2)
    rollout_phase_profile.add_argument(
        "--max-measured-update",
        type=int,
        default=20,
        help="Last update index included in printed breakdown.",
    )
    rollout_phase_profile.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    rollout_phase_profile.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for summary JSON (includes per-update phase rows).",
    )

    rollout_phase_breakdown = subparsers.add_parser(
        "rollout-phase-breakdown",
        help=(
            "Print rollout collect phase breakdown from gate JSON or *_jax.jsonl "
            "(requires telemetry=rollout_phase_timing)."
        ),
    )
    rollout_phase_breakdown.add_argument(
        "input",
        type=Path,
        help="Gate result JSON (reads stage.log_path) or logs/*_jax.jsonl path.",
    )
    rollout_phase_breakdown.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    rollout_phase_breakdown.add_argument("--warmup", type=int, default=2)
    rollout_phase_breakdown.add_argument("--max-measured-update", type=int, default=20)

    from src.benchmark.map_pool import build_map_pool_parser

    build_map_pool_parser(subparsers)

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
    from src.benchmark.production import rollout_group_summary
    from src.benchmark.training import (
        E2E_THROUGHPUT_GATE,
        aggregate_e2e_run_payloads,
        check_baseline_device_match,
        compare_e2e_throughput_to_baseline,
        compose_benchmark_config,
        default_benchmark_updates,
        derive_e2e_pass_band,
        format_profile_name,
        load_e2e_baseline,
        resolve_benchmark_overrides,
        resolve_e2e_measured_for_gate,
        resolve_e2e_pass_band,
        run_training_benchmark,
        training_benchmark_payload,
    )

    _init_benchmark_runtime()
    updates = (
        int(args.updates)
        if args.updates is not None
        else default_benchmark_updates(preset=args.preset)
    )
    overrides = resolve_benchmark_overrides(
        preset=args.preset,
        overrides=args.overrides,
    )
    cfg = compose_benchmark_config(overrides)
    group_specs = rollout_group_summary(cfg)

    run_payloads: list[dict[str, object]] = []
    repeats = max(int(args.repeats), 1)
    for repeat_idx in range(repeats):
        label = args.label if repeats == 1 else f"{args.label}_r{repeat_idx + 1}"
        result = run_training_benchmark(
            cfg,
            label=label,
            overrides=tuple(overrides),
            warmup=args.warmup,
            updates=updates,
            snapshot_updates=frozenset(args.snapshot_updates),
            detailed_timing=bool(args.detailed_timing),
        )
        payload = training_benchmark_payload(result)
        payload.update(
            {
                "commit_sha": _git_head_sha(),
                "tier": args.tier,
                "jax_version": jax.__version__,
                "format": format_profile_name(overrides),
                "rollout_groups": [dict(group) for group in group_specs],
                "rollout_microbatch_envs": (
                    int(cfg.training.rollout_microbatch_envs)
                    if cfg.training.rollout_microbatch_envs is not None
                    else None
                ),
                "gate": E2E_THROUGHPUT_GATE
                if args.preset == "primary"
                else "stability",
            }
        )
        run_payloads.append(payload)

    if repeats == 1:
        output_payload: dict[str, object] = run_payloads[0]
    else:
        aggregate = aggregate_e2e_run_payloads(run_payloads)
        within_pct = (
            float(args.assert_within_pct)
            if args.assert_within_pct is not None
            else 10.0
        )
        pass_band = derive_e2e_pass_band(aggregate, within_pct=within_pct)
        output_payload = {
            "gate": E2E_THROUGHPUT_GATE,
            "label": args.label,
            "commit_sha": _git_head_sha(),
            "jax_version": jax.__version__,
            "overrides": overrides,
            "updates": updates,
            "warmup": args.warmup,
            "repeats": repeats,
            "runs": run_payloads,
            "aggregate": aggregate,
            "pass_band": pass_band,
        }

    if args.baseline is not None or args.assert_within_pct is not None:
        if args.baseline is None:
            print("--assert-within-pct requires --baseline", file=sys.stderr)
            return 1
        try:
            baseline = load_e2e_baseline(args.baseline)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        measured = resolve_e2e_measured_for_gate(
            repeats=repeats,
            run_payloads=run_payloads,
            aggregate=output_payload.get("aggregate"),
        )
        device_ok, device_message = check_baseline_device_match(
            baseline,
            devices=run_payloads[0]["devices"],  # type: ignore[arg-type]
            default_backend=str(run_payloads[0]["default_backend"]),
            mode=str(args.device_check),
            force=bool(args.force),
        )
        if device_message:
            print(device_message, file=sys.stderr)
        if not device_ok:
            return 1
        pass_band = resolve_e2e_pass_band(
            baseline,
            within_pct=args.assert_within_pct,
        )
        passed, failures = compare_e2e_throughput_to_baseline(
            measured,
            pass_band=pass_band,
        )
        output_payload["baseline_path"] = str(args.baseline)
        output_payload["pass_band_applied"] = pass_band
        output_payload["measured_for_gate"] = measured
        output_payload["gate_passed"] = passed
        if failures:
            output_payload["gate_failures"] = failures
        if not passed:
            for reason in failures:
                print(reason, file=sys.stderr)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(output_payload, indent=2) + "\n", encoding="utf-8"
            )
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_payload, sort_keys=True))
    return 0


def run_factorized_sampler_cli(args: argparse.Namespace) -> int:
    from src.benchmark.factorized_sampler import run_factorized_sampler_benchmark

    return run_factorized_sampler_benchmark(
        max_moves_k=int(args.max_moves_k),
        decoder_carry=bool(args.decoder_carry),
        batch_size=int(args.batch_size),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        assert_max_ms=args.assert_max_ms,
    )


def run_sanity_cli(args: argparse.Namespace) -> int:
    import jax
    from src.benchmark.training import (
        compose_benchmark_config,
        run_training_benchmark,
        training_benchmark_payload,
    )
    from src.jax.preflight import (
        PreflightVerdict,
        compare_repro_snapshots,
        write_report,
    )

    _init_benchmark_runtime()
    from src.benchmark.training import WORKSTATION_VALIDATION_OVERRIDES

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


def run_rollout_phase_profile_cli(args: argparse.Namespace) -> int:
    from src.benchmark.rollout_phase_profile import (
        compose_profile_config,
        format_profile_report,
        profile_result_payload,
        resolve_profile_overrides,
        run_rollout_phase_profile,
    )
    from src.jax.rollout.phase_timing_report import PhaseTimingWindow

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
    if not quick:
        print(
            "warning: --full-geometry uses host-timed 32×256 collect; "
            "first update may take 30+ minutes",
            file=sys.stderr,
            flush=True,
        )
    result = run_rollout_phase_profile(
        cfg,
        warmup=int(args.warmup),
        updates=int(args.updates),
        window=PhaseTimingWindow(
            warmup=int(args.warmup),
            max_measured_update=int(args.max_measured_update),
        ),
    )
    payload = profile_result_payload(
        result,
        overrides=overrides,
        preset=args.preset,
    )
    payload["geometry_mode"] = "full" if args.full_geometry else "quick"
    payload["per_update_records"] = list(result.per_update_records)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_profile_report(payload), flush=True)
    return 0


def run_rollout_phase_breakdown_cli(args: argparse.Namespace) -> int:
    from src.jax.rollout.phase_timing_report import (
        PhaseTimingWindow,
        extract_rollout_phase_breakdown_from_input,
        format_rollout_phase_breakdown,
    )

    input_path = Path(args.input)
    try:
        window = PhaseTimingWindow(
            warmup=int(args.warmup),
            max_measured_update=int(args.max_measured_update),
        )
        payload = extract_rollout_phase_breakdown_from_input(input_path, window=window)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_rollout_phase_breakdown(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    match args.command:
        case "training":
            return run_training_benchmark_cli(args)
        case "sanity":
            return run_sanity_cli(args)
        case "learn-proof":
            return run_learn_proof_cli(args)
        case "calibrate":
            return run_calibrate_cli(args)
        case "gate":
            from src.cli.benchmark_gate_cli import run_gate_cli

            return run_gate_cli(args)
        case "factorized-sampler":
            return run_factorized_sampler_cli(args)
        case "map-pool":
            from src.benchmark.map_pool import dispatch_map_pool

            return dispatch_map_pool(args)
        case "rollout-phase-profile":
            return run_rollout_phase_profile_cli(args)
        case "rollout-phase-breakdown":
            return run_rollout_phase_breakdown_cli(args)
        case _:
            parser.error(f"unknown benchmark command: {args.command!r}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
