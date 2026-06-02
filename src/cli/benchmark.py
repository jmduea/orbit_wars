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
        "  calibrate-seed-scheduler Reseed-interval calibration\n"
        "  gate                     Run one preflight gate from conf/benchmark/gates/\n"
        "  shortlist-planet-flow-sweep  Rank finished Planet Flow W&B sweep runs\n"
        "  planet-flow-noop-smoke   Noop smoke on shortlist top-K before learn-proof\n\n"
        "Examples:\n"
        "  make preflight-sanity\n"
        "  make preflight-learn-proof\n"
        "  uv run ow benchmark calibrate --analyze-only --analyze-campaigns\n"
        "  uv run ow benchmark gate --list\n"
        "  uv run ow benchmark gate beat_noop --dry-run\n"
        "  uv run ow benchmark gate beat_random --dry-run\n\n"
        "E2E throughput (launch hygiene):\n"
        "  make test-launch-hygiene-e2e-throughput\n"
        "  uv run ow benchmark training --preset primary --label gate --out /tmp/gate.json \\\n"
        "    --baseline docs/benchmarks/launch-hygiene-e2e-baseline.json --assert-within-pct 10\n\n"
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
        choices=("validation", "primary", "planet_flow_p0"),
        default=None,
        help=(
            "Benchmark bundle: validation (workstation stability), primary "
            "(task=shield_cheap e2e throughput gate profile), or "
            "planet_flow_p0 (Planet Flow compiler-control proof)."
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
            "Exit non-zero when measured env_steps/s, samples/s, or "
            "seconds/update exceed baseline pass band (default from baseline or 10)."
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
        "--assert-min-env-steps-per-sec",
        type=float,
        default=None,
        help="Fail if measured env_steps_per_sec is below this floor (single-run).",
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
    training.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Write a JAX trace with named benchmark regions to this directory.",
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
        "--thresholds-path",
        type=Path,
        default=None,
        help="Optional preflight calibration JSON for learning gates.",
    )
    learn_proof.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Optional preflight-profiles.json (default docs/benchmarks/preflight-profiles.json).",
    )
    learn_proof.add_argument(
        "--train-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides appended after gate and profile overrides.",
    )
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

    shortlist_pf = subparsers.add_parser(
        "shortlist-planet-flow-sweep",
        help="Deterministic Planet Flow sweep shortlist (window-mean KL guardrails).",
    )
    shortlist_pf.add_argument("--sweep-id", required=True)
    shortlist_pf.add_argument("--entity", default="jmduea-jdueadev")
    shortlist_pf.add_argument("--project", default="planet-flow-policy")
    shortlist_pf.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/preflight/planet_flow_sweep_shortlist.json"),
    )
    shortlist_pf.add_argument(
        "--max-kl",
        type=float,
        default=None,
        help="Window-mean KL ceiling (default 0.15).",
    )
    shortlist_pf.add_argument(
        "--min-entropy",
        type=float,
        default=None,
        help="Window-mean entropy floor (default 1e-3).",
    )
    shortlist_pf.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max eligible entries in ranked output.",
    )

    noop_smoke = subparsers.add_parser(
        "planet-flow-noop-smoke",
        help="200-update noop trains on shortlist top-K; beat_noop gate check.",
    )
    noop_smoke.add_argument(
        "--shortlist",
        type=Path,
        required=True,
        help="JSON from shortlist-planet-flow-sweep.",
    )
    noop_smoke.add_argument("--top-k", type=int, default=3)
    noop_smoke.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/preflight/planet_flow_noop_smoke.json"),
    )
    noop_smoke.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    noop_smoke.add_argument("--thresholds-path", type=Path, default=None)
    noop_smoke.add_argument("--dry-run", action="store_true")

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
    calibrate.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Optional preflight-profiles.json for per-model PPO overrides.",
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

    gate = subparsers.add_parser(
        "gate",
        help="Run one composable preflight gate (YAML recipe in conf/benchmark/gates/).",
    )
    gate.add_argument(
        "gate_id",
        nargs="?",
        default=None,
        help="Gate id matching conf/benchmark/gates/<id>.yaml (omit with --list).",
    )
    gate.add_argument(
        "--list",
        action="store_true",
        help="List gate YAML recipes.",
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
    gate.add_argument("--dry-run", action="store_true")
    gate.add_argument("--thresholds-path", type=Path, default=None)
    gate.add_argument("--profile-path", type=Path, default=None)
    gate.add_argument(
        "--train-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides appended after gate overrides.",
    )

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
        E2E_THROUGHPUT_GATE,
        aggregate_e2e_run_payloads,
        check_baseline_device_match,
        compare_e2e_throughput_to_baseline,
        compose_benchmark_config,
        default_benchmark_updates,
        derive_e2e_pass_band,
        e2e_throughput_metric_values,
        format_profile_name,
        load_e2e_baseline,
        resolve_benchmark_overrides,
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
            profile_dir=args.profile_dir,
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

    if repeats == 1 and args.preset == "planet_flow_p0":
        required_control_metrics = (
            "planet_flow_control_emitted_launch_count",
            "planet_flow_control_emitted_ship_mass_rate",
            "planet_flow_emitted_launch_count_delta_vs_control",
        )
        missing = [
            key
            for key in required_control_metrics
            if run_payloads[0].get(key) is None
        ]
        if missing:
            print(
                "Planet Flow benchmark proof is missing compiler-control metrics: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 1

    if args.baseline is not None or args.assert_within_pct is not None:
        if args.baseline is None:
            print("--assert-within-pct requires --baseline", file=sys.stderr)
            return 1
        try:
            baseline = load_e2e_baseline(args.baseline)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        measured_source = run_payloads[0] if repeats == 1 else output_payload
        measured = e2e_throughput_metric_values(measured_source)
        if repeats > 1:
            aggregate_obj = output_payload.get("aggregate")
            if isinstance(aggregate_obj, dict):
                measured = {
                    key: float(aggregate_obj[key]["mean"])  # type: ignore[index]
                    for key in measured
                    if key in aggregate_obj
                }
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
    if (
        repeats == 1
        and args.assert_min_env_steps_per_sec is not None
        and float(run_payloads[0]["env_steps_per_sec"])
        < args.assert_min_env_steps_per_sec
    ):
        print(
            "env_steps_per_sec "
            f"{float(run_payloads[0]['env_steps_per_sec']):.3f} < "
            f"{args.assert_min_env_steps_per_sec:.3f}",
            file=sys.stderr,
        )
        return 1
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
    raw_games_per_pair = win_proof.get("games_per_pair")
    games_per_pair = (
        int(raw_games_per_pair)
        if isinstance(raw_games_per_pair, int | float | str)
        else int(args.games_per_pair)
    )
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

    extra_train_overrides = tuple(args.train_overrides)
    started = __import__("time").perf_counter()
    if args.gate is not None:
        evaluation = run_preflight_gate(
            args.gate,
            model=args.model,
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            dry_run=args.dry_run,
            thresholds_path=args.thresholds_path,
            profiles_path=args.profile_path,
            extra_train_overrides=extra_train_overrides,
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
            thresholds_path=args.thresholds_path,
            profiles_path=args.profile_path,
            extra_train_overrides=extra_train_overrides,
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


def run_gate_cli(args: argparse.Namespace) -> int:
    from src.cli.benchmark_gates import list_gate_recipes, run_gate_cli as run_gate

    if args.list or args.gate_id is None:
        payload = {"gates": list_gate_recipes()}
        print(json.dumps(payload, indent=2))
        return 0
    return run_gate(
        args.gate_id,
        model=args.model,
        output_root=args.output_root,
        dry_run=bool(args.dry_run),
        thresholds_path=args.thresholds_path,
        profiles_path=args.profile_path,
        train_overrides=tuple(args.train_overrides),
        out=args.out,
    )


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
        case "shortlist-planet-flow-sweep":
            return run_shortlist_planet_flow_sweep_cli(args)
        case "planet-flow-noop-smoke":
            return run_planet_flow_noop_smoke_cli(args)
        case "gate":
            return run_gate_cli(args)
        case _:
            parser.error(f"unknown benchmark command: {args.command!r}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
