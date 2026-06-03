"""``ow benchmark`` CLI for stability benchmarks and pre-flight learning gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LEARN_PROOF_PRIMITIVES: tuple[str, ...] = (
    "ow benchmark gate run beat_noop",
    "ow benchmark gate run beat_random",
    "ow benchmark gate run curriculum_staged",
    "ow benchmark tournament-proof --eval-checkpoint <ckpt>",
)


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
        "  gate                     Composable preflight gates (run/list)\n"
        "  tournament-proof         Gate 5 held-out tournament win proof\n"
        "  shortlist-planet-flow-sweep  Rank finished Planet Flow W&B sweep runs\n"
        "  planet-flow-noop-smoke   Noop smoke on shortlist top-K before learn-proof\n"
        "  factorized-sampler     Tier-1 launch-hygiene microbenchmark (script wrapper)\n\n"
        "Examples:\n"
        "  make preflight-sanity\n"
        "  make preflight-learn-proof\n"
        "  uv run ow benchmark calibrate --analyze-only --analyze-campaigns\n"
        "  uv run ow benchmark gate --list\n"
        "  uv run ow benchmark gate run beat_noop --dry-run\n"
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
        "--steps",
        default=None,
        metavar="GATES",
        help="Comma-separated gate ids to run in ladder order (e.g. beat_noop,beat_random).",
    )
    learn_proof.add_argument(
        "--print-primitives",
        action="store_true",
        help="Print primitive command chain JSON and exit (no training).",
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

    factorized_sampler = subparsers.add_parser(
        "factorized-sampler",
        help="Tier-1 factorized shield sampler microbenchmark (delegates to scripts/).",
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

    unified_cal = subparsers.add_parser(
        "calibrate-unified-tournament",
        help="Unified Stage-1 calibration sweep (games-per-pair + combined floors).",
    )
    unified_cal.add_argument(
        "--out",
        type=Path,
        default=Path("docs/benchmarks/preflight-calibration.json"),
        help="Preflight calibration JSON to merge unified_tournament section into.",
    )
    unified_cal.add_argument(
        "--artifact-out",
        type=Path,
        default=Path("docs/benchmarks/unified-tournament-calibration.json"),
        help="Full calibration campaign report JSON.",
    )
    unified_cal.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    unified_cal.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Representative checkpoint for calibration campaign (repeatable).",
    )
    unified_cal.add_argument(
        "--games-per-pair",
        default="2,4,8",
        help="Comma-separated games-per-pair values to sweep during calibration.",
    )
    unified_cal.add_argument("--analyze-only", action="store_true")
    unified_cal.add_argument("--dry-run", action="store_true")
    unified_cal.add_argument(
        "--write-stub",
        action="store_true",
        help="Merge non-enforcing unified_tournament stub into --out JSON (no GPU sweep).",
    )

    gate = subparsers.add_parser(
        "gate",
        help="Composable preflight gates (YAML in conf/benchmark/gates/).",
    )
    gate.add_argument(
        "tokens",
        nargs="*",
        default=[],
        help="`list`, `run <id>`, or legacy `<id>` (beat_noop, beat_random, curriculum_staged).",
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
    gate.add_argument("--dry-run", action="store_true")
    gate.add_argument("--thresholds-path", type=Path, default=None)
    gate.add_argument("--profile-path", type=Path, default=None)
    gate.add_argument(
        "--train-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides appended after gate overrides.",
    )

    tournament_proof = subparsers.add_parser(
        "tournament-proof",
        help="Gate 5: held-out tournament win proof for a checkpoint.",
    )
    tournament_proof.add_argument(
        "--eval-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path for tournament eval.",
    )
    tournament_proof.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/preflight/tournament_proof_report.json"),
    )
    tournament_proof.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    tournament_proof.add_argument("--dry-run", action="store_true")
    tournament_proof.add_argument(
        "--thresholds-path",
        type=Path,
        default=None,
        help="Optional preflight calibration JSON for win-proof thresholds.",
    )
    tournament_proof.add_argument(
        "--baselines",
        default="random",
        help="Comma-separated baselines (first entry used).",
    )
    tournament_proof.add_argument("--campaign", default="preflight_held_out")
    tournament_proof.add_argument("--seeds", default="0,1,2,3,4")
    tournament_proof.add_argument("--games-per-pair", type=int, default=4)

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


def run_tournament_proof_cli(args: argparse.Namespace) -> int:
    from src.artifacts.tournament.unified.ladder import run_unified_ladder
    from src.artifacts.tournament.unified.spec import load_unified_tournament_spec
    from src.jax.preflight import PreflightVerdict, write_report
    from src.jax.preflight_calibration import (
        default_calibration_json_path,
    )

    checkpoint = Path(args.eval_checkpoint)
    if not checkpoint.is_file():
        print(f"missing checkpoint: {checkpoint}", file=sys.stderr)
        return 1

    thresholds_path = args.thresholds_path or default_calibration_json_path(REPO_ROOT)
    spec = load_unified_tournament_spec(thresholds_path)
    has_unified_section = False
    if thresholds_path.is_file():
        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        has_unified_section = isinstance(payload.get("unified_tournament"), dict)

    output_dir = (
        args.output_root
        / "campaigns"
        / args.campaign
        / "evaluations"
        / "preflight_win_proof_unified"
    )

    if args.dry_run:
        stage1_count = (
            len(spec.stage1.opponents)
            * len(spec.stage1.seeds)
            * spec.stage1.games_per_pair
            * (1 + ("4p_challenger_vs_baselines" in spec.stage1.formats))
        )
        if "4p_challenger_vs_baselines" in spec.stage1.formats:
            stage1_count = (
                len(spec.stage1.opponents)
                * len(spec.stage1.seeds)
                * spec.stage1.games_per_pair
                + len(spec.stage1.seeds) * spec.stage1.games_per_pair
            )
        plan = {
            "gate": "win_proof",
            "verdict": PreflightVerdict.INCONCLUSIVE.value,
            "dry_run": True,
            "unified": True,
            "enforcement": spec.enforcement,
            "needs_calibration": spec.needs_calibration,
            "stage1": {
                "opponents": list(spec.stage1.opponents),
                "seeds": list(spec.stage1.seeds),
                "games_per_pair": spec.stage1.games_per_pair,
                "scheduled_matches": stage1_count,
                "floors": dict(spec.stage1.floors),
            },
            "stage2": {
                "seeds": list(spec.stage2.seeds),
                "games_per_pair": spec.stage2.games_per_pair,
                "blocking_reason": spec.blocking_reason,
            },
            "output_dir": str(output_dir),
        }
        write_report(args.out, plan)
        print(json.dumps(plan, indent=2))
        return 0

    if spec.needs_calibration and not has_unified_section:
        report = {
            "gate": "win_proof",
            "commit_sha": _git_head_sha(),
            "verdict": PreflightVerdict.INCONCLUSIVE.value,
            "reasons": ["missing unified_tournament section in calibration JSON"],
            "checkpoint": str(checkpoint),
            "thresholds_path": str(thresholds_path),
            "evaluation_mode": "unified_tournament",
        }
        write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    verdict = run_unified_ladder(
        checkpoint,
        spec,
        output_dir,
        campaign=args.campaign,
        output_root=args.output_root,
    )

    preflight_verdict = PreflightVerdict.VERIFIED
    reasons: list[str] = []
    if not verdict.passed:
        if spec.enforcement:
            preflight_verdict = PreflightVerdict.NOT_VERIFIED
        else:
            preflight_verdict = PreflightVerdict.INCONCLUSIVE
        reasons.append(verdict.reason)

    report = {
        "gate": "win_proof",
        "commit_sha": _git_head_sha(),
        "verdict": preflight_verdict.value,
        "reasons": reasons,
        "unified_verdict": verdict.to_dict(),
        "unified_verdict_path": str(output_dir / "unified_verdict.json"),
        "checkpoint": str(checkpoint),
        "evaluation_mode": "unified_tournament",
        "enforcement": spec.enforcement,
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if preflight_verdict == PreflightVerdict.VERIFIED else 1


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
        if base_section and base_section.get("incumbent_checkpoint_path"):
            stub["incumbent_checkpoint_path"] = base_section["incumbent_checkpoint_path"]
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


def _learn_proof_primitive_payload() -> dict[str, object]:
    return {
        "workflow": "ow benchmark learn-proof",
        "prefer_primitives": True,
        "primitives": list(LEARN_PROOF_PRIMITIVES),
        "gate_list_command": "uv run ow benchmark gate --list",
    }


def _resolve_learn_proof_gates(args: argparse.Namespace) -> tuple[str, ...]:
    from src.jax.preflight import GATE_ORDER

    if args.steps:
        if args.gate is not None or args.through is not None:
            raise SystemExit("Use only one of --steps, --gate, or --through.")
        requested = tuple(
            item.strip()
            for item in str(args.steps).split(",")
            if item.strip()
        )
        if not requested:
            raise SystemExit("--steps requires at least one gate id.")
        unknown = [gate_id for gate_id in requested if gate_id not in GATE_ORDER]
        if unknown:
            raise SystemExit(
                f"Unknown learn-proof step(s): {', '.join(unknown)} "
                f"(expected subset of {', '.join(GATE_ORDER)})"
            )
        return tuple(gate_id for gate_id in GATE_ORDER if gate_id in requested)
    if args.gate is not None and args.through is not None:
        raise SystemExit("Use only one of --gate or --through.")
    if args.gate is not None:
        return (str(args.gate),)
    through = args.through or "beat_random"
    stop_index = GATE_ORDER.index(through)
    return GATE_ORDER[: stop_index + 1]


def run_learn_proof_cli(args: argparse.Namespace) -> int:
    """Thin composer over gate-run and tournament-proof primitives."""

    from src.jax.preflight import (
        GATE_ORDER,
        PreflightVerdict,
        gate_evaluation_to_dict,
        run_preflight_gate,
        write_report,
    )

    if args.print_primitives:
        print(json.dumps(_learn_proof_primitive_payload(), indent=2))
        return 0

    if args.eval_checkpoint is not None:
        return run_tournament_proof_cli(args)

    selected_gates = _resolve_learn_proof_gates(args)

    extra_train_overrides = tuple(args.train_overrides)
    started = __import__("time").perf_counter()
    evaluations = []
    overall_verdict = PreflightVerdict.VERIFIED
    for gate_id in selected_gates:
        gate_model = (
            "transformer_factorized"
            if gate_id == "curriculum_staged" and args.model != "planet_flow_target_heatmap"
            else args.model
        )
        evaluation = run_preflight_gate(
            gate_id,
            model=gate_model,
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            dry_run=args.dry_run,
            thresholds_path=args.thresholds_path,
            profiles_path=args.profile_path,
            extra_train_overrides=extra_train_overrides,
        )
        evaluations.append(evaluation)
        if evaluation.verdict != PreflightVerdict.VERIFIED:
            overall_verdict = evaluation.verdict
            break
    stages = [gate_evaluation_to_dict(item) for item in evaluations]

    report: dict[str, object] = {
        "gate": "learn_proof",
        "commit_sha": _git_head_sha(),
        "seconds_total": __import__("time").perf_counter() - started,
        "verdict": overall_verdict.value,
        "through": args.through or args.gate or ",".join(selected_gates),
        "steps": list(selected_gates),
        "model": args.model,
        "gate_order": list(GATE_ORDER),
        "stages": stages,
        **_learn_proof_primitive_payload(),
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if overall_verdict == PreflightVerdict.VERIFIED else 1


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


def _resolve_gate_id(args: argparse.Namespace) -> str | None:
    if args.list:
        return None
    tokens = [str(token) for token in args.tokens]
    if not tokens or tokens[0] == "list":
        return None
    if tokens[0] == "run":
        return tokens[1] if len(tokens) > 1 else None
    return tokens[0]


def run_factorized_sampler_cli(args: argparse.Namespace) -> int:
    script = REPO_ROOT / "scripts" / "benchmark_factorized_sampler.py"
    cmd = [
        sys.executable,
        str(script),
        "--max-moves-k",
        str(args.max_moves_k),
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]
    if args.decoder_carry:
        cmd.append("--decoder-carry")
    else:
        cmd.append("--no-decoder-carry")
    if args.assert_max_ms is not None:
        cmd.extend(["--assert-max-ms", str(args.assert_max_ms)])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return int(proc.returncode)


def run_gate_cli(args: argparse.Namespace) -> int:
    from src.cli.benchmark_gates import list_gate_recipes, run_gate_cli as run_gate

    gate_id = _resolve_gate_id(args)
    if gate_id is None:
        payload = {"gates": list_gate_recipes()}
        print(json.dumps(payload, indent=2))
        return 0
    if gate_id not in {"beat_noop", "beat_random", "curriculum_staged"}:
        print(
            f"Unknown gate id {gate_id!r}. Use: ow benchmark gate run beat_noop",
            file=sys.stderr,
        )
        return 2
    return run_gate(
        gate_id,
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
        case "calibrate-unified-tournament":
            return run_calibrate_unified_tournament_cli(args)
        case "shortlist-planet-flow-sweep":
            return run_shortlist_planet_flow_sweep_cli(args)
        case "planet-flow-noop-smoke":
            return run_planet_flow_noop_smoke_cli(args)
        case "factorized-sampler":
            return run_factorized_sampler_cli(args)
        case "gate":
            return run_gate_cli(args)
        case "tournament-proof":
            return run_tournament_proof_cli(args)
        case _:
            parser.error(f"unknown benchmark command: {args.command!r}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
