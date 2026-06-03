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
    "ow eval package --checkpoint <ckpt> --output-dir <dir> --validate-docker",
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
        "  tournament-proof         Gate 5: Docker validate, then held-out ladder\n"
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

    tournament_proof = subparsers.add_parser(
        "tournament-proof",
        help=(
            "Gate 5: Kaggle Docker packaging validation, then held-out unified "
            "tournament ladder (submit-valid order)."
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


def run_tournament_proof_cli(args: argparse.Namespace) -> int:
    from src.artifacts.submit_valid_funnel import (
        docker_gate_passed,
        run_submit_valid_docker_gate,
    )
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
    docker_output_dir = output_dir / "docker_validation"

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
            "submit_valid_order": ["docker_validation", "unified_tournament_ladder"],
            "docker_output_dir": str(docker_output_dir),
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

    docker_manifest: dict[str, object] = {}
    try:
        docker_manifest = run_submit_valid_docker_gate(
            checkpoint_path=checkpoint,
            output_dir=docker_output_dir,
            repo_root=REPO_ROOT,
        )
    except (OSError, RuntimeError) as exc:
        report = {
            "gate": "win_proof",
            "commit_sha": _git_head_sha(),
            "verdict": PreflightVerdict.NOT_VERIFIED.value,
            "reasons": [f"docker_validation_failed: {exc}"],
            "checkpoint": str(checkpoint),
            "evaluation_mode": "submit_valid_docker_gate",
            "docker_output_dir": str(docker_output_dir),
            "tournament_skipped": True,
            "tournament_skipped_reason": "docker_validation_failed",
        }
        write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    if not docker_gate_passed(docker_manifest):
        report = {
            "gate": "win_proof",
            "commit_sha": _git_head_sha(),
            "verdict": PreflightVerdict.NOT_VERIFIED.value,
            "reasons": ["docker_validation_failed"],
            "checkpoint": str(checkpoint),
            "evaluation_mode": "submit_valid_docker_gate",
            "docker_manifest": docker_manifest,
            "docker_output_dir": str(docker_output_dir),
            "tournament_skipped": True,
            "tournament_skipped_reason": "docker_validation_failed",
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
        "docker_validation_ok": True,
        "docker_output_dir": str(docker_output_dir),
        "docker_manifest": docker_manifest,
        "unified_verdict": verdict.to_dict(),
        "unified_verdict_path": str(output_dir / "unified_verdict.json"),
        "checkpoint": str(checkpoint),
        "evaluation_mode": "unified_tournament",
        "enforcement": spec.enforcement,
        "submit_valid_order": ["docker_validation", "unified_tournament_ladder"],
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
