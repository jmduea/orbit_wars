"""Argparse tree for ``ow benchmark``."""

from __future__ import annotations

import argparse
from pathlib import Path


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
        help=(
            "Tier-1 factorized shield sampler microbenchmark "
            "(in-process JAX via src/jax/factorized_sampler_benchmark.py)."
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
