"""Shared helpers for ``ow benchmark``."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

LEARN_PROOF_PRIMITIVES: tuple[str, ...] = (
    "ow benchmark gate run admission",
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
        "  calibrate-unified-tournament Unified Stage-1 tournament floor calibration\n"
        "  calibrate-qualifier-seeds SSOT qualifier per-leg win-rate floor calibration\n"
        "  gate                     Composable preflight gates (run/list)\n"
        "  admission-throughput     Extract speed from gate JSONL (updates 3–20)\n"
        "  rollout-phase-profile    Offline admission-shaped phase profile (integration)\n"
        "  rollout-phase-breakdown  Itemize rollout cost from profile/gate JSONL\n"
        "  tournament-proof         Gate 5: Docker validate, then held-out ladder\n"
        "  shortlist-planet-flow-sweep  Rank finished Planet Flow W&B sweep runs\n"
        "  planet-flow-noop-smoke   Noop smoke on shortlist top-K before learn-proof\n"
        "  factorized-sampler     Tier-1 launch-hygiene microbenchmark (in-process JAX)\n"
        "  policy-path-profile    Encoder / decoder / shield_off vs cheap breakdown\n"
        "  env-parity-ab          Legacy vs train vs kaggle env step throughput A/B\n\n"
        "Examples:\n"
        "  make preflight-sanity\n"
        "  make preflight-learn-proof\n"
        "  uv run ow benchmark calibrate --analyze-only --analyze-campaigns\n"
        "  uv run ow benchmark gate --list\n"
        "  uv run ow benchmark gate run admission --out /tmp/admission.json\n"
        "  make gate-admission REPO_ROOT=/path/to/worktree\n"
        "  uv run ow benchmark gate run beat_noop --dry-run --verbose\n"
        "  uv run ow benchmark gate beat_random --dry-run\n"
        "  uv run ow benchmark admission-throughput outputs/.../anchor_learn_proof.json \\\n"
        "    --baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json\n\n"
        "E2E throughput (launch hygiene):\n"
        "  make test-launch-hygiene-e2e-throughput\n"
        "  uv run ow benchmark training --preset primary --label gate --out /tmp/gate.json \\\n"
        "    --baseline docs/benchmarks/launch-hygiene-e2e-baseline.json --assert-within-pct 10\n\n"
        "More: uv run ow benchmark learn-proof --help\n"
    )


def _init_benchmark_runtime() -> None:
    from src.jax.device import (
        configure_jax_runtime_for_host,
        ensure_jax_accelerator_backend,
    )

    configure_jax_runtime_for_host()
    ensure_jax_accelerator_backend()
