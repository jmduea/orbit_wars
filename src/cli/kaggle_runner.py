"""Argparse entry for the Kaggle training runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.orchestration import kaggle_runner as orch

REPO_ROOT = orch.REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    """Build the Kaggle runner argument parser."""

    parser = argparse.ArgumentParser(
        description="Launch and inspect standalone or W&B Kaggle training workers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_package_args(prepare)
    prepare.add_argument("--sweep-id", default=None)
    prepare.add_argument("--accelerator", action="append", dest="accelerators")

    preflight = subparsers.add_parser("preflight")
    _add_package_args(preflight)
    preflight.add_argument("--project", default="orbit_wars")
    preflight.add_argument("--entity", default=None)
    preflight.add_argument("--timeout-seconds", type=int, default=30)

    launch = subparsers.add_parser("launch")
    _add_package_args(launch)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--sweep-id", default=None)
    launch.add_argument("--create-sweep", action="store_true")
    launch.add_argument("--project", default="orbit_wars")
    launch.add_argument("--entity", default=None)
    launch.add_argument("--timeout-seconds", type=int, default=43200)
    launch.add_argument("--accelerator", action="append", dest="accelerators")
    launch.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)
    launch.add_argument(
        "--no-accelerator-flag-fallback",
        action="store_true",
        help=(
            "Do not retry a push without --accelerator when the local Kaggle CLI "
            "appears not to support the accelerator flag."
        ),
    )

    status = subparsers.add_parser("status")
    status.add_argument("kernel_ref")
    status.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)

    sync = subparsers.add_parser("sync-output")
    sync.add_argument("kernel_ref")
    sync.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/kaggle_runner/synced",
    )
    sync.add_argument("--force", action="store_true")
    sync.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)

    shortlist = subparsers.add_parser("shortlist")
    shortlist.add_argument("--project", default="orbit_wars")
    shortlist.add_argument("--entity", default=None)
    shortlist.add_argument("--sweep-id", required=True)
    shortlist.add_argument("--limit", type=int, default=10)
    shortlist.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/kaggle_runner/shortlist.json",
    )

    latest = subparsers.add_parser("latest-checkpoint")
    latest.add_argument("--project", default="orbit_wars")
    latest.add_argument("--entity", default=None)
    latest.add_argument("--sweep-id", required=True)
    latest.add_argument("--run-id", default=None)
    latest.add_argument("--limit", type=int, default=1000)

    return parser


def _add_package_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, default=orch.DEFAULT_WORK_DIR)
    parser.add_argument("--kernel-id", default=orch.default_kernel_id())
    parser.add_argument("--title", default="orbit wars kaggle runner")
    parser.add_argument("--sweep-yaml", type=Path, default=orch.DEFAULT_SWEEP)
    parser.add_argument(
        "--run-type",
        choices=("full", "smoke", "benchmark"),
        default=None,
        help=(
            "Packaged worker run mode. smoke uses short training/calibration "
            "defaults for first live validation. benchmark runs a calibrated "
            "throughput grid (3 variants, warmup=2, updates=30, timeout=3600s)."
        ),
    )
    parser.add_argument(
        "--calibration-max-variants",
        type=int,
        default=None,
        metavar="N",
        help="Packaged ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS (default: run-type preset).",
    )
    parser.add_argument(
        "--calibration-warmup",
        type=int,
        default=None,
        metavar="N",
        help="Packaged ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP.",
    )
    parser.add_argument(
        "--calibration-updates",
        type=int,
        default=None,
        metavar="N",
        help="Packaged ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES.",
    )
    parser.add_argument(
        "--calibration-timeout-seconds",
        type=int,
        default=None,
        metavar="SEC",
        help="Packaged ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help=(
            "Standalone worker mode: skip W&B sweep agent and Kaggle Secrets. "
            "Training config comes from the packaged sweep YAML fixed parameters."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        dest="standalone_overrides",
        metavar="KEY=VALUE",
        help=(
            "Extra Hydra overrides for standalone (--no-wandb) workers. "
            "Packaged into worker-env.json."
        ),
    )


def run(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch a Kaggle runner subcommand."""

    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        orch.run_prepare(args)
        return 0
    if args.command == "preflight":
        return orch.run_preflight(args)
    if args.command == "launch":
        orch.run_launch(args)
        return 0
    if args.command == "status":
        orch.run_status(args)
        return 0
    if args.command == "sync-output":
        return orch.run_sync_output(args)
    if args.command == "shortlist":
        orch.run_shortlist(args)
        return 0
    if args.command == "latest-checkpoint":
        orch.run_latest_checkpoint(args)
        return 0
    raise SystemExit(f"Unknown command: {args.command!r}")


def main() -> None:
    """Console entrypoint for ``python -m src.cli.kaggle_runner``."""

    raise SystemExit(run(sys.argv[1:]))
