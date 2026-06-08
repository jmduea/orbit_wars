"""Argparse entry for the Colab training runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.orchestration import colab_runner as orch

REPO_ROOT = orch.REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    """Build the Colab runner argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch and inspect Colab training workers. Prefer `ow train colab` "
            "for the primary entrypoint; this module is also used standalone."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_common_args(preflight)

    prepare = subparsers.add_parser("prepare")
    _add_common_args(prepare)

    launch = subparsers.add_parser("launch")
    _add_common_args(launch)
    launch.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--session", required=True)
    status.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)
    status.add_argument("--sessions-path", type=Path, default=orch.DEFAULT_SESSIONS)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--session", required=True)
    sync.add_argument("--sync-dir", type=Path, default=orch.DEFAULT_SYNC_DIR)
    sync.add_argument("--timeout", type=int, default=orch.DEFAULT_LAUNCH_TIMEOUT)
    sync.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)
    sync.add_argument("--sessions-path", type=Path, default=orch.DEFAULT_SESSIONS)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--session", required=True)
    stop.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)

    shortlist = subparsers.add_parser("shortlist")
    shortlist.add_argument("--project", default="orbit_wars")
    shortlist.add_argument("--entity", default=None)
    shortlist.add_argument("--sweep-id", required=True)
    shortlist.add_argument("--limit", type=int, default=10)
    shortlist.add_argument("--out", type=Path, default=orch.DEFAULT_SHORTLIST, dest="output")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, default=orch.DEFAULT_WORK_DIR)
    parser.add_argument("--gpu", default=orch.DEFAULT_GPU)
    parser.add_argument("--timeout", type=int, default=orch.DEFAULT_LAUNCH_TIMEOUT)
    parser.add_argument("--ledger", type=Path, default=orch.DEFAULT_LEDGER)
    parser.add_argument("--sessions-path", type=Path, default=orch.DEFAULT_SESSIONS)
    parser.add_argument(
        "--from-shortlist",
        type=Path,
        default=None,
        help="Apply Hydra overrides from a ranked W&B shortlist JSON row.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Shortlist row index when using --from-shortlist.",
    )
    parser.add_argument(
        "--trust-base-jax",
        default="0",
        help="Packaged ORBIT_WARS_COLAB_TRUST_BASE_JAX value (default: 0).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        dest="hydra_overrides",
        metavar="KEY=VALUE",
        help="Hydra overrides packaged into worker-env.json.",
    )


def run(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch a Colab runner subcommand."""

    args = build_parser().parse_args(argv)
    request = orch.ColabRequest.from_namespace(args)
    if args.command == "preflight":
        return orch.run_preflight(request)
    if args.command == "prepare":
        orch.run_prepare(request)
        return 0
    if args.command == "launch":
        orch.run_launch(request)
        return 0
    if args.command == "status":
        orch.run_status(request)
        return 0
    if args.command == "sync":
        return orch.run_sync(request)
    if args.command == "stop":
        return orch.run_stop(request)
    if args.command == "shortlist":
        orch.run_shortlist(request)
        return 0
    raise SystemExit(f"Unknown command: {args.command!r}")


def main() -> None:
    """Console entrypoint for ``python -m src.cli.colab_runner``."""

    raise SystemExit(run(sys.argv[1:]))
