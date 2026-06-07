"""``ow runs`` — list and inspect campaign run directories."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from src.cli.run_status import queue_is_active, summarize_run_status


def _default_outputs_root() -> Path:
    return Path("outputs")


def _discover_run_manifests(outputs_root: Path) -> list[Path]:
    campaigns = outputs_root / "campaigns"
    if not campaigns.is_dir():
        return []
    manifests: list[Path] = []
    for manifest in campaigns.glob("*/runs/*/manifest.json"):
        if manifest.is_file():
            manifests.append(manifest)
    return sorted(manifests, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_log_path(run_dir: Path) -> Path | None:
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return None
    candidates = sorted(logs_dir.glob("*_jax.jsonl"))
    return candidates[0] if candidates else None


def cmd_list(args: argparse.Namespace) -> int:
    outputs_root = Path(args.outputs_root).resolve()
    manifests = _discover_run_manifests(outputs_root)
    if args.campaign:
        slug = str(args.campaign)
        manifests = [
            m for m in manifests if m.parent.parent.parent.name == slug
        ]
    rows: list[dict[str, object]] = []
    for manifest_path in manifests[: max(int(args.limit), 0) or len(manifests)]:
        manifest = _load_manifest(manifest_path)
        rows.append(
            {
                "campaign": manifest.get("campaign"),
                "run_id": manifest.get("run_id"),
                "run_dir": str(manifest_path.parent),
                "created_at": manifest.get("created_at"),
                "job_type": manifest.get("job_type"),
            }
        )
    if args.format == "json":
        print(json.dumps({"runs": rows}, indent=2))
    else:
        for row in rows:
            print(
                f"{row.get('campaign')}/{row.get('run_id')}  "
                f"dir={row.get('run_dir')}  created={row.get('created_at')}"
            )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"No manifest.json under run directory: {run_dir}")
    manifest = _load_manifest(manifest_path)
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(json.dumps(manifest, indent=2))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    log_path = _resolve_log_path(run_dir)
    if log_path is None:
        raise SystemExit(f"No *_jax.jsonl log under {run_dir / 'logs'}")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    tail = max(int(args.tail), 0)
    selected = lines[-tail:] if tail else lines
    for line in selected:
        print(line)
    return 0


def _campaign_slug_from_run_dir(run_dir: Path) -> str:
    return run_dir.parent.parent.name


def _archive_destination(run_dir: Path, outputs_root: Path) -> Path:
    campaign = _campaign_slug_from_run_dir(run_dir)
    return (
        outputs_root
        / "archived"
        / "campaigns"
        / campaign
        / "runs"
        / run_dir.name
    )


def _promoted_checkpoint_paths(run_dir: Path) -> set[Path]:
    promoted_manifest = (
        run_dir.parent.parent / "promoted" / "current_best" / "manifest.json"
    )
    if not promoted_manifest.is_file():
        return set()
    try:
        payload = json.loads(promoted_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    checkpoint_value = payload.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        return set()
    return {Path(checkpoint_value).resolve()}


def _resolve_checkpoint_targets(run_dir: Path, names: tuple[str, ...]) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    if not names:
        return sorted(checkpoints_dir.glob("jax_ckpt_*.pkl"))
    targets: list[Path] = []
    for name in names:
        candidate = checkpoints_dir / name
        if not candidate.name.endswith(".pkl"):
            candidate = checkpoints_dir / f"{name}.pkl"
        if candidate.is_file():
            targets.append(candidate.resolve())
    return targets


def cmd_archive(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"No manifest.json under run directory: {run_dir}")
    summary = summarize_run_status(run_dir)
    if queue_is_active(summary):
        raise SystemExit(
            "Refusing archive: run queue has active jobs. "
            "Cancel jobs first (`ow eval jobs cancel`)."
        )
    destination = _archive_destination(run_dir, outputs_root)
    payload: dict[str, object] = {
        "action": "archive",
        "run_dir": str(run_dir),
        "destination": str(destination),
        "dry_run": bool(args.dry_run),
    }
    if destination.exists():
        raise SystemExit(f"Archive destination already exists: {destination}")
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    if not args.confirm:
        raise SystemExit("Refusing archive without --confirm (use --dry-run to preview).")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))
    payload["archived"] = True
    print(json.dumps(payload, indent=2))
    return 0


def cmd_checkpoint_delete(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    names = tuple(
        part.strip()
        for part in (args.checkpoint or "").split(",")
        if part.strip()
    )
    targets = _resolve_checkpoint_targets(run_dir, names)
    if not targets:
        raise SystemExit(f"No checkpoint files matched under {run_dir / 'checkpoints'}")
    protected = _promoted_checkpoint_paths(run_dir)
    blocked = [path for path in targets if path in protected]
    if blocked:
        raise SystemExit(
            "Refusing delete: checkpoint is current promoted incumbent: "
            + ", ".join(str(p) for p in blocked)
        )
    payload: dict[str, object] = {
        "action": "checkpoint_delete",
        "run_dir": str(run_dir),
        "paths": [str(p) for p in targets],
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    if not args.confirm:
        raise SystemExit(
            "Refusing delete without --confirm (use --dry-run to preview)."
        )
    for path in targets:
        path.unlink()
    payload["deleted_count"] = len(targets)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    poll_seconds = max(float(args.poll_seconds), 0.1)
    idle_since: float | None = None
    while True:
        summary = summarize_run_status(run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if not queue_is_active(summary):
            if args.idle_exit_seconds is None:
                return 0
            now = time.monotonic()
            if idle_since is None:
                idle_since = now
            if now - idle_since >= float(args.idle_exit_seconds):
                return 0
        else:
            idle_since = None
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect training runs under outputs/campaigns (ow runs).",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=_default_outputs_root(),
        help="Outputs root (default: outputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List recent runs.")
    list_parser.add_argument(
        "--outputs-root",
        type=Path,
        default=parser.get_default("outputs_root"),
    )
    list_parser.add_argument("--campaign", default=None, help="Filter by campaign slug.")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
    )
    list_parser.set_defaults(handler=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show run manifest.json.")
    show_parser.add_argument(
        "--outputs-root",
        type=Path,
        default=parser.get_default("outputs_root"),
    )
    show_parser.add_argument("--run", type=Path, required=True)
    show_parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
    )
    show_parser.set_defaults(handler=cmd_show)

    logs_parser = subparsers.add_parser("logs", help="Tail training JSONL log.")
    logs_parser.add_argument(
        "--outputs-root",
        type=Path,
        default=parser.get_default("outputs_root"),
    )
    logs_parser.add_argument("--run", type=Path, required=True)
    logs_parser.add_argument("--tail", type=int, default=5)
    logs_parser.set_defaults(handler=cmd_logs)

    watch_parser = subparsers.add_parser(
        "watch",
        help="Poll run queue status and last log marker.",
    )
    watch_parser.add_argument(
        "--outputs-root",
        type=Path,
        default=parser.get_default("outputs_root"),
    )
    watch_parser.add_argument("--run", type=Path, required=True)
    watch_parser.add_argument("--poll-seconds", type=float, default=5.0)
    watch_parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=None,
        help="Exit after this many seconds with no queued/running jobs.",
    )
    watch_parser.set_defaults(handler=cmd_watch)

    archive_parser = subparsers.add_parser(
        "archive",
        help="Move a completed run tree under outputs/archived/ (operator cleanup).",
    )
    archive_parser.add_argument(
        "--outputs-root",
        type=Path,
        default=parser.get_default("outputs_root"),
    )
    archive_parser.add_argument("--run", type=Path, required=True)
    archive_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON plan without moving files.",
    )
    archive_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to perform the move (ignored with --dry-run).",
    )
    archive_parser.set_defaults(handler=cmd_archive)

    checkpoint_parser = subparsers.add_parser(
        "checkpoint",
        help="Delete checkpoint files under a run (not promotion rollback).",
    )
    checkpoint_sub = checkpoint_parser.add_subparsers(dest="checkpoint_command", required=True)
    delete_parser = checkpoint_sub.add_parser(
        "delete",
        help="Delete one or more jax_ckpt_*.pkl files.",
    )
    delete_parser.add_argument(
        "--outputs-root",
        type=Path,
        default=parser.get_default("outputs_root"),
    )
    delete_parser.add_argument("--run", type=Path, required=True)
    delete_parser.add_argument(
        "--checkpoint",
        default="",
        help="Comma-separated checkpoint basenames (default: all jax_ckpt_*.pkl).",
    )
    delete_parser.add_argument("--dry-run", action="store_true")
    delete_parser.add_argument("--confirm", action="store_true")
    delete_parser.set_defaults(handler=cmd_checkpoint_delete)

    return parser


def print_runs_help() -> None:
    print(
        "ow runs — inspect outputs/campaigns runs\n\n"
        "Subcommands:\n"
        "  list [--campaign SLUG] [--limit N]\n"
        "  show --run outputs/campaigns/<c>/runs/<id>\n"
        "  logs --run <path> [--tail N]\n"
        "  watch --run <path> [--poll-seconds 5]\n"
        "  archive --run <path> [--dry-run] [--confirm]\n"
        "  checkpoint delete --run <path> [--checkpoint NAME] [--dry-run] [--confirm]\n\n"
        "Examples:\n"
        "  uv run ow runs list --limit 10\n"
        "  uv run ow runs show --run outputs/campaigns/smoke/runs/run-001\n"
        "  uv run ow runs logs --run outputs/campaigns/smoke/runs/run-001 --tail 3\n"
    )


def main(argv: list[str] | None = None) -> int:
    if not argv:
        print_runs_help()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        print_runs_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
