"""``ow runs`` — list and inspect campaign run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    return parser


def print_runs_help() -> None:
    print(
        "ow runs — inspect outputs/campaigns runs\n\n"
        "Subcommands:\n"
        "  list [--campaign SLUG] [--limit N]\n"
        "  show --run outputs/campaigns/<c>/runs/<id>\n"
        "  logs --run <path> [--tail N]\n\n"
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
