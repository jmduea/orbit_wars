"""``ow promote`` — inspect and roll back campaign promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.artifacts.promotion_ops import (
    campaign_dir,
    demote_campaign,
    load_current_promoted_manifest,
    read_promotion_index,
)


def _default_outputs_root() -> Path:
    return Path("outputs")


def cmd_show(args: argparse.Namespace) -> int:
    camp_dir = campaign_dir(Path(args.output_root).resolve(), str(args.campaign))
    payload = load_current_promoted_manifest(camp_dir)
    if payload is None:
        raise SystemExit(
            f"No promoted manifest for campaign {args.campaign!r} under {camp_dir}"
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    records = read_promotion_index(root, campaign=str(args.campaign))
    limit = max(int(args.limit), 0) if args.limit is not None else len(records)
    selected = records[-limit:] if limit else records
    body = {
        "campaign": str(args.campaign),
        "output_root": str(root),
        "count": len(selected),
        "records": selected,
    }
    if args.format == "json":
        print(json.dumps(body, indent=2))
    else:
        for record in selected:
            event = record.get("event", "promoted")
            metric = record.get("metric_name")
            value = record.get("metric_value")
            checkpoint = record.get("checkpoint_path")
            updated = record.get("updated_at")
            print(
                f"{event}  metric={metric}={value}  "
                f"checkpoint={checkpoint}  updated={updated}"
            )
    return 0


def cmd_demote(args: argparse.Namespace) -> int:
    result = demote_campaign(
        Path(args.output_root).resolve(),
        str(args.campaign),
        to_previous=bool(args.to_previous),
        dry_run=bool(args.dry_run),
        reason=str(args.reason),
    )
    print(
        json.dumps(
            {
                "action": result.action,
                "reason": result.reason,
                "campaign": result.campaign,
                "campaign_dir": str(result.campaign_dir),
                "dry_run": result.dry_run,
                "previous_manifest_path": (
                    str(result.previous_manifest_path)
                    if result.previous_manifest_path
                    else None
                ),
                "restored_manifest_path": (
                    str(result.restored_manifest_path)
                    if result.restored_manifest_path
                    else None
                ),
            },
            indent=2,
        )
    )
    if result.action == "noop" and not result.dry_run:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Campaign promotion manifests under outputs/campaigns (ow promote).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_outputs_root(),
        help="Outputs root (default: outputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    root_default = parser.get_default("output_root")

    show = subparsers.add_parser(
        "show", help="Print promoted/current_best/manifest.json."
    )
    show.add_argument("--output-root", type=Path, default=root_default)
    show.add_argument("--campaign", required=True, help="Campaign slug.")
    show.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format (default: json).",
    )
    show.set_defaults(handler=cmd_show)

    history = subparsers.add_parser(
        "history",
        help="List promotion index rows from indexes/promoted.jsonl.",
    )
    history.add_argument("--output-root", type=Path, default=root_default)
    history.add_argument("--campaign", required=True, help="Campaign slug.")
    history.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max records to print (default: 20, most recent).",
    )
    history.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format (default: json).",
    )
    history.set_defaults(handler=cmd_history)

    demote = subparsers.add_parser(
        "demote",
        help="Clear current promotion or restore the previous indexed promotion.",
    )
    demote.add_argument("--output-root", type=Path, default=root_default)
    demote.add_argument("--campaign", required=True, help="Campaign slug.")
    demote.add_argument(
        "--to-previous",
        action="store_true",
        help="Restore the prior promotion from indexes/promoted.jsonl.",
    )
    demote.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the demote action without modifying manifests.",
    )
    demote.add_argument(
        "--reason",
        default="operator_demote",
        help="Audit reason stored on the demote index row.",
    )
    demote.set_defaults(handler=cmd_demote)

    return parser


def print_promote_help() -> None:
    print(
        "ow promote — inspect and roll back campaign promotion\n\n"
        "Subcommands:\n"
        "  show --campaign <name>\n"
        "  history --campaign <name> [--limit N]\n"
        "  demote --campaign <name> [--to-previous] [--dry-run]\n"
    )


def main(argv: list[str] | None = None) -> int:
    if not argv:
        print_promote_help()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
