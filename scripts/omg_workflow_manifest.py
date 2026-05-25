#!/usr/bin/env python3
"""Validate and inspect .omg/workflow-manifest.json for coding agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / ".omg" / "workflow-manifest.json"
ACTIVE_STATUSES = {"draft", "approved", "planned", "executing"}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict) -> int:
    errors: list[str] = []
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("Manifest entries must be a list")

    seen_ids: set[str] = set()
    by_id: dict[str, dict] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            errors.append("Invalid entry object")
            continue
        entry_id = raw.get("id")
        path_value = raw.get("path")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append("Entry missing id")
            continue
        if entry_id in seen_ids:
            errors.append(f"Duplicate id: {entry_id}")
        seen_ids.add(entry_id)
        by_id[entry_id] = raw

        if not isinstance(path_value, str) or not path_value:
            errors.append(f"{entry_id}: missing path")
            continue
        if not (REPO_ROOT / path_value).exists():
            errors.append(f"{entry_id}: missing file {path_value}")

        for link_field in ("spec_id", "plan_id"):
            link = raw.get(link_field)
            if isinstance(link, str) and link and link not in by_id and link not in seen_ids:
                # linked entry may appear later; defer broken-link check
                pass

    for raw in entries:
        if not isinstance(raw, dict):
            continue
        entry_id = str(raw.get("id", ""))
        for link_field in ("spec_id", "plan_id"):
            link = raw.get(link_field)
            if isinstance(link, str) and link and link not in by_id:
                errors.append(f"{entry_id}: broken {link_field} -> {link}")
        related = raw.get("related_ids", [])
        if isinstance(related, list):
            for related_id in related:
                if isinstance(related_id, str) and related_id not in by_id:
                    errors.append(f"{entry_id}: broken related_ids -> {related_id}")

    registered = {
        str(entry.get("path"))
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    for folder in ("specs", "plans"):
        dir_path = REPO_ROOT / ".omg" / folder
        if not dir_path.exists():
            continue
        for md_path in sorted(dir_path.glob("*.md")):
            rel = md_path.relative_to(REPO_ROOT).as_posix()
            if rel not in registered:
                errors.append(f"Unregistered markdown: {rel}")

    if errors:
        print("workflow-manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"workflow-manifest ok ({len(entries)} entries)")
    return 0


def list_active(manifest: dict) -> int:
    entries = manifest.get("entries", [])
    active = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("status") in ACTIVE_STATUSES
    ]
    print(json.dumps({"total": len(active), "entries": active}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("validate", "active"),
        default="validate",
        help="validate manifest integrity or print active backlog",
    )
    args = parser.parse_args()
    manifest = load_manifest()
    if args.command == "active":
        return list_active(manifest)
    return validate_manifest(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
