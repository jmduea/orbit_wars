#!/usr/bin/env python3
"""Validate and inspect docs/ROADMAP.md for humans and coding agents."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP_PATH = REPO_ROOT / "docs" / "ROADMAP.md"
MANIFEST_PATH = REPO_ROOT / ".omg" / "workflow-manifest.json"

SECTION_ORDER = ("now", "next", "later", "done")
SECTION_HEADERS = {
    "now": "## Now",
    "next": "## Next",
    "later": "## Later",
    "done": "## Done (last 5)",
}
MAX_NOW = 3
MAX_DONE = 5
STALE_TRIAGED_DAYS = 21

ACTIVE_MANIFEST_STATUSES = {"draft", "approved", "planned", "executing"}


@dataclass(slots=True)
class RoadmapRow:
    item: str
    link: str


@dataclass(slots=True)
class RoadmapDocument:
    phase: str | None
    last_triaged: date | None
    sections: dict[str, list[RoadmapRow]]


def load_roadmap_text() -> str:
    if not ROADMAP_PATH.exists():
        raise SystemExit(f"Missing roadmap: {ROADMAP_PATH}")
    return ROADMAP_PATH.read_text(encoding="utf-8")


def _parse_table_rows(lines: list[str]) -> list[RoadmapRow]:
    rows: list[RoadmapRow] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        if re.match(r"^\|\s*-+", stripped):
            in_table = True
            continue
        if not in_table:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        item, link = cells[0], cells[1]
        if item.lower() in {"item", ""}:
            continue
        rows.append(RoadmapRow(item=item, link=link))
    return rows


def parse_roadmap(text: str) -> RoadmapDocument:
    phase_match = re.search(r"\*\*Phase:\*\*\s*(.+)", text)
    phase = phase_match.group(1).strip() if phase_match else None

    triaged_match = re.search(r"_Last triaged:\s*(\d{4}-\d{2}-\d{2})_", text)
    last_triaged: date | None = None
    if triaged_match:
        last_triaged = date.fromisoformat(triaged_match.group(1))

    sections: dict[str, list[RoadmapRow]] = {key: [] for key in SECTION_ORDER}
    for key, header in SECTION_HEADERS.items():
        start = text.find(header)
        if start < 0:
            continue
        chunk = text[start + len(header) :]
        next_header = re.search(r"\n## ", chunk)
        if next_header:
            chunk = chunk[: next_header.start()]
        sections[key] = _parse_table_rows(chunk.splitlines())

    return RoadmapDocument(phase=phase, last_triaged=last_triaged, sections=sections)


def validate_roadmap(
    doc: RoadmapDocument,
    *,
    strict_links: bool = False,
) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    for key in SECTION_ORDER:
        if key not in doc.sections:
            errors.append(f"Missing section: {SECTION_HEADERS[key]}")

    now_rows = doc.sections.get("now", [])
    done_rows = doc.sections.get("done", [])
    if len(now_rows) > MAX_NOW:
        errors.append(f"Now has {len(now_rows)} items (max {MAX_NOW})")
    if len(done_rows) > MAX_DONE:
        errors.append(f"Done has {len(done_rows)} items (max {MAX_DONE})")

    if doc.last_triaged is None:
        errors.append("Missing _Last triaged: YYYY-MM-DD_ footer")
    elif doc.last_triaged > date.today():
        errors.append(f"Last triaged is in the future: {doc.last_triaged}")
    elif (date.today() - doc.last_triaged).days > STALE_TRIAGED_DAYS:
        warnings.append(
            f"Last triaged is {(date.today() - doc.last_triaged).days} days ago "
            f"(>{STALE_TRIAGED_DAYS}); confirm Now still reflects real work"
        )

    if not doc.phase:
        warnings.append("Missing **Phase:** line")

    seen_items: dict[str, str] = {}
    for section, rows in doc.sections.items():
        for row in rows:
            if row.item in seen_items and section in {"now", "next"}:
                errors.append(
                    f"Duplicate item in {section}: {row.item!r} "
                    f"(also in {seen_items[row.item]})"
                )
            seen_items.setdefault(row.item, section)
            if section in {"now", "next"} and row.link.strip() in {"—", "-", "#TBD", ""}:
                warnings.append(
                    f"{section}: {row.item!r} has no issue/spec link "
                    "(prefer GitHub issue #NNN when work starts)"
                )
            if strict_links and row.link.startswith("["):
                path_match = re.match(r"\[.*?\]\((.*?)\)", row.link)
                if path_match:
                    rel = path_match.group(1).split("#")[0]
                    if rel and not rel.startswith("#") and not (REPO_ROOT / rel).exists():
                        errors.append(f"Broken link path: {rel} ({row.item!r})")

    placeholder_done = [
        row
        for row in done_rows
        if "none yet" in row.item.lower() or row.item.startswith("_(")
    ]
    if len(done_rows) > 0 and len(placeholder_done) == len(done_rows):
        pass
    elif len(done_rows) == 0:
        warnings.append("Done section is empty")

    return errors + [f"WARNING: {message}" for message in warnings]


def load_manifest_active() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("status") in ACTIVE_MANIFEST_STATUSES
    ]


def status_payload(doc: RoadmapDocument) -> dict:
    return {
        "path": str(ROADMAP_PATH.relative_to(REPO_ROOT)),
        "phase": doc.phase,
        "last_triaged": doc.last_triaged.isoformat() if doc.last_triaged else None,
        "counts": {key: len(doc.sections.get(key, [])) for key in SECTION_ORDER},
        "limits": {"now_max": MAX_NOW, "done_max": MAX_DONE},
        "sections": {
            key: [asdict(row) for row in doc.sections.get(key, [])] for key in SECTION_ORDER
        },
    }


def agent_payload(doc: RoadmapDocument) -> dict:
    active = load_manifest_active()
    return {
        "canonical_human_priority": "docs/ROADMAP.md Now section",
        "canonical_agent_packages": ".omg/workflow-manifest.json (active statuses only)",
        "roadmap": status_payload(doc),
        "manifest_active": {
            "total": len(active),
            "entries": [
                {
                    "id": entry.get("id"),
                    "status": entry.get("status"),
                    "title": entry.get("title"),
                    "path": entry.get("path"),
                }
                for entry in active
            ],
        },
        "agent_rules": [
            "Prefer ROADMAP Now over manifest when choosing what to work on next.",
            "Use manifest + linked specs/plans only for multi-step agent execution.",
            "After changing ROADMAP, run: uv run python scripts/roadmap.py validate",
            "On finish: close GitHub issue, move row to Done (keep ≤5), do not duplicate evidence in ROADMAP.",
        ],
    }


def cmd_validate(args: argparse.Namespace) -> int:
    doc = parse_roadmap(load_roadmap_text())
    messages = validate_roadmap(doc, strict_links=args.strict_links)
    errors = [message for message in messages if not message.startswith("WARNING:")]
    warnings = [message.removeprefix("WARNING: ") for message in messages if message.startswith("WARNING:")]

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if errors:
        print("roadmap validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"roadmap ok (now={len(doc.sections.get('now', []))}, "
        f"next={len(doc.sections.get('next', []))}, "
        f"later={len(doc.sections.get('later', []))}, "
        f"done={len(doc.sections.get('done', []))})"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    doc = parse_roadmap(load_roadmap_text())
    print(json.dumps(status_payload(doc), indent=2))
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    doc = parse_roadmap(load_roadmap_text())
    print(json.dumps(agent_payload(doc), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("validate", "status", "agent"),
        default="validate",
        help="validate structure, print JSON status, or agent-oriented combined view",
    )
    parser.add_argument(
        "--strict-links",
        action="store_true",
        help="fail when markdown link targets are missing repo files",
    )
    args = parser.parse_args()
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "status":
        return cmd_status(args)
    return cmd_agent(args)


if __name__ == "__main__":
    raise SystemExit(main())
