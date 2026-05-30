#!/usr/bin/env python3
"""ROADMAP validate, intake, claims, gates, and wrap-up checks for agents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP_PATH = REPO_ROOT / "docs" / "ROADMAP.md"
MANIFEST_PATH = REPO_ROOT / ".omg" / "workflow-manifest.json"
IMPL_GATE_PATH = REPO_ROOT / ".omg" / "state" / "impl-gate.json"
ARCHIVED_BRAIN_DUMP = "docs/archive/brain_dump.md"

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
BANNED_INBOX_PATHS = ("brain_dump.md", "docs/brain_dump.md")

ACTIVE_MANIFEST_STATUSES = {"draft", "approved", "planned", "executing"}

TRIVIAL_REQUEST_PATTERNS = (
    re.compile(r"\b(typo|spelling|comment only|whitespace|formatting)\b", re.I),
    re.compile(r"\b(update (the )?readme|fix link)\b", re.I),
)

ISSUE_REF = re.compile(r"#(\d+)\b")
MANIFEST_ID = re.compile(r"\b([a-z][a-z0-9-]{2,})\b")

WORK_SESSION_PATH = REPO_ROOT / ".omg" / "state" / "work-session.json"

IMPL_PATH_PREFIXES = ("src/", "conf/", "tests/")
HOOK_EXEMPT_PREFIXES = (
    "scripts/roadmap.py",
    "scripts/roadmap_claims.py",
    "tests/test_roadmap.py",
    ".github/hooks/",
    ".cursor/hooks/",
    "docs/ROADMAP.md",
    "AGENTS.md",
    ".cursor/rules/",
)

_IMPL_PATH_RE = re.compile(
    r"(?:^|[\"'\s])(?P<path>(?:src|conf|tests)/[\w./_-]+)",
    re.MULTILINE,
)


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


def _link_contains_brain_dump(link: str) -> bool:
    lower = link.lower()
    return "brain_dump" in lower


def _extract_issue_ids(link: str) -> list[int]:
    return [int(match) for match in ISSUE_REF.findall(link)]


def issue_ids_by_section(doc: RoadmapDocument) -> dict[int, list[str]]:
    """Map GitHub issue number to ROADMAP sections that reference it."""
    by_issue: dict[int, list[str]] = {}
    for section in SECTION_ORDER:
        for row in doc.sections.get(section, []):
            for issue_id in _extract_issue_ids(row.link):
                by_issue.setdefault(issue_id, []).append(section)
    return by_issue


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

            if section in {"now", "next"}:
                if _link_contains_brain_dump(row.link):
                    errors.append(
                        f"{section}: {row.item!r} links to retired brain_dump; "
                        "use GitHub issue #NNN or .omg spec/plan path"
                    )
                if row.link.strip() in {"—", "-", "#TBD", ""}:
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

    if len(done_rows) == 0:
        warnings.append("Done section is empty")

    for issue_id, sections in issue_ids_by_section(doc).items():
        unique_sections = list(dict.fromkeys(sections))
        if len(unique_sections) > 1:
            errors.append(
                f"Issue #{issue_id} listed in {', '.join(unique_sections)}; "
                "keep each issue in exactly one ROADMAP section"
            )

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


def load_impl_gate() -> dict | None:
    if not IMPL_GATE_PATH.exists():
        return None
    try:
        data = json.loads(IMPL_GATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def save_impl_gate(payload: dict) -> None:
    IMPL_GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMPL_GATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_impl_gate() -> None:
    if IMPL_GATE_PATH.exists():
        IMPL_GATE_PATH.unlink()


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _score_row_match(request_tokens: set[str], row: RoadmapRow) -> float:
    item_tokens = _normalize_tokens(row.item)
    link_tokens = _normalize_tokens(row.link)
    overlap = len(request_tokens & (item_tokens | link_tokens))
    if overlap == 0:
        return 0.0
    return overlap / max(len(request_tokens), 1)


def _is_trivial_request(request: str) -> bool:
    return any(pattern.search(request) for pattern in TRIVIAL_REQUEST_PATTERNS)


def _impl_gate_strict_enabled() -> bool:
    value = os.environ.get("ORBIT_WARS_IMPL_GATE", "1").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def normalize_repo_path(path: str) -> str:
    """Return a repo-relative POSIX path when possible."""
    cleaned = path.replace("\\", "/").strip()
    if not cleaned:
        return ""
    candidate = Path(cleaned)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve().relative_to(REPO_ROOT.resolve())).replace(
                "\\", "/"
            )
        except ValueError:
            return cleaned.lstrip("/")
    return cleaned.lstrip("./")


def is_implementation_path(path: str) -> bool:
    norm = normalize_repo_path(path)
    if not norm:
        return False
    if any(norm.startswith(prefix) for prefix in HOOK_EXEMPT_PREFIXES):
        return False
    return any(norm.startswith(prefix) for prefix in IMPL_PATH_PREFIXES)


def extract_paths_from_tool_input(tool_name: str, tool_input: object) -> list[str]:
    """Collect file paths from Cursor/Copilot pre-tool JSON payloads."""
    paths: list[str] = []
    if tool_input is None:
        return paths

    parsed: object = tool_input
    if isinstance(tool_input, str):
        stripped = tool_input.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = tool_input
        else:
            parsed = tool_input

    if isinstance(parsed, dict):
        for key in (
            "path",
            "filePath",
            "file_path",
            "target_file",
            "targetNotebook",
            "target_notebook",
        ):
            value = parsed.get(key)
            if value:
                paths.append(str(value))
        extra = parsed.get("paths")
        if isinstance(extra, list):
            paths.extend(str(item) for item in extra if item)
        for key in ("old_string", "new_string", "contents"):
            blob = parsed.get(key)
            if isinstance(blob, str):
                for match in _IMPL_PATH_RE.finditer(blob):
                    paths.append(match.group("path"))
    elif isinstance(parsed, str):
        for match in _IMPL_PATH_RE.finditer(parsed):
            paths.append(match.group("path"))

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        norm = normalize_repo_path(raw)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def hook_guard(*, paths: list[str]) -> dict:
    """Fail closed on src/conf/tests edits without an approved impl-gate (Cursor hook)."""
    if os.environ.get("ORBIT_WARS_HOOK_DISABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return {"allow": True, "skipped": "ORBIT_WARS_HOOK_DISABLE", "touched": []}

    touched = [path for path in paths if is_implementation_path(path)]
    if not touched:
        return {"allow": True, "touched": []}

    gate = implementation_gate(request=None)
    if gate["allowed"]:
        return {
            "allow": True,
            "touched": touched,
            "impl_gate": gate.get("impl_gate"),
        }

    blockers = gate.get("blockers") or []
    summary = "; ".join(blockers[:2])
    extra = f" (+{len(touched) - 3} more)" if len(touched) > 3 else ""
    shown = ", ".join(touched[:3]) + extra
    reason = (
        f"ROADMAP funnel blocked edits to {shown}. {summary} "
        'First: uv run python scripts/roadmap.py begin "<user request>" '
        "then claim + approve-impl before src/conf/tests changes."
    )
    return {
        "allow": False,
        "reason": reason,
        "blockers": blockers,
        "touched": touched,
        "next_steps": gate.get("next_steps") or [],
    }


def save_work_session(payload: dict) -> None:
    WORK_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORK_SESSION_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def begin_work(request: str) -> dict:
    """Run agent + intake + gate for free-form user text; persist session for the hook."""
    request = request.strip()
    if not request:
        raise ValueError("begin requires non-empty request text")

    doc = parse_roadmap(load_roadmap_text())
    intake = intake_request(request, doc)
    gate = implementation_gate(request=request)
    may_implement = bool(
        gate["allowed"]
        and intake.get("capture_to") is None
        and not intake.get("requires_planning")
    )
    issue_ids = intake.get("issue_ids") or []
    primary_issue = issue_ids[0] if issue_ids else None

    payload = {
        "request": request,
        "intake": intake,
        "gate": {
            "allowed": gate["allowed"],
            "strict_mode": gate["strict_mode"],
            "blockers": gate["blockers"],
        },
        "may_implement": may_implement,
        "primary_issue": primary_issue,
        "next_steps": intake.get("next_steps") or gate.get("next_steps") or [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    save_work_session(payload)
    return payload


def intake_request(request: str, doc: RoadmapDocument) -> dict:
    request = request.strip()
    request_tokens = _normalize_tokens(request)
    issue_ids_in_request = [int(n) for n in ISSUE_REF.findall(request)]

    best: tuple[float, str, RoadmapRow] | None = None
    min_match_score = 0.34
    sections = ("now", "next", "later")
    if issue_ids_in_request:
        sections = ("now", "next", "later", "done")
    for section in sections:
        for row in doc.sections.get(section, []):
            score = _score_row_match(request_tokens, row)
            row_issues = _extract_issue_ids(row.link)
            if issue_ids_in_request and any(i in row_issues for i in issue_ids_in_request):
                score = max(score, 1.0)
            if score >= min_match_score and (best is None or score > best[0]):
                best = (score, section, row)

    active = load_manifest_active()
    manifest_match: dict | None = None
    for entry in active:
        entry_id = str(entry.get("id", ""))
        title = str(entry.get("title", ""))
        haystack = f"{entry_id} {title}".lower()
        if entry_id and entry_id in request.lower():
            manifest_match = entry
            break
        overlap = _normalize_tokens(haystack) & request_tokens
        if len(overlap) >= 2 or (entry_id and entry_id in request.lower()):
            manifest_match = entry
            break

    matched_section = best[1] if best else None
    matched_row = asdict(best[2]) if best else None
    matched_issues = _extract_issue_ids(best[2].link) if best else []

    requires_planning = True
    suggested_workflow = "ralplan"
    if _is_trivial_request(request):
        requires_planning = False
        suggested_workflow = "quick"
    elif matched_section == "now" and matched_issues:
        requires_planning = False
        suggested_workflow = "execute"
    elif matched_section == "done" and matched_issues:
        requires_planning = False
        suggested_workflow = "execute"
    elif matched_section in {"now", "next"}:
        suggested_workflow = "ralplan"
    elif matched_section == "later":
        suggested_workflow = "deep-interview"
    elif manifest_match:
        suggested_workflow = "ralplan"
    else:
        suggested_workflow = "deep-interview"

    capture_to = None
    if matched_section is None and manifest_match is None:
        capture_to = "later"

    return {
        "request": request,
        "matched": matched_section is not None or manifest_match is not None,
        "roadmap_section": matched_section,
        "roadmap_row": matched_row,
        "issue_ids": matched_issues,
        "manifest_match": (
            {
                "id": manifest_match.get("id"),
                "status": manifest_match.get("status"),
                "title": manifest_match.get("title"),
            }
            if manifest_match
            else None
        ),
        "capture_to": capture_to,
        "requires_planning": requires_planning,
        "suggested_workflow": suggested_workflow,
        "next_steps": _intake_next_steps(
            matched_section=matched_section,
            requires_planning=requires_planning,
            suggested_workflow=suggested_workflow,
            capture_to=capture_to,
        ),
    }


def _intake_next_steps(
    *,
    matched_section: str | None,
    requires_planning: bool,
    suggested_workflow: str,
    capture_to: str | None,
) -> list[str]:
    if capture_to == "later":
        return [
            "Add one line to docs/ROADMAP.md Later (no implementation)",
            "Run: uv run python scripts/roadmap.py validate",
            f"When ready: /{suggested_workflow} then promote to Next/Now with GitHub issue",
        ]
    if requires_planning:
        return [
            f"Run /{suggested_workflow} (or /omg-autopilot through spec approval)",
            "Produce execution plan: chunks, manifest register, GitHub issues with AC",
            "Run: uv run python scripts/roadmap.py claim --issue N --path <dirs>",
            "Run: uv run python scripts/roadmap.py approve-impl --issue N",
        ]
    return [
        "Run: uv run python scripts/roadmap.py claim --issue N --path <dirs>",
        "Run: uv run python scripts/roadmap.py approve-impl --issue N",
        "Implement on branch issue/N-*; run tests",
        "gh issue close N --comment '<evidence>'",
        "Run: uv run python scripts/roadmap.py wrap-up --issue N --evidence 'tests+commit'",
        "Move ROADMAP row to Done; manifest complete",
    ]


def implementation_gate(*, request: str | None = None) -> dict:
    doc = parse_roadmap(load_roadmap_text())
    intake = intake_request(request, doc) if request else None
    impl = load_impl_gate()
    strict = _impl_gate_strict_enabled()

    allowed = False
    blockers: list[str] = []

    from scripts import roadmap_claims

    if impl and impl.get("approved"):
        allowed = True
        if impl.get("issue"):
            issue_num = int(str(impl["issue"]).lstrip("#"))
            if roadmap_claims.load_claim(issue_num) is None:
                blockers.append(
                    f"No active claim for {impl['issue']}; run: roadmap.py claim --issue {issue_num} --path …"
                )
        if request and intake:
            gate_issue = impl.get("issue")
            gate_manifest = impl.get("manifest_id")
            intake_issues = [f"#{i}" for i in intake.get("issue_ids", [])]
            if gate_issue and intake_issues and gate_issue not in intake_issues:
                blockers.append(
                    f"impl-gate issue {gate_issue} does not match intake {intake_issues}"
                )
            if gate_manifest and intake.get("manifest_match"):
                mid = intake["manifest_match"].get("id")
                if mid and mid != gate_manifest:
                    blockers.append(
                        f"impl-gate manifest {gate_manifest!r} != intake {mid!r}"
                    )
    elif _is_trivial_request(request or ""):
        allowed = True
    else:
        blockers.append(
            "No approved implementation gate (.omg/state/impl-gate.json). "
            "Complete planning phases then: roadmap.py approve-impl --issue N"
        )

    if intake and not intake["matched"] and not (impl and impl.get("approved")):
        blockers.append(
            "Request does not match ROADMAP Now/Next or active manifest; "
            "capture to Later or run intake after updating ROADMAP"
        )

    if blockers:
        allowed = False

    if not strict and blockers and _is_trivial_request(request or ""):
        allowed = True

    return {
        "allowed": allowed,
        "strict_mode": strict,
        "impl_gate": impl,
        "intake": intake,
        "blockers": blockers,
        "next_steps": [] if allowed else (intake or {}).get("next_steps", []),
    }


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
    from scripts import roadmap_claims

    active = load_manifest_active()
    impl = load_impl_gate()
    return {
        "canonical_human_priority": "docs/ROADMAP.md Now section",
        "canonical_agent_packages": ".omg/workflow-manifest.json (active statuses only)",
        "impl_gate_approved": bool(impl and impl.get("approved")),
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
        "active_claims": roadmap_claims.load_all_claims(active_only=True),
        "workflow_phases": [
            "0: roadmap.py agent + omg_workflow_manifest.py active + roadmap.py claims",
            '1: roadmap.py begin "<user message>" (intake + gate + work-session.json)',
            "2: /deep-interview or /ralplan (or /omg-autopilot through spec approval)",
            "3: execution plan → issues + manifest + ROADMAP promote",
            "4: roadmap.py claim --issue N --path …",
            "5: roadmap.py approve-impl + implement (gate)",
            "6: gh issue close + roadmap.py wrap-up --issue N --evidence",
            "7: ROADMAP Done, manifest complete",
        ],
        "agent_rules": [
            "One claim per issue; no overlapping paths across active claims.",
            "Do not use docs/brain_dump.md; retired inbox — ROADMAP + issues only.",
            "Prefer ROADMAP Now over manifest when choosing what to work on next.",
            'Free-form chat: first command on implementation intent: roadmap.py begin "<user message>".',
            "No src/conf/tests edits until approve-impl; Cursor pre-tool hook enforces impl-gate.",
            "Session end: wrap-up + check-wrap-up + check-session --require-clean.",
            "Create GitHub issues after execution planning (phase 3), not before for new work.",
            "After changing ROADMAP: uv run python scripts/roadmap.py validate",
            "Before push after closing an issue: add ROADMAP Done row, remove from Now/Next, then make roadmap-check.",
            "wrap-up fails if the issue is not under ROADMAP Done (when GitHub issue is CLOSED).",
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


def cmd_intake(args: argparse.Namespace) -> int:
    if not args.request.strip():
        print("intake requires non-empty request text", file=sys.stderr)
        return 1
    doc = parse_roadmap(load_roadmap_text())
    print(json.dumps(intake_request(args.request, doc), indent=2))
    return 0


def cmd_begin(args: argparse.Namespace) -> int:
    if not args.request.strip():
        print("begin requires non-empty request text", file=sys.stderr)
        return 1
    try:
        payload = begin_work(args.request)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    if args.require_may_implement and not payload.get("may_implement"):
        return 1
    return 0


def cmd_hook_check(args: argparse.Namespace) -> int:
    paths: list[str] = list(args.paths or [])
    tool_name = ""
    if not paths:
        raw = sys.stdin.read()
        if raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                print("hook-check: invalid JSON on stdin", file=sys.stderr)
                return 1
            tool_name = str(
                payload.get("toolName")
                or payload.get("tool_name")
                or payload.get("toolName")
                or ""
            )
            tool_input = payload.get("toolInput") or payload.get("tool_input")
            paths = extract_paths_from_tool_input(tool_name, tool_input)

    result = hook_guard(paths=paths)
    decision = "approve" if result.get("allow") else "deny"
    out: dict = {"decision": decision, "hook": result}
    if not result.get("allow"):
        out["reason"] = result.get("reason", "ROADMAP funnel blocked")
    print(json.dumps(out))
    return 0 if result.get("allow") else 1


def cmd_gate(args: argparse.Namespace) -> int:
    payload = implementation_gate(request=args.request)
    print(json.dumps(payload, indent=2))
    if args.require_allowed and not payload["allowed"]:
        return 1
    return 0


def cmd_approve_impl(args: argparse.Namespace) -> int:
    if not args.issue and not args.manifest_id:
        print("approve-impl requires --issue and/or --manifest-id", file=sys.stderr)
        return 1
    payload = {
        "approved": True,
        "issue": f"#{args.issue}" if args.issue else None,
        "manifest_id": args.manifest_id,
        "summary": args.summary or "",
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_impl_gate(payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_clear_impl(args: argparse.Namespace) -> int:
    clear_impl_gate()
    print("impl-gate cleared")
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    from scripts import roadmap_claims

    if not args.issue:
        print("claim requires --issue", file=sys.stderr)
        return 1
    owner = args.owner or roadmap_claims.default_agent_owner()
    try:
        payload = roadmap_claims.claim_issue(
            issue=args.issue,
            owner=owner,
            paths=args.paths or [],
            branch=args.branch,
            manifest_id=args.manifest_id,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    from scripts import roadmap_claims

    print(json.dumps({"claims": roadmap_claims.load_all_claims(active_only=True)}, indent=2))
    return 0


def cmd_wrap_up(args: argparse.Namespace) -> int:
    from scripts import roadmap_claims

    if not args.issue:
        print("wrap-up requires --issue", file=sys.stderr)
        return 1
    owner = args.owner or roadmap_claims.default_agent_owner()
    result = roadmap_claims.finalize_wrap_up(
        issue=args.issue,
        evidence=args.evidence or "",
        evidence_file=args.evidence_file,
        owner=owner,
        skip_github=args.skip_github_check,
    )
    print(json.dumps(result, indent=2))
    if args.require_passed and not result.get("passed"):
        return 1
    return 0


def cmd_check_wrap_up(args: argparse.Namespace) -> int:
    from scripts import roadmap_claims

    if not args.issue:
        print("check-wrap-up requires --issue", file=sys.stderr)
        return 1
    owner = args.owner or roadmap_claims.default_agent_owner()
    result = roadmap_claims.wrap_up_check(
        issue=args.issue,
        evidence=args.evidence or "",
        evidence_file=args.evidence_file,
        owner=owner,
        skip_github=args.skip_github_check,
    )
    print(json.dumps(result, indent=2))
    if args.require_passed and not result.get("passed"):
        return 1
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    from scripts import roadmap_claims

    if not args.issue:
        print("release requires --issue", file=sys.stderr)
        return 1
    owner = args.owner or roadmap_claims.default_agent_owner()
    try:
        result = roadmap_claims.release_issue(
            issue=args.issue, owner=owner, force=args.force
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get("released") else 1


def cmd_check_session(args: argparse.Namespace) -> int:
    from scripts import roadmap_claims

    owner = args.owner or roadmap_claims.default_agent_owner()
    claims = roadmap_claims.load_all_claims(active_only=True)
    open_for_owner = []
    for claim in claims:
        if claim.get("owner") != owner:
            continue
        issue = int(claim["issue"])
        completion = roadmap_claims.load_completion(issue)
        if not (completion and completion.get("wrapped_up")):
            open_for_owner.append(claim)
    payload = {
        "owner": owner,
        "open_claims": open_for_owner,
        "passed": len(open_for_owner) == 0,
        "blockers": [
            f"Issue #{c['issue']} claimed without wrap-up (run: roadmap.py wrap-up --issue {c['issue']} --evidence '…')"
            for c in open_for_owner
        ],
    }
    print(json.dumps(payload, indent=2))
    allow_open = os.environ.get("ORBIT_WARS_ALLOW_OPEN_CLAIMS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if args.require_clean and not payload["passed"] and not allow_open:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "validate",
            "status",
            "agent",
            "intake",
            "begin",
            "gate",
            "hook-check",
            "approve-impl",
            "clear-impl",
            "claim",
            "claims",
            "wrap-up",
            "check-wrap-up",
            "release",
            "check-session",
        ),
        default="validate",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default="",
        help="request text for intake/gate/begin",
    )
    parser.add_argument("--strict-links", action="store_true")
    parser.add_argument("--issue", type=int, help="GitHub issue number for approve-impl")
    parser.add_argument("--manifest-id", help="manifest entry id for approve-impl")
    parser.add_argument("--summary", help="short description stored in impl-gate")
    parser.add_argument(
        "--require-allowed",
        action="store_true",
        help="exit 1 when gate denies implementation",
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="path prefix for claim (repeatable)",
    )
    parser.add_argument("--branch", help="git branch for this issue claim")
    parser.add_argument("--owner", help="agent owner id (default ORBIT_WARS_AGENT_ID)")
    parser.add_argument("--evidence", help="wrap-up evidence text")
    parser.add_argument("--evidence-file", help="optional file merged into evidence")
    parser.add_argument(
        "--skip-github-check",
        action="store_true",
        help="skip gh issue closed verification (tests only)",
    )
    parser.add_argument(
        "--require-passed",
        action="store_true",
        help="exit 1 when wrap-up/check-wrap-up fails",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="exit 1 when check-session finds open claims",
    )
    parser.add_argument(
        "--require-may-implement",
        action="store_true",
        help="exit 1 when begin reports may_implement=false",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force release without wrap-up (abandon claim)",
    )

    args = parser.parse_args()

    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "agent":
        return cmd_agent(args)
    if args.command == "intake":
        return cmd_intake(args)
    if args.command == "begin":
        return cmd_begin(args)
    if args.command == "hook-check":
        return cmd_hook_check(args)
    if args.command == "gate":
        return cmd_gate(args)
    if args.command == "approve-impl":
        return cmd_approve_impl(args)

    if args.command == "claim":
        return cmd_claim(args)
    if args.command == "claims":
        return cmd_claims(args)
    if args.command == "wrap-up":
        return cmd_wrap_up(args)
    if args.command == "check-wrap-up":
        return cmd_check_wrap_up(args)
    if args.command == "release":
        return cmd_release(args)
    if args.command == "check-session":
        return cmd_check_session(args)
    return cmd_clear_impl(args)


if __name__ == "__main__":
    raise SystemExit(main())
