"""ROADMAP.md structure checks for CI and agents."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.roadmap import (
    MAX_NOW,
    RoadmapDocument,
    RoadmapRow,
    agent_payload,
    parse_roadmap,
    validate_roadmap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP_PATH = REPO_ROOT / "docs" / "ROADMAP.md"


@pytest.mark.config
def test_repo_roadmap_validates() -> None:
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    doc = parse_roadmap(text)
    errors = [message for message in validate_roadmap(doc) if not message.startswith("WARNING:")]
    assert errors == [], errors


@pytest.mark.config
def test_repo_roadmap_now_within_cap() -> None:
    doc = parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))
    assert len(doc.sections["now"]) <= MAX_NOW


@pytest.mark.config
def test_agent_payload_includes_manifest_and_rules() -> None:
    doc = parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))
    payload = agent_payload(doc)
    assert "roadmap" in payload
    assert "manifest_active" in payload
    assert payload["roadmap"]["counts"]["now"] >= 1
    assert any("ROADMAP Now" in rule for rule in payload["agent_rules"])


def test_parse_minimal_document() -> None:
    text = """# Roadmap

**Phase:** test-phase

## Now

| Item | Link |
|------|------|
| Alpha | #1 |

## Next

| Item | Link |
|------|------|

## Later

| Item | Link |
|------|------|

## Done (last 5)

| Item | Link |
|------|------|

_Last triaged: 2026-01-15_
"""
    doc = parse_roadmap(text)
    assert doc.phase == "test-phase"
    assert doc.last_triaged == date(2026, 1, 15)
    assert len(doc.sections["now"]) == 1
    assert doc.sections["now"][0] == RoadmapRow(item="Alpha", link="#1")


def test_validate_rejects_too_many_now() -> None:
    rows = [RoadmapRow(item=f"item{i}", link="#1") for i in range(MAX_NOW + 1)]
    doc = RoadmapDocument(
        phase="p",
        last_triaged=date(2026, 5, 1),
        sections={"now": rows, "next": [], "later": [], "done": []},
    )
    errors = [message for message in validate_roadmap(doc) if not message.startswith("WARNING:")]
    assert any("Now has" in error for error in errors)


def test_status_json_roundtrip_keys() -> None:
    from scripts.roadmap import status_payload

    doc = parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))
    payload = status_payload(doc)
    serialized = json.dumps(payload)
    loaded = json.loads(serialized)
    assert set(loaded["sections"]) == {"now", "next", "later", "done"}
