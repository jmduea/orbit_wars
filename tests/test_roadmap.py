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
    implementation_gate,
    intake_request,
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
    assert any("brain_dump" in rule.lower() for rule in payload["agent_rules"])
    assert "workflow_phases" in payload


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


@pytest.mark.config
def test_repo_roadmap_has_no_brain_dump_in_now_next() -> None:
    text = ROADMAP_PATH.read_text(encoding="utf-8").lower()
    now_chunk = text.split("## now", 1)[1].split("## next", 1)[0]
    next_chunk = text.split("## next", 1)[1].split("## later", 1)[0]
    assert "brain_dump" not in now_chunk
    assert "brain_dump" not in next_chunk


def test_validate_rejects_brain_dump_link_in_now() -> None:
    doc = RoadmapDocument(
        phase="p",
        last_triaged=date(2026, 5, 1),
        sections={
            "now": [RoadmapRow(item="Bad", link="[brain_dump](brain_dump.md)")],
            "next": [],
            "later": [],
            "done": [],
        },
    )
    errors = [m for m in validate_roadmap(doc) if not m.startswith("WARNING:")]
    assert any("brain_dump" in e for e in errors)


def test_intake_matches_kaggle_validation_now() -> None:
    doc = parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))
    result = intake_request("fix kaggle docker validation submission", doc)
    assert result["matched"] is True
    assert result["roadmap_section"] == "now"
    assert 96 in result["issue_ids"]


def test_intake_unknown_request_captures_to_later() -> None:
    doc = parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))
    result = intake_request("implement quantum blockchain for orbit wars", doc)
    assert result["matched"] is False
    assert result["capture_to"] == "later"
    assert result["requires_planning"] is True


def test_gate_blocks_without_approve_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_IMPL_GATE", "1")
    payload = implementation_gate(request="fix kaggle docker validation")
    assert payload["allowed"] is False
    assert payload["blockers"]
