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
    begin_work,
    extract_paths_from_tool_input,
    hook_guard,
    implementation_gate,
    intake_request,
    is_implementation_path,
    parse_roadmap,
    save_impl_gate,
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
    assert payload["strict_mode"] is True


def test_is_implementation_path_exempts_roadmap_script() -> None:
    assert is_implementation_path("scripts/roadmap.py") is False
    assert is_implementation_path("src/jax/train.py") is True


def test_extract_paths_from_write_payload() -> None:
    paths = extract_paths_from_tool_input(
        "editFiles",
        {"path": "src/orchestration/kaggle_runner.py"},
    )
    assert paths == ["src/orchestration/kaggle_runner.py"]


def test_hook_guard_blocks_src_without_impl_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORBIT_WARS_HOOK_DISABLE", raising=False)
    result = hook_guard(paths=["src/jax/train.py"])
    assert result["allow"] is False
    assert (
        "impl-gate" in result["reason"].lower()
        or "approve-impl" in result["reason"].lower()
    )


def test_hook_guard_allows_with_approved_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "hook-agent")
    monkeypatch.setenv("ORBIT_WARS_HOOK_DISABLE", "")
    gate_path = tmp_path / "impl-gate.json"
    monkeypatch.setattr("scripts.roadmap.IMPL_GATE_PATH", gate_path)
    save_impl_gate({"approved": True, "issue": "#96", "summary": "test"})
    roadmap_claims.claim_issue(issue=96, owner="hook-agent", paths=["src/jax/"])
    result = hook_guard(paths=["src/jax/train.py"])
    assert result["allow"] is True


def test_begin_work_matches_issue_96(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.roadmap.WORK_SESSION_PATH", tmp_path / "work-session.json"
    )
    payload = begin_work("work on issue #96 docker validation")
    assert payload["intake"]["matched"] is True
    assert payload["primary_issue"] == 96
    assert "next_steps" in payload


def test_hook_guard_ignores_non_impl_paths() -> None:
    result = hook_guard(paths=["docs/ROADMAP.md", "README.md"])
    assert result["allow"] is True
    assert result["touched"] == []


def test_claim_path_overlap_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-a")
    roadmap_claims.claim_issue(issue=1, owner="agent-a", paths=["src/orchestration/"])
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-b")
    with pytest.raises(ValueError, match="overlap"):
        roadmap_claims.claim_issue(issue=2, owner="agent-b", paths=["src/orchestration/kaggle_runner.py"])


def test_wrap_up_requires_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import roadmap_claims

    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(tmp_path / "state"))
    result = roadmap_claims.wrap_up_check(issue=99, evidence="short", skip_github=True)
    assert result["passed"] is False
    assert any("Evidence too short" in b for b in result["blockers"])


def test_finalize_wrap_up_records_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-wrap")
    roadmap_claims.claim_issue(issue=42, owner="agent-wrap", paths=["docs/"])
    evidence = "make test-fast passed; commit deadbeef; updated ROADMAP Done row"
    result = roadmap_claims.finalize_wrap_up(
        issue=42, evidence=evidence, owner="agent-wrap", skip_github=True
    )
    assert result["passed"] is True
    assert roadmap_claims.load_completion(42) is not None
    assert roadmap_claims.load_claim(42) is None


def test_check_session_flags_open_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import os
    import subprocess
    import sys

    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-open")
    roadmap_claims.claim_issue(issue=7, owner="agent-open", paths=["src/"])
    env = os.environ.copy()
    env["ORBIT_WARS_STATE_DIR"] = str(state)
    env["ORBIT_WARS_AGENT_ID"] = "agent-open"
    proc = subprocess.run(
        [sys.executable, "scripts/roadmap.py", "check-session", "--require-clean"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["passed"] is False
