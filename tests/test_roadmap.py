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
    issue_ids_by_section,
    parse_roadmap,
    save_impl_gate,
    validate_roadmap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP_PATH = REPO_ROOT / "docs" / "ROADMAP.md"

# Synthetic ROADMAP for wrap-up tests — avoids coupling to live issue numbers.
_WRAP_UP_ROADMAP_TEXT = """# Roadmap

**Phase:** test

## Now

| Item | Link |
|------|------|

## Next

| Item | Link |
|------|------|
| Open next item | [#9001](https://github.com/jmduea/orbit_wars/issues/9001) |

## Later

| Item | Link |
|------|------|

## Done (last 5)

| Item | Link |
|------|------|
| Closed done item | [#9002](https://github.com/jmduea/orbit_wars/issues/9002) |

_Last triaged: 2026-05-30_
"""
_WRAP_UP_NEXT_ISSUE = 9001
_WRAP_UP_DONE_ISSUE = 9002


def _repo_roadmap_doc() -> RoadmapDocument:
    return parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))


def _issue_unique_to_section(doc: RoadmapDocument, section: str) -> int:
    """Return one issue that appears only in ``section`` (not duplicated elsewhere)."""
    candidates = [
        issue_id
        for issue_id, sections in issue_ids_by_section(doc).items()
        if sections == [section]
    ]
    if not candidates:
        pytest.skip(f"No issue unique to ROADMAP {section!r}")
    return candidates[0]


@pytest.fixture
def wrap_up_roadmap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.roadmap.load_roadmap_text", lambda: _WRAP_UP_ROADMAP_TEXT)


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
    counts = payload["roadmap"]["counts"]
    assert counts["now"] >= 0
    assert counts["next"] >= 0
    assert any("brain_dump" in rule.lower() for rule in payload["agent_rules"])
    assert any("Done row" in rule for rule in payload["agent_rules"])
    assert "workflow_phases" in payload


def test_agent_payload_allows_empty_now_and_next() -> None:
    text = """# Roadmap

**Phase:** planning

## Now

| Item | Link |
|------|------|

## Next

| Item | Link |
|------|------|

## Later

| Item | Link |
|------|------|
| Backlog item | — |

## Done (last 5)

| Item | Link |
|------|------|

_Last triaged: 2026-05-30_
"""
    doc = parse_roadmap(text)
    errors = [message for message in validate_roadmap(doc) if not message.startswith("WARNING:")]
    assert errors == [], errors
    payload = agent_payload(doc)
    assert payload["roadmap"]["counts"]["now"] == 0
    assert payload["roadmap"]["counts"]["next"] == 0


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


def test_validate_rejects_issue_in_multiple_sections() -> None:
    doc = RoadmapDocument(
        phase="p",
        last_triaged=date(2026, 5, 1),
        sections={
            "now": [RoadmapRow(item="Open", link="[#96](https://github.com/jmduea/orbit_wars/issues/96)")],
            "next": [],
            "later": [],
            "done": [RoadmapRow(item="Closed", link="[#96](https://github.com/jmduea/orbit_wars/issues/96)")],
        },
    )
    assert issue_ids_by_section(doc)[96] == ["now", "done"]
    errors = [m for m in validate_roadmap(doc) if not m.startswith("WARNING:")]
    assert any("Issue #96" in e and "now, done" in e for e in errors)


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


def test_intake_matches_done_row_by_issue_ref() -> None:
    doc = _repo_roadmap_doc()
    done_issue = _issue_unique_to_section(doc, "done")
    result = intake_request(f"work on issue #{done_issue}", doc)
    assert result["matched"] is True
    assert result["roadmap_section"] == "done"
    assert done_issue in result["issue_ids"]


def test_intake_unknown_request_captures_to_later() -> None:
    doc = _repo_roadmap_doc()
    result = intake_request("implement quantum blockchain for orbit wars", doc)
    assert result["matched"] is False
    assert result["capture_to"] == "later"
    assert result["requires_planning"] is True


def test_gate_blocks_without_approve_impl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ORBIT_WARS_IMPL_GATE", "1")
    monkeypatch.delenv("ORBIT_WARS_ISSUE_ID", raising=False)
    monkeypatch.setattr("scripts.roadmap.IMPL_GATE_PATH", tmp_path / "no-gate.json")
    monkeypatch.setattr("scripts.roadmap.IMPL_GATES_DIR", tmp_path / "empty-gates")
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORBIT_WARS_HOOK_DISABLE", raising=False)
    monkeypatch.delenv("ORBIT_WARS_ISSUE_ID", raising=False)
    monkeypatch.setattr("scripts.roadmap.IMPL_GATE_PATH", tmp_path / "no-gate.json")
    monkeypatch.setattr("scripts.roadmap.IMPL_GATES_DIR", tmp_path / "empty-gates")
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
    save_impl_gate({"approved": True, "issue": "#8801", "summary": "test"})
    monkeypatch.setenv("ORBIT_WARS_ISSUE_ID", "8801")
    roadmap_claims.claim_issue(
        issue=8801,
        owner="hook-agent",
        paths=["src/jax/"],
        branch="issue/8801",
        setup_worktree=False,
    )
    monkeypatch.setattr("scripts.roadmap_git.current_branch", lambda _root=None: "issue/8801")
    result = hook_guard(paths=["src/jax/train.py"])
    assert result["allow"] is True


def test_begin_work_matches_done_issue_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.roadmap.WORK_SESSION_PATH", tmp_path / "work-session.json"
    )
    done_issue = _issue_unique_to_section(_repo_roadmap_doc(), "done")
    payload = begin_work(f"work on issue #{done_issue}")
    assert payload["intake"]["matched"] is True
    assert payload["intake"]["roadmap_section"] == "done"
    assert payload["primary_issue"] == done_issue
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


def test_claim_rejects_comma_separated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-comma")
    with pytest.raises(ValueError, match="comma"):
        roadmap_claims.claim_issue(
            issue=3,
            owner="agent-comma",
            paths=["src/,tests/"],
        )


def test_approve_impl_blocks_overwrite_same_issue_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from scripts.roadmap import IMPL_GATES_DIR, cmd_approve_impl

    monkeypatch.setattr("scripts.roadmap.IMPL_GATES_DIR", tmp_path / "impl-gates")
    save_impl_gate({"approved": True, "issue": "#1", "summary": "first"}, issue=1)
    args = argparse.Namespace(issue=1, manifest_id=None, summary="second", force=False)
    assert cmd_approve_impl(args) == 1


def test_approve_impl_allows_parallel_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from scripts.roadmap import cmd_approve_impl, load_impl_gate_for_issue

    monkeypatch.setattr("scripts.roadmap.IMPL_GATES_DIR", tmp_path / "impl-gates")
    save_impl_gate({"approved": True, "issue": "#1", "summary": "first"}, issue=1)
    args = argparse.Namespace(issue=2, manifest_id=None, summary="second", force=False)
    assert cmd_approve_impl(args) == 0
    assert load_impl_gate_for_issue(1) is not None
    assert load_impl_gate_for_issue(2) is not None


def test_begin_rejects_multiple_issue_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.roadmap.WORK_SESSION_PATH", tmp_path / "work-session.json"
    )
    with pytest.raises(ValueError, match="multiple issues"):
        begin_work("fix #1 and also #2 in one session")


def test_begin_allows_multiple_issue_refs_with_batch_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.roadmap.WORK_SESSION_PATH", tmp_path / "work-session.json"
    )
    payload = begin_work("fix #1 and also #2 in one session", batch_ok=True)
    assert payload["request"] == "fix #1 and also #2 in one session"
    assert payload["may_implement"] is False


def test_begin_issue_override_sets_primary_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.roadmap.WORK_SESSION_PATH", tmp_path / "work-session.json"
    )
    payload = begin_work("implement quantum blockchain", issue=42)
    assert payload["primary_issue"] == 42
    assert payload["issue"] == 42
    assert 42 in payload["intake"]["issue_ids"]


def test_wrap_up_requires_evidence(
    wrap_up_roadmap: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(tmp_path / "state"))
    result = roadmap_claims.wrap_up_check(
        issue=_WRAP_UP_NEXT_ISSUE,
        evidence="short",
        skip_github=True,
    )
    assert result["passed"] is False
    assert any("Evidence too short" in b for b in result["blockers"])


def test_wrap_up_blocks_when_issue_not_in_done(wrap_up_roadmap: None) -> None:
    from scripts import roadmap_claims

    result = roadmap_claims.wrap_up_check(
        issue=_WRAP_UP_NEXT_ISSUE,
        evidence=(
            "make test-fast passed; commit abc; updated ROADMAP Done row "
            f"for #{_WRAP_UP_NEXT_ISSUE}"
        ),
        skip_github=True,
    )
    assert result["passed"] is False
    assert result["roadmap_section"] == "next"
    assert any("ROADMAP Next" in b for b in result["blockers"])


def test_wrap_up_passes_when_issue_in_done(
    wrap_up_roadmap: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(tmp_path / "state"))
    result = roadmap_claims.wrap_up_check(
        issue=_WRAP_UP_DONE_ISSUE,
        evidence="make test-domain-artifacts passed; Kaggle episode 78216645; commit f6231fc",
        skip_github=True,
    )
    assert result["passed"] is True
    assert result.get("roadmap_section") == "done"


def test_finalize_wrap_up_records_completion(
    wrap_up_roadmap: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-wrap")
    monkeypatch.setattr("scripts.roadmap.IMPL_GATE_PATH", state / "impl-gate.json")
    roadmap_claims.claim_issue(
        issue=_WRAP_UP_DONE_ISSUE, owner="agent-wrap", paths=["docs/"]
    )
    evidence = "make test-fast passed; commit deadbeef; updated ROADMAP Done row"
    result = roadmap_claims.finalize_wrap_up(
        issue=_WRAP_UP_DONE_ISSUE,
        evidence=evidence,
        owner="agent-wrap",
        skip_github=True,
    )
    assert result["passed"] is True
    assert roadmap_claims.load_completion(_WRAP_UP_DONE_ISSUE) is not None
    assert roadmap_claims.load_claim(_WRAP_UP_DONE_ISSUE) is None


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


def test_find_stale_claims_detects_roadmap_done(
    wrap_up_roadmap: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(tmp_path / "state"))
    roadmap_claims.claim_issue(
        issue=_WRAP_UP_DONE_ISSUE,
        owner="stale-agent",
        paths=["docs/"],
        branch=f"issue/{_WRAP_UP_DONE_ISSUE}",
    )
    stale = roadmap_claims.find_stale_claims(skip_github=True)
    issues = [int(entry["issue"]) for entry in stale]
    assert _WRAP_UP_DONE_ISSUE in issues
    assert "roadmap_done" in stale[0]["stale_reasons"]


def test_release_stale_claims_dry_run(
    wrap_up_roadmap: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(tmp_path / "state"))
    roadmap_claims.claim_issue(
        issue=_WRAP_UP_DONE_ISSUE,
        owner="stale-agent",
        paths=["docs/"],
        branch=f"issue/{_WRAP_UP_DONE_ISSUE}",
    )
    result = roadmap_claims.release_stale_claims(dry_run=True, skip_github=True)
    assert result["released_count"] == 0
    assert roadmap_claims.load_claim(_WRAP_UP_DONE_ISSUE) is not None
    applied = roadmap_claims.release_stale_claims(dry_run=False, skip_github=True)
    assert applied["released_count"] == 1
    assert roadmap_claims.load_claim(_WRAP_UP_DONE_ISSUE) is None


def test_hook_guard_uses_per_issue_gate_with_issue_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "hook-agent")
    monkeypatch.setenv("ORBIT_WARS_ISSUE_ID", "8802")
    monkeypatch.setenv("ORBIT_WARS_HOOK_DISABLE", "")
    monkeypatch.setattr("scripts.roadmap.IMPL_GATES_DIR", state / "impl-gates")
    save_impl_gate({"approved": True, "issue": "#8802", "summary": "test"}, issue=8802)
    roadmap_claims.claim_issue(
        issue=8802,
        owner="hook-agent",
        paths=["src/jax/"],
        branch="issue/8802",
        setup_worktree=False,
    )
    monkeypatch.setattr("scripts.roadmap_git.current_branch", lambda _root=None: "issue/8802")
    result = hook_guard(paths=["src/jax/train.py"])
    assert result["allow"] is True


def test_check_session_global_flags_any_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import subprocess
    import sys

    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "coordinator")
    roadmap_claims.claim_issue(issue=8, owner="other-agent", paths=["src/"])
    env = os.environ.copy()
    env["ORBIT_WARS_STATE_DIR"] = str(state)
    env["ORBIT_WARS_AGENT_ID"] = "coordinator"
    proc = subprocess.run(
        [sys.executable, "scripts/roadmap.py", "check-session", "--global"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "global"
    assert len(payload["open_claims"]) == 1
    assert proc.returncode == 0
