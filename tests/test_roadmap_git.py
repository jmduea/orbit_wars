"""Git branch and worktree guard tests for parallel agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.roadmap_git import (
    branch_guard,
    effective_claim_branch,
    issue_branch_name,
    is_protected_base_branch,
    slugify,
)


def test_slugify_normalizes_titles() -> None:
    slug = slugify("Delete broken duplicate test_jax_policy_gnn.py (P0)")
    assert slug.startswith("delete-broken-duplicate-test-jax-policy-gnn")


def test_issue_branch_name_with_and_without_slug() -> None:
    assert issue_branch_name(102) == "issue/102"
    assert issue_branch_name(102, "delete-gnn-test") == "issue/102-delete-gnn-test"


def test_effective_claim_branch_defaults_when_missing() -> None:
    assert effective_claim_branch({"issue": 99, "branch": None}) == "issue/99"
    assert effective_claim_branch({"issue": 99, "branch": "issue/99-cleanup"}) == "issue/99-cleanup"


def test_is_protected_base_branch() -> None:
    assert is_protected_base_branch("main") is True
    assert is_protected_base_branch("issue/102") is False


def test_branch_guard_grandfathers_null_branch_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-null-branch")
    monkeypatch.delenv("ORBIT_WARS_ISSUE_ID", raising=False)
    monkeypatch.setenv("ORBIT_WARS_BRANCH_ENFORCE", "1")
    roadmap_claims.claim_issue(
        issue=55,
        owner="agent-null-branch",
        paths=["tests/"],
        branch=None,
        setup_worktree=False,
    )
    # Force null branch as legacy claims store
    claim = roadmap_claims.load_claim(55)
    assert claim is not None
    claim["branch"] = None
    roadmap_claims.save_claim(claim)

    monkeypatch.setattr("scripts.roadmap_git.current_branch", lambda _root=None: "main")
    result = branch_guard(owner="agent-null-branch", repo_root=tmp_path)
    assert result["allow"] is True
    assert "grandfathered" in result.get("branch_warning", "").lower()


def test_branch_guard_blocks_main_when_branch_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-branch")
    monkeypatch.setenv("ORBIT_WARS_ISSUE_ID", "77")
    monkeypatch.setenv("ORBIT_WARS_BRANCH_ENFORCE", "1")
    roadmap_claims.claim_issue(
        issue=77,
        owner="agent-branch",
        paths=["src/jax/"],
        branch="issue/77",
        setup_worktree=False,
    )
    monkeypatch.setattr("scripts.roadmap_git.current_branch", lambda _root=None: "main")

    result = branch_guard(owner="agent-branch", repo_root=tmp_path)
    assert result["allow"] is False
    assert result.get("issue") == 77
    assert "main" in result.get("reason", "")


def test_branch_guard_denies_multiple_claims_without_issue_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "agent-multi")
    monkeypatch.delenv("ORBIT_WARS_ISSUE_ID", raising=False)
    monkeypatch.setenv("ORBIT_WARS_BRANCH_ENFORCE", "1")
    roadmap_claims.claim_issue(
        issue=10,
        owner="agent-multi",
        paths=["src/jax/"],
        branch="issue/10",
        setup_worktree=False,
    )
    roadmap_claims.claim_issue(
        issue=11,
        owner="agent-multi",
        paths=["tests/"],
        branch="issue/11",
        setup_worktree=False,
    )
    monkeypatch.setattr("scripts.roadmap_git.current_branch", lambda _root=None: "issue/10")

    result = branch_guard(owner="agent-multi", repo_root=tmp_path)
    assert result["allow"] is False
    assert "multiple open claims" in result.get("reason", "").lower()
    assert result.get("next_steps")


def test_hook_guard_blocks_src_on_main_with_branch_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims
    from scripts.roadmap import hook_guard, save_impl_gate

    state = tmp_path / "state"
    gate = state / "impl-gate.json"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    monkeypatch.setenv("ORBIT_WARS_IMPL_GATE", "1")
    monkeypatch.setenv("ORBIT_WARS_AGENT_ID", "hook-branch-agent")
    monkeypatch.setenv("ORBIT_WARS_ISSUE_ID", "88")
    monkeypatch.setattr("scripts.roadmap.IMPL_GATE_PATH", gate)
    save_impl_gate({"approved": True, "issue": "#88", "summary": "test"})
    roadmap_claims.claim_issue(
        issue=88,
        owner="hook-branch-agent",
        paths=["src/jax/"],
        branch="issue/88",
        setup_worktree=False,
    )
    monkeypatch.setattr("scripts.roadmap_git.current_branch", lambda _root=None: "main")

    result = hook_guard(paths=["src/jax/train.py"])
    assert result["allow"] is False
    assert "main" in result.get("reason", "").lower()


def test_git_push_guard_blocks_explicit_issue_branch() -> None:
    from scripts.roadmap_git import git_push_guard

    result = git_push_guard("git push -u origin issue/127-telemetry-cleanup")
    assert result["allow"] is False
    assert "land-issue" in result.get("reason", "")


def test_git_push_guard_allows_main_push() -> None:
    from scripts.roadmap_git import git_push_guard

    result = git_push_guard("git push origin main")
    assert result["allow"] is True


def test_git_push_guard_blocks_bare_push_on_issue_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.roadmap_git import git_push_guard

    monkeypatch.setattr(
        "scripts.roadmap_git.current_branch",
        lambda _root=None: "issue/127-slug",
    )
    result = git_push_guard("git push -u origin HEAD")
    assert result["allow"] is False


def test_land_issue_dry_run_reports_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import roadmap_claims
    from scripts.roadmap_git import land_issue_branch

    state = tmp_path / "state"
    monkeypatch.setenv("ORBIT_WARS_STATE_DIR", str(state))
    roadmap_claims.claim_issue(
        issue=60,
        owner="land-agent",
        paths=["docs/"],
        branch="issue/60-land-test",
        setup_worktree=False,
    )
    payload = land_issue_branch(60, repo_root=tmp_path, dry_run=True)
    assert payload["status"] == "planned"
    assert payload["branch"] == "issue/60-land-test"
    assert payload["dry_run"] is True
    assert payload["branch_exists"] is False
