#!/usr/bin/env python3
"""Git branch and worktree helpers for parallel ROADMAP agents."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKTREES_ROOT = REPO_ROOT / "worktrees"
PROTECTED_BASE_BRANCHES = frozenset({"main", "master"})


def issue_branch_name(issue: int, slug: str | None = None) -> str:
    """Default branch name for an issue (``issue/NN`` or ``issue/NN-slug``)."""
    if slug:
        cleaned = slugify(slug)
        return f"issue/{issue}-{cleaned}" if cleaned else f"issue/{issue}"
    return f"issue/{issue}"


def slugify(text: str, *, max_len: int = 48) -> str:
    """Normalize arbitrary text into a git-branch-safe slug segment."""
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if not cleaned:
        return ""
    return cleaned[:max_len].rstrip("-")


def worktree_dir(issue: int) -> Path:
    """Filesystem path for an issue-isolated worktree (gitignored)."""
    return WORKTREES_ROOT / f"issue-{issue}"


def current_branch(repo_root: Path | None = None) -> str | None:
    """Return the current branch name, or None if not in a git repo."""
    root = repo_root or REPO_ROOT
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    branch = proc.stdout.strip()
    return branch or None


def branch_exists(branch: str, repo_root: Path | None = None) -> bool:
    root = repo_root or REPO_ROOT
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            capture_output=True,
            check=True,
            cwd=root,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def is_protected_base_branch(branch: str | None) -> bool:
    return branch in PROTECTED_BASE_BRANCHES if branch else False


def effective_claim_branch(claim: dict) -> str:
    """Branch recorded on the claim, or the canonical default ``issue/NN``."""
    stored = claim.get("branch")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    return issue_branch_name(int(claim["issue"]))


def setup_issue_worktree(
    issue: int,
    branch: str | None = None,
    *,
    base: str = "main",
    repo_root: Path | None = None,
) -> dict:
    """Create or reuse a git worktree for parallel work on ``issue``.

    Args:
        issue: GitHub issue number.
        branch: Target branch; defaults to ``issue/NN``.
        base: Branch to fork from when creating a new branch.
        repo_root: Repository root (defaults to this repo).

    Returns:
        Status payload with ``path``, ``branch``, and shell ``cd`` hint.

    Raises:
        RuntimeError: If ``git worktree`` fails.
    """
    root = repo_root or REPO_ROOT
    branch = branch or issue_branch_name(issue)
    wt_path = worktree_dir(issue)
    wt_path.parent.mkdir(parents=True, exist_ok=True)

    if wt_path.exists() and (wt_path / ".git").exists():
        actual = current_branch(wt_path) or branch
        return {
            "status": "exists",
            "issue": issue,
            "path": str(wt_path),
            "branch": actual,
            "cd": f"cd {wt_path}",
            "hint": "Open this directory in the agent session (or set workspace to the worktree path).",
        }

    cmd: list[str] = ["git", "worktree", "add"]
    if branch_exists(branch, root):
        cmd.extend([str(wt_path), branch])
        status = "attached"
    else:
        cmd.extend(["-b", branch, str(wt_path), base])
        status = "created"

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=root)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"git worktree add failed: {detail}") from exc

    return {
        "status": status,
        "issue": issue,
        "path": str(wt_path),
        "branch": branch,
        "cd": f"cd {wt_path}",
        "hint": (
            f"Set ORBIT_WARS_ISSUE_ID={issue} and ORBIT_WARS_AGENT_ID to a unique id "
            f"(e.g. cursor-issue-{issue}) in this worktree before editing src/conf/tests."
        ),
    }


def branch_guard(
    *,
    owner: str,
    repo_root: Path | None = None,
) -> dict:
    """Check whether the current git branch matches the active claim for ``owner``.

    Uses ``ORBIT_WARS_ISSUE_ID`` when set to disambiguate multiple claims per owner.
    Enforcement is controlled by ``ORBIT_WARS_BRANCH_ENFORCE`` (default on).
    """
    from scripts import roadmap_claims

    if os.environ.get("ORBIT_WARS_BRANCH_ENFORCE", "1").strip().lower() in {
        "0",
        "false",
        "no",
    }:
        return {"allow": True, "skipped": "ORBIT_WARS_BRANCH_ENFORCE=0"}

    root = repo_root or REPO_ROOT
    actual = current_branch(root)
    if actual is None:
        return {"allow": True, "skipped": "not a git checkout"}

    open_claims: list[dict] = []
    for claim in roadmap_claims.load_all_claims(active_only=True):
        if claim.get("owner") != owner:
            continue
        issue = int(claim["issue"])
        if roadmap_claims.load_completion(issue):
            continue
        open_claims.append(claim)

    issue_env = os.environ.get("ORBIT_WARS_ISSUE_ID", "").strip()
    if issue_env:
        try:
            issue_filter = int(issue_env)
        except ValueError:
            return {
                "allow": False,
                "reason": f"ORBIT_WARS_ISSUE_ID={issue_env!r} is not an integer",
                "blockers": ["Set ORBIT_WARS_ISSUE_ID to the GitHub issue number for this agent"],
            }
        open_claims = [c for c in open_claims if int(c["issue"]) == issue_filter]

    if not open_claims:
        return {"allow": True, "skipped": "no open claim for owner"}

    if len(open_claims) > 1:
        issues = ", ".join(f"#{int(c['issue'])}" for c in open_claims)
        return {
            "allow": True,
            "branch_warning": (
                f"Owner {owner!r} has multiple open claims ({issues}); "
                f"set ORBIT_WARS_ISSUE_ID=<N> per parallel agent/subagent"
            ),
        }

    claim = open_claims[0]
    issue = int(claim["issue"])
    stored_branch = claim.get("branch")
    if not (isinstance(stored_branch, str) and stored_branch.strip()):
        return {
            "allow": True,
            "branch_warning": (
                f"Claim issue #{issue} has no branch yet (grandfathered); "
                f"run: uv run python scripts/roadmap.py claim --issue {issue} "
                "--path <dirs> --setup-worktree when ready to isolate"
            ),
        }

    expected = effective_claim_branch(claim)
    wt_path = worktree_dir(issue)

    if actual == expected:
        return {"allow": True, "branch": actual, "issue": issue}

    if is_protected_base_branch(actual):
        setup_cmd = f"uv run python scripts/roadmap.py worktree --issue {issue}"
        reason = (
            f"Implementation edits blocked on protected branch {actual!r} "
            f"for claim issue #{issue} (expected {expected!r}). "
            f"Run: {setup_cmd} — then open {wt_path} as the agent workspace. "
            f"Set ORBIT_WARS_ISSUE_ID={issue} and a unique ORBIT_WARS_AGENT_ID."
        )
        return {
            "allow": False,
            "reason": reason,
            "blockers": [reason],
            "expected_branch": expected,
            "actual_branch": actual,
            "issue": issue,
            "worktree_path": str(wt_path),
            "next_steps": [
                setup_cmd,
                f"export ORBIT_WARS_ISSUE_ID={issue}",
                f"export ORBIT_WARS_AGENT_ID=cursor-issue-{issue}",
                f"cd {wt_path}",
            ],
        }

    reason = (
        f"Current branch {actual!r} does not match claim for issue #{issue} "
        f"(expected {expected!r}). Checkout the issue branch or use the issue worktree."
    )
    return {
        "allow": False,
        "reason": reason,
        "blockers": [reason],
        "expected_branch": expected,
        "actual_branch": actual,
        "issue": issue,
        "next_steps": [
            f"git checkout {expected}",
            f"uv run python scripts/roadmap.py worktree --issue {issue}",
        ],
    }
