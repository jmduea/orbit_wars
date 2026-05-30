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
ISSUE_BRANCH_PREFIX = "issue/"


def is_issue_branch(branch: str | None) -> bool:
    """Return True when ``branch`` is an agent issue isolation branch."""
    return bool(branch and branch.startswith(ISSUE_BRANCH_PREFIX))


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
            f"(e.g. cursor-issue-{issue}) in this worktree before editing src/conf/tests. "
            f"Commit here; land on main with: roadmap.py land-issue --issue {issue} "
            "(do not git push issue/* branches)."
        ),
    }


def git_push_guard(command: str, *, repo_root: Path | None = None) -> dict:
    """Block ``git push`` of ``issue/*`` branches unless explicitly allowed.

    Agents should merge to ``main`` locally via :func:`land_issue_branch` and only
    push ``main`` when the user requests it — not publish issue branches to origin.
    """
    if os.environ.get("ORBIT_WARS_ALLOW_ISSUE_BRANCH_PUSH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return {"allow": True, "skipped": "ORBIT_WARS_ALLOW_ISSUE_BRANCH_PUSH=1"}

    stripped = command.strip()
    if not re.search(r"\bgit\s+push\b", stripped):
        return {"allow": True, "skipped": "not a git push"}

    # Explicit push of integration branches is allowed (user may request main push).
    if re.search(r"\borigin\s+(main|master)\b", stripped) or re.search(
        r"\bpush\s+(main|master)\b", stripped
    ):
        return {"allow": True, "skipped": "push targets main/master"}

    # Explicit issue/* ref in the push command.
    for match in re.finditer(r"(?:origin\s+)?(issue/[a-zA-Z0-9_./-]+)", stripped):
        branch = match.group(1)
        reason = (
            f"git push of issue branch {branch!r} blocked. "
            f"Commit locally in the worktree, then from repo root run: "
            f"uv run python scripts/roadmap.py land-issue --issue <N>. "
            f"Push main only when the user asks. "
            f"Override: ORBIT_WARS_ALLOW_ISSUE_BRANCH_PUSH=1"
        )
        return {
            "allow": False,
            "reason": reason,
            "blockers": [reason],
            "branch": branch,
            "next_steps": [
                "uv run python scripts/roadmap.py land-issue --issue N",
                "git push origin main   # only when user requests push",
            ],
        }

    # Bare `git push` / `git push -u origin HEAD` from an issue worktree.
    if re.search(r"\bHEAD\b", stripped) or re.search(r"\bgit\s+push(\s|$)", stripped):
        root = repo_root or REPO_ROOT
        cwd_branch = current_branch(root)
        if is_issue_branch(cwd_branch):
            reason = (
                f"git push from issue branch {cwd_branch!r} blocked. "
                f"Use land-issue to merge into main first."
            )
            return {
                "allow": False,
                "reason": reason,
                "blockers": [reason],
                "branch": cwd_branch,
                "next_steps": [
                    "uv run python scripts/roadmap.py land-issue --issue N",
                ],
            }

    return {"allow": True}


def land_issue_branch(
    issue: int,
    *,
    base: str = "main",
    repo_root: Path | None = None,
    dry_run: bool = False,
    ff_only: bool = False,
) -> dict:
    """Merge an issue worktree branch into ``base`` at the repo root.

    Args:
        issue: GitHub issue number.
        base: Target branch (default ``main``).
        repo_root: Repository root checkout (not the worktree path).
        dry_run: When true, report planned merge without mutating git state.
        ff_only: Use ``git merge --ff-only`` instead of ``--no-ff``.

    Returns:
        Status payload with merge result and push guidance.

    Raises:
        RuntimeError: When merge prerequisites fail.
    """
    from scripts import roadmap_claims

    root = repo_root or REPO_ROOT
    claim = roadmap_claims.load_claim(issue)
    branch = effective_claim_branch(claim) if claim else issue_branch_name(issue)
    wt_path = worktree_dir(issue)
    exists = branch_exists(branch, root)

    payload: dict = {
        "issue": issue,
        "branch": branch,
        "base": base,
        "worktree_path": str(wt_path),
        "dry_run": dry_run,
        "branch_exists": exists,
    }

    if dry_run:
        payload["status"] = "planned"
        if not exists:
            payload["warning"] = (
                f"Branch {branch!r} not found yet; commit in {wt_path} before landing."
            )
        payload["hint"] = (
            f"Would checkout {base} at repo root and merge {branch!r}. "
            "Run without --dry-run to apply."
        )
        return payload

    if not exists:
        raise RuntimeError(
            f"Branch {branch!r} not found. Commit in {wt_path} on the issue branch first."
        )

    try:
        subprocess.run(
            ["git", "checkout", base],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"git checkout {base} failed: {detail}") from exc

    merge_cmd = ["git", "merge", branch]
    if ff_only:
        merge_cmd.append("--ff-only")
    else:
        merge_cmd.extend(["--no-ff", "-m", f"Merge {branch} (closes #{issue})"])

    try:
        proc = subprocess.run(
            merge_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"git merge {branch} into {base} failed: {detail}. "
            "Resolve conflicts at repo root, commit, then wrap-up."
        ) from exc

    merge_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout.strip()

    payload.update(
        {
            "status": "merged",
            "merge_commit": merge_commit,
            "current_branch": current_branch(root),
            "merge_output": (proc.stdout or proc.stderr or "").strip(),
            "next_steps": [
                "make test-fast (or domain tests) on main at repo root",
                "ROADMAP Done row + make roadmap-check",
                "gh issue close "
                f"{issue} --comment 'Evidence: …' then roadmap.py wrap-up --issue "
                f"{issue}",
                "git push origin main   # only when the user explicitly requests push",
            ],
            "hint": (
                "Do not push the issue branch to origin. Main now contains the merge; "
                "push main only on user request."
            ),
        }
    )
    return payload


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

    if len(open_claims) > 1 and not issue_env:
        issues = ", ".join(f"#{int(c['issue'])}" for c in open_claims)
        reason = (
            f"Owner {owner!r} has multiple open claims ({issues}); "
            f"set ORBIT_WARS_ISSUE_ID=<N> per parallel agent/subagent"
        )
        return {
            "allow": False,
            "reason": reason,
            "blockers": [reason],
            "next_steps": [
                "export ORBIT_WARS_ISSUE_ID=<issue-number>",
                "export ORBIT_WARS_AGENT_ID=<unique-id>",
                "uv run python scripts/roadmap.py claim --issue N --path … --setup-worktree",
                "cd worktrees/issue-N/",
            ],
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
