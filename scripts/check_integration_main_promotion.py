from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_WORKTREE = REPO_ROOT.parent / "orbit_wars"
DEFAULT_BACKUP_BRANCH = "backup/main-pre-integration-20260616"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str


def _run_git(
    repo: Path, args: list[str], *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_text(repo: Path, args: list[str]) -> str:
    result = _run_git(repo, args, check=True)
    return result.stdout.strip()


def _git_status(repo: Path) -> list[str]:
    result = _run_git(repo, ["status", "--short"], check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _remote_ref(repo: Path, ref: str) -> str | None:
    result = _run_git(repo, ["rev-parse", "--verify", "--quiet", ref])
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _check_clean_status(repo: Path, *, allow_paths: set[str]) -> Check:
    lines = _git_status(repo)
    unexpected: list[str] = []
    for line in lines:
        path = line[3:] if len(line) > 3 else line
        if path not in allow_paths:
            unexpected.append(line)
    if unexpected:
        return Check(
            "integration_worktree_clean",
            "warn",
            "unexpected dirty entries: " + "; ".join(unexpected),
        )
    if lines:
        return Check(
            "integration_worktree_clean",
            "ok",
            "only allowed dirty entries: " + "; ".join(lines),
        )
    return Check("integration_worktree_clean", "ok", "clean")


def build_report(
    *,
    integration_repo: Path,
    main_worktree: Path,
    backup_branch: str,
    allow_dirty: set[str],
) -> dict[str, Any]:
    head = _git_text(integration_repo, ["rev-parse", "HEAD"])
    origin_main = _git_text(integration_repo, ["rev-parse", "origin/main"])
    local_main = _git_text(integration_repo, ["rev-parse", "refs/heads/main"])
    merge_base = _git_text(integration_repo, ["merge-base", "HEAD", "origin/main"])
    divergence = _git_text(
        integration_repo,
        ["rev-list", "--left-right", "--count", "HEAD...origin/main"],
    )
    backup = _git_text(integration_repo, ["rev-parse", f"refs/heads/{backup_branch}"])
    remote_integration = _remote_ref(
        integration_repo,
        "refs/remotes/origin/refactor/artifacts-metric-promotion-commit",
    )
    main_worktree_status = _git_status(main_worktree) if main_worktree.is_dir() else []
    main_worktree_branch = (
        _git_text(main_worktree, ["branch", "--show-current"])
        if main_worktree.is_dir()
        else ""
    )

    checks: list[Check] = [
        _check_clean_status(integration_repo, allow_paths=allow_dirty),
        Check(
            "backup_matches_old_main",
            "ok" if backup == origin_main == local_main else "error",
            f"backup={backup[:12]} local_main={local_main[:12]} origin_main={origin_main[:12]}",
        ),
        Check(
            "main_worktree_branch",
            "ok" if main_worktree_branch == "main" else "warn",
            f"{main_worktree}: branch={main_worktree_branch or '<missing>'}",
        ),
        Check(
            "main_worktree_clean",
            "ok" if not main_worktree_status else "warn",
            "clean" if not main_worktree_status else "; ".join(main_worktree_status),
        ),
        Check(
            "remote_integration_current",
            "ok" if remote_integration == head else "warn",
            (
                "remote integration matches HEAD"
                if remote_integration == head
                else f"remote={remote_integration[:12] if remote_integration else '<missing>'} head={head[:12]}"
            ),
        ),
    ]
    ready_for_local_ref_move = all(
        check.status == "ok"
        for check in checks
        if check.name
        in {
            "integration_worktree_clean",
            "backup_matches_old_main",
            "main_worktree_branch",
        }
    )
    ready_for_remote_promotion = all(check.status == "ok" for check in checks)
    return {
        "integration_repo": str(integration_repo),
        "main_worktree": str(main_worktree),
        "head": head,
        "origin_main": origin_main,
        "local_main": local_main,
        "merge_base": merge_base,
        "divergence_left_right": divergence,
        "backup_branch": backup_branch,
        "backup": backup,
        "remote_integration": remote_integration,
        "ready_for_local_ref_move": ready_for_local_ref_move,
        "ready_for_remote_promotion": ready_for_remote_promotion,
        "checks": [check.__dict__ for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only readiness check for promoting integration to main.",
    )
    parser.add_argument("--integration-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--main-worktree", type=Path, default=DEFAULT_MAIN_WORKTREE)
    parser.add_argument("--backup-branch", default=DEFAULT_BACKUP_BRANCH)
    parser.add_argument(
        "--allow-dirty",
        action="append",
        default=["COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md"],
        help="Integration worktree path allowed to be dirty during readiness checks.",
    )
    args = parser.parse_args()

    report = build_report(
        integration_repo=args.integration_repo.resolve(),
        main_worktree=args.main_worktree.resolve(),
        backup_branch=str(args.backup_branch),
        allow_dirty=set(args.allow_dirty),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_remote_promotion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
