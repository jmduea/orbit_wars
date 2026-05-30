#!/usr/bin/env python3
"""Issue claims, wrap-up checks, and completion records for ROADMAP workflow."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def state_dir() -> Path:
    env = os.environ.get("ORBIT_WARS_STATE_DIR", "").strip()
    return Path(env) if env else REPO_ROOT / ".omg" / "state"


def claims_dir() -> Path:
    env = os.environ.get("ORBIT_WARS_CLAIMS_DIR", "").strip()
    return Path(env) if env else state_dir() / "claims"


def completions_dir() -> Path:
    env = os.environ.get("ORBIT_WARS_COMPLETIONS_DIR", "").strip()
    return Path(env) if env else state_dir() / "completions"


CLAIM_TTL_HOURS = 168
MIN_EVIDENCE_CHARS = 40


def default_agent_owner() -> str:
    return (
        os.environ.get("ORBIT_WARS_AGENT_ID", "").strip()
        or os.environ.get("USER", "").strip()
        or "anonymous"
    )


def _claim_path(issue: int) -> Path:
    return claims_dir() / f"issue-{issue}.json"


def _completion_path(issue: int) -> Path:
    return completions_dir() / f"issue-{issue}.json"


def _normalize_path_prefix(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if cleaned in {"", "."}:
        return ""
    return cleaned.rstrip("/") + "/"


def paths_overlap(left: list[str], right: list[str]) -> bool:
    left_norm = [_normalize_path_prefix(p) for p in left if p.strip()]
    right_norm = [_normalize_path_prefix(p) for p in right if p.strip()]
    for a in left_norm:
        for b in right_norm:
            if a.startswith(b) or b.startswith(a):
                return True
    return False


def load_claim(issue: int) -> dict | None:
    path = _claim_path(issue)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def load_all_claims(*, active_only: bool = True) -> list[dict]:
    if not claims_dir().exists():
        return []
    now = datetime.now(timezone.utc)
    claims: list[dict] = []
    for path in sorted(claims_dir().glob("issue-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        expires_raw = data.get("expires_at")
        if active_only and isinstance(expires_raw, str):
            try:
                expires = datetime.fromisoformat(expires_raw.replace("Z", "+00:00"))
                if expires < now:
                    continue
            except ValueError:
                pass
        claims.append(data)
    return claims


def save_claim(payload: dict) -> None:
    issue = int(payload["issue"])
    claims_dir().mkdir(parents=True, exist_ok=True)
    _claim_path(issue).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def delete_claim(issue: int) -> None:
    path = _claim_path(issue)
    if path.exists():
        path.unlink()


def claim_issue(
    *,
    issue: int,
    owner: str,
    paths: list[str],
    branch: str | None = None,
    slug: str | None = None,
    manifest_id: str | None = None,
    setup_worktree: bool = False,
) -> dict:
    from scripts.roadmap_git import issue_branch_name, setup_issue_worktree, slugify

    if not paths:
        raise ValueError("claim requires at least one --path")

    for path in paths:
        if "," in path:
            raise ValueError(
                f"claim path {path!r} contains a comma; "
                "repeat --path for each directory instead of comma-separated paths"
            )

    resolved_branch = branch
    if not resolved_branch:
        title_slug: str | None = slug
        if not title_slug:
            gh = gh_issue_view(issue)
            if gh and gh.get("title"):
                title_slug = slugify(str(gh["title"]))
        resolved_branch = issue_branch_name(issue, title_slug)

    for existing in load_all_claims(active_only=True):
        other_issue = int(existing.get("issue", -1))
        if other_issue == issue:
            if existing.get("owner") != owner:
                raise ValueError(
                    f"Issue #{issue} already claimed by {existing.get('owner')!r}"
                )
            existing["branch"] = resolved_branch
            existing["paths"] = paths
            if manifest_id:
                existing["manifest_id"] = manifest_id
            save_claim(existing)
            payload = existing
            if setup_worktree:
                payload = {
                    **existing,
                    "worktree": setup_issue_worktree(issue, resolved_branch),
                }
            return payload
        if paths_overlap(paths, list(existing.get("paths", []))):
            raise ValueError(
                f"Path overlap with claim on #{other_issue} "
                f"({existing.get('paths')}) by {existing.get('owner')!r}"
            )
    now = datetime.now(timezone.utc)
    payload = {
        "issue": issue,
        "owner": owner,
        "paths": paths,
        "branch": resolved_branch,
        "manifest_id": manifest_id,
        "claimed_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=CLAIM_TTL_HOURS)).isoformat(),
        "wrapped_up": False,
    }
    save_claim(payload)
    if setup_worktree:
        return {**payload, "worktree": setup_issue_worktree(issue, resolved_branch)}
    return payload


def load_completion(issue: int) -> dict | None:
    path = _completion_path(issue)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def save_completion(payload: dict) -> None:
    issue = int(payload["issue"])
    completions_dir().mkdir(parents=True, exist_ok=True)
    _completion_path(issue).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def gh_issue_view(issue: int) -> dict | None:
    try:
        proc = subprocess.run(
            [
                "gh",
                "issue",
                "view",
                str(issue),
                "--json",
                "state,closedAt,title,url",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def roadmap_section_for_issue(issue: int) -> str | None:
    from scripts.roadmap import SECTION_ORDER, _extract_issue_ids, load_roadmap_text, parse_roadmap

    doc = parse_roadmap(load_roadmap_text())
    for section in SECTION_ORDER:
        for row in doc.sections.get(section, []):
            if issue in _extract_issue_ids(row.link):
                return section
    return None


def wrap_up_check(
    *,
    issue: int,
    evidence: str,
    evidence_file: str | None = None,
    owner: str | None = None,
    skip_github: bool = False,
) -> dict:
    from scripts.roadmap import load_impl_gate

    claim = load_claim(issue)
    completion = load_completion(issue)
    blockers: list[str] = []
    warnings: list[str] = []

    if completion and completion.get("wrapped_up"):
        return {
            "passed": True,
            "issue": issue,
            "already_complete": True,
            "completion": completion,
            "blockers": [],
            "warnings": ["Issue already has a completion record"],
        }

    if claim and owner and claim.get("owner") != owner:
        blockers.append(
            f"Claim owner {claim.get('owner')!r} != {owner!r}; release only by claim holder"
        )

    gh_data = None if skip_github else gh_issue_view(issue)
    if gh_data is None and not skip_github:
        blockers.append(
            f"Cannot verify GitHub issue #{issue} is closed (gh unavailable). "
            "Close with: gh issue close N --comment 'evidence…' or use --skip-github-check for tests"
        )
    elif gh_data is not None:
        state = str(gh_data.get("state", "")).upper()
        if state != "CLOSED":
            blockers.append(
                f"GitHub issue #{issue} state is {state!r}, expected CLOSED. "
                f"Close before wrap-up: gh issue close {issue}"
            )

    evidence_text = evidence.strip()
    if evidence_file:
        file_path = Path(evidence_file)
        if not file_path.is_absolute():
            file_path = REPO_ROOT / file_path
        if not file_path.exists():
            blockers.append(f"Evidence file missing: {evidence_file}")
        else:
            file_excerpt = file_path.read_text(encoding="utf-8")[:2000].strip()
            evidence_text = f"{evidence_text}\n\n{file_excerpt}".strip()
    if len(evidence_text) < MIN_EVIDENCE_CHARS:
        blockers.append(
            f"Evidence too short ({len(evidence_text)} chars, need >={MIN_EVIDENCE_CHARS}). "
            "Include tests run, commit/PR, paths changed, or verification commands."
        )

    section = roadmap_section_for_issue(issue)
    gh_closed = gh_data is not None and str(gh_data.get("state", "")).upper() == "CLOSED"
    if section == "done":
        pass
    elif section == "now":
        msg = (
            f"Issue #{issue} still under ROADMAP Now; add a Done row and remove from Now "
            "before wrap-up (then make roadmap-check)"
        )
        if gh_closed or skip_github:
            blockers.append(msg)
        else:
            warnings.append(msg)
    elif section == "next":
        msg = (
            f"Issue #{issue} still under ROADMAP Next; move to Done before wrap-up "
            "(then make roadmap-check)"
        )
        if gh_closed or skip_github:
            blockers.append(msg)
        else:
            warnings.append(msg)
    elif section == "later":
        warnings.append(
            f"Issue #{issue} is under ROADMAP Later; promote to Done when closing work"
        )
    elif section is None:
        msg = (
            f"Issue #{issue} not found in ROADMAP tables; add a Done row with [#{issue}](...) "
            "before wrap-up"
        )
        if gh_closed or skip_github:
            blockers.append(msg)
        else:
            warnings.append(msg)

    impl = load_impl_gate()
    impl_issue = impl.get("issue") if impl else None
    if impl_issue and impl_issue != f"#{issue}":
        warnings.append(f"impl-gate is for {impl_issue}, not #{issue}")

    passed = len(blockers) == 0
    return {
        "passed": passed,
        "issue": issue,
        "github": gh_data,
        "roadmap_section": section,
        "claim": claim,
        "blockers": blockers,
        "warnings": warnings,
        "evidence_preview": evidence_text[:240],
    }


def finalize_wrap_up(
    *,
    issue: int,
    evidence: str,
    evidence_file: str | None = None,
    owner: str | None = None,
    skip_github: bool = False,
) -> dict:
    from scripts.roadmap import clear_impl_gate, load_impl_gate

    result = wrap_up_check(
        issue=issue,
        evidence=evidence,
        evidence_file=evidence_file,
        owner=owner,
        skip_github=skip_github,
    )
    if not result["passed"]:
        return result

    now = datetime.now(timezone.utc).isoformat()
    completion = {
        "issue": issue,
        "owner": owner or (result.get("claim") or {}).get("owner") or default_agent_owner(),
        "evidence": evidence.strip(),
        "evidence_file": evidence_file,
        "wrapped_up": True,
        "completed_at": now,
        "github": result.get("github"),
        "roadmap_section_at_close": result.get("roadmap_section"),
    }
    save_completion(completion)

    claim = load_claim(issue)
    if claim:
        claim["wrapped_up"] = True
        claim["completed_at"] = now
        save_claim(claim)
        delete_claim(issue)

    impl = load_impl_gate()
    cleared_impl_gate = bool(impl and impl.get("issue") == f"#{issue}")
    if cleared_impl_gate:
        clear_impl_gate()

    result["completion"] = completion
    result["released_claim"] = True
    result["cleared_impl_gate"] = cleared_impl_gate
    return result


def release_issue(issue: int, owner: str, force: bool = False) -> dict:
    """Release an issue claim; requires wrap-up completion unless ``force``."""
    from scripts.roadmap import clear_impl_gate, load_impl_gate

    claim = load_claim(issue)
    if claim is None:
        return {
            "released": False,
            "issue": issue,
            "blockers": [f"No active claim for issue #{issue}"],
        }

    if claim.get("owner") != owner:
        return {
            "released": False,
            "issue": issue,
            "claim": claim,
            "blockers": [
                f"Claim owner {claim.get('owner')!r} != {owner!r}; only claim holder can release"
            ],
        }

    if not force:
        completion = load_completion(issue)
        if not (completion and completion.get("wrapped_up")):
            return {
                "released": False,
                "issue": issue,
                "claim": claim,
                "blockers": [
                    f"Issue #{issue} requires wrap-up before release; "
                    "run finalize_wrap_up with evidence or use force=True to abandon claim"
                ],
            }

    delete_claim(issue)

    impl = load_impl_gate()
    cleared_impl_gate = bool(impl and impl.get("issue") == f"#{issue}")
    if cleared_impl_gate:
        clear_impl_gate()

    return {
        "released": True,
        "issue": issue,
        "force": force,
        "cleared_impl_gate": cleared_impl_gate,
    }
