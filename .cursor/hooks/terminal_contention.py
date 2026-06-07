"""beforeShellExecution policy: block GPU-heavy commands when repo terminals are busy."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEAVY = re.compile(
    r"|".join(
        (
            r"\bow\s+train\b",
            r"\bow\s+benchmark\b",
            r"\bow\s+sweep\b",
            r"\bmake\s+test\b",
            r"\bmake\s+test-",
            r"\bpytest\b",
            r"\bwandb\s+agent\b",
            r"\bow\s+sweep\s+create\b",
            r"calibrate-seed",
            r"preflight-calibrate",
            r"preflight-learn-proof",
            r"test-launch-hygiene-e2e",
            r"test-sweep\b",
            r"test-premerge\b",
            r"test-full\b",
            r"test-jax\b",
            r"test-daily\b",
        )
    ),
    re.IGNORECASE,
)

GPU_HINT = (
    "One GPU — defer ow train, calibration, wandb agent, and pytest when another "
    "session is active. Inspect terminals before starting heavy work."
)


def _terminal_header_meta(text: str) -> str | None:
    parts = text.split("---", 2)
    if len(parts) < 2:
        return None
    return parts[1]


def _cwd_under_repo(cwd_raw: str, repo_root: Path) -> bool:
    try:
        cwd_path = Path(cwd_raw.strip().strip('"')).resolve()
        cwd_path.relative_to(repo_root.resolve())
        return True
    except (OSError, ValueError):
        return False


def active_terminal_commands(repo_root: Path, *, home: Path | None = None) -> list[str]:
    """Active commands in Cursor terminals whose cwd is inside repo_root.

    Includes light and heavy commands; use ``active_heavy_terminal_commands`` for
    GPU-contention policy.
    """
    projects_root = (home or Path.home()) / ".cursor" / "projects"
    if not projects_root.is_dir():
        return []

    active: list[str] = []
    resolved_root = repo_root.resolve()
    for terminals_dir in sorted(projects_root.glob("*/terminals")):
        if not terminals_dir.is_dir():
            continue
        for path in sorted(terminals_dir.glob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "exit_code:" in text:
                continue
            meta = _terminal_header_meta(text)
            if meta is None:
                continue
            if "running_for_ms:" not in meta:
                continue
            cwd_match = re.search(r'^cwd:\s*"?([^"\n]+)"?\s*$', meta, re.MULTILINE)
            if cwd_match is None or not _cwd_under_repo(
                cwd_match.group(1), resolved_root
            ):
                continue
            cmd_match = re.search(r'^command:\s*"?([^"\n]+)"?\s*$', meta, re.MULTILINE)
            cmd = cmd_match.group(1).strip() if cmd_match else path.name
            active.append(cmd)
    return active


def active_heavy_terminal_commands(
    repo_root: Path, *, home: Path | None = None
) -> list[str]:
    """Active GPU-contention commands in repo terminals (excludes light servers, git, etc.)."""
    return [cmd for cmd in active_terminal_commands(repo_root, home=home) if is_heavy_command(cmd)]


def is_heavy_command(command: str) -> bool:
    return bool(HEAVY.search(command))


def evaluate(
    command: str,
    repo_root: Path,
    *,
    home: Path | None = None,
) -> dict[str, str]:
    active_heavy = active_heavy_terminal_commands(repo_root, home=home)
    if not active_heavy or not is_heavy_command(command):
        return {"permission": "allow", "user_message": "", "agent_message": ""}

    summary = "; ".join(active_heavy[:3])
    if len(active_heavy) > 3:
        summary += f" (+{len(active_heavy) - 3} more)"
    terminals_hint = str((home or Path.home()) / ".cursor" / "projects")
    user_message = (
        "Another Cursor agent is running GPU-heavy work in this project. "
        f"Check {terminals_hint} for GPU contention before running shell commands. "
        f"{GPU_HINT}"
    )
    agent_message = (
        f"Blocked GPU/contention-heavy command because {len(active_heavy)} terminal "
        f"session(s) are still running GPU-heavy commands in this repo: {summary}. "
        f"Check terminals under {terminals_hint} before ow train / pytest / wandb agent. "
        f"{GPU_HINT}"
    )
    return {
        "permission": "deny",
        "user_message": user_message,
        "agent_message": agent_message,
    }


def main(repo_root: Path) -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print(
            json.dumps({"permission": "allow", "user_message": "", "agent_message": ""})
        )
        return 0

    command = str(payload.get("command") or "").strip()
    workspace_roots = payload.get("workspace_roots") or []
    resolved_repo = repo_root
    for root in workspace_roots:
        candidate = Path(str(root)).resolve()
        if (candidate / ".cursor" / "hooks.json").is_file():
            resolved_repo = candidate
            break

    result = evaluate(command, resolved_repo)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
