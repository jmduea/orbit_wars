"""Shared git metadata helpers for benchmark CLIs and calibration writers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_head_sha(repo_root: Path) -> str | None:
    """Return the current ``HEAD`` commit SHA for ``repo_root``, or ``None`` on failure."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_identity(repo_root: Path) -> dict[str, object]:
    """Return ``{"commit": sha|None, "dirty": bool|None}`` for ``repo_root``."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"commit": None, "dirty": None}
