"""Run Kaggle Docker submission validation for a checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_docker_validation_subprocess(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    repo_root: Path,
    docker_image: str,
    seed: int,
    player_count: str,
    per_step_seconds: float,
    overage_budget_seconds: float,
    episode_steps: int,
) -> dict[str, Any]:
    """Package and validate a checkpoint in Kaggle Docker. Raises on failure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "validate_kaggle_docker_submission.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--docker-image",
        docker_image,
        "--seed",
        str(seed),
        "--player-count",
        player_count,
        "--per-step-seconds",
        str(per_step_seconds),
        "--overage-budget-seconds",
        str(overage_budget_seconds),
        "--episode-steps",
        str(episode_steps),
    ]
    timeout_seconds = max(
        120.0,
        float(episode_steps) * float(per_step_seconds) + float(overage_budget_seconds) + 120.0,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
            env=dict(os.environ),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Docker validation timed out after {timeout_seconds:.0f}s; "
            f"see {output_dir}"
        ) from exc
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Docker validation failed with exit code {completed.returncode}; "
            f"see {output_dir / 'stderr.log'}"
        )
    replay_html_paths = sorted(
        str(path) for path in (output_dir / "replays").glob("*.html")
    )
    package_path = output_dir / "submission.tar.gz"
    return {
        "validation_ok": True,
        "output_dir": str(output_dir),
        "package_path": str(package_path) if package_path.exists() else None,
        "replay_html_paths": replay_html_paths,
        "stdout_path": str(output_dir / "stdout.log"),
        "stderr_path": str(output_dir / "stderr.log"),
    }
