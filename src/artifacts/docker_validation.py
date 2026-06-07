"""Run Kaggle Docker submission validation for a checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.artifacts.run_paths import atomic_write_json

DEFAULT_DOCKER_IMAGE = "gcr.io/kaggle-images/python-simulations"


def run_submit_valid_docker_gate(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    repo_root: Path,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    seed: int = 42,
    player_count: str = "both",
    per_step_seconds: float = 1.0,
    overage_budget_seconds: float = 60.0,
    episode_steps: int = 500,
) -> dict[str, Any]:
    """Validate checkpoint packaging in Kaggle Docker before tournament work."""

    return run_docker_validation_subprocess(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        repo_root=repo_root,
        docker_image=docker_image,
        seed=seed,
        player_count=player_count,
        per_step_seconds=per_step_seconds,
        overage_budget_seconds=overage_budget_seconds,
        episode_steps=episode_steps,
    )


def docker_gate_passed(manifest: dict[str, Any]) -> bool:
    """Return whether a Docker gate manifest indicates submit-valid packaging."""

    return bool(manifest.get("validation_ok"))


def run_docker_gate_for_job(
    job: dict[str, object],
    *,
    result_dir: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], bool]:
    """Run docker validation for a queued job and write ``docker_manifest.json``."""

    checkpoint_path = Path(str(job["checkpoint_path"]))
    docker_output_dir = result_dir / "docker_validation"
    docker_manifest = run_submit_valid_docker_gate(
        checkpoint_path=checkpoint_path,
        output_dir=docker_output_dir,
        repo_root=repo_root,
        docker_image=str(job.get("docker_image", DEFAULT_DOCKER_IMAGE)),
        seed=int(job.get("seed", 42)),
        player_count=str(job.get("player_count", "both")),
        per_step_seconds=float(job.get("per_step_seconds", 1.0)),
        overage_budget_seconds=float(job.get("overage_budget_seconds", 60.0)),
        episode_steps=int(job.get("episode_steps", 500)),
    )
    atomic_write_json(result_dir / "docker_manifest.json", docker_manifest)
    return docker_manifest, docker_gate_passed(docker_manifest)


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
        float(episode_steps) * float(per_step_seconds)
        + float(overage_budget_seconds)
        + 120.0,
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
