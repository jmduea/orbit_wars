"""Package checkpoints and submit to Kaggle competitions via the Kaggle CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from scripts.validate_kaggle_docker_submission import (
    DOCKER_IMAGE,
    build_submission_package,
    run_docker_validation,
)

DEFAULT_COMPETITION = "orbit-wars"


def resolve_kaggle_executable() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "kaggle CLI not found on PATH. Install it and configure credentials "
            "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."
        )
    return executable


def package_checkpoint_submission(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    validate_docker: bool = False,
    docker_image: str = DOCKER_IMAGE,
    seed: int = 42,
    player_count: str = "both",
    per_step_seconds: float = 1.0,
    overage_budget_seconds: float = 60.0,
    episode_steps: int = 500,
) -> Path:
    """Build ``submission.tar.gz`` from a checkpoint; optionally validate in Docker."""

    args = argparse.Namespace(
        checkpoint=checkpoint_path,
        output_dir=output_dir,
        docker_image=docker_image,
        seed=seed,
        player_count=player_count,
        per_step_seconds=per_step_seconds,
        overage_budget_seconds=overage_budget_seconds,
        episode_steps=episode_steps,
        skip_docker=not validate_docker,
        keep_staging=False,
    )
    package_path = build_submission_package(args)
    if validate_docker:
        run_docker_validation(package_path, args)
    return package_path


def submit_competition_package(
    package_path: Path,
    message: str,
    *,
    competition: str = DEFAULT_COMPETITION,
    quiet: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Upload a tarball to a Kaggle competition via ``kaggle competitions submit``."""

    package_path = package_path.resolve()
    if not package_path.is_file():
        raise FileNotFoundError(f"submission package not found: {package_path}")

    command = [
        resolve_kaggle_executable(),
        "competitions",
        "submit",
        competition,
        "-f",
        str(package_path),
        "-m",
        message,
    ]
    if quiet:
        command.append("-q")
    if dry_run:
        print(" ".join(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=__import__("sys").stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"kaggle competitions submit failed with exit code {completed.returncode}"
        )
    return completed
