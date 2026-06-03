"""Submit-valid funnel: Docker packaging validation before tournament ladder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.artifacts.docker_validation import run_docker_validation_subprocess

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
