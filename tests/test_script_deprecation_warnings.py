"""Direct script invocation warns agents to prefer ow eval primitives."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_script_main_prints_ow_eval_preference() -> None:
    script = REPO_ROOT / "scripts" / "validate_kaggle_docker_submission.py"
    completed = subprocess.run(
        [sys.executable, str(script), "-h"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "ow eval package" in completed.stderr
    assert "--validate-docker" in completed.stderr


def test_run_artifact_worker_main_prints_ow_eval_worker_preference() -> None:
    script = REPO_ROOT / "scripts" / "run_artifact_worker.py"
    completed = subprocess.run(
        [sys.executable, str(script), "-h"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "ow eval worker" in completed.stderr
