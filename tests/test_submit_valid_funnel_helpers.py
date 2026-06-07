"""Direct tests for submit-valid funnel helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.artifacts.submit_valid_funnel import (
    DEFAULT_DOCKER_IMAGE,
    docker_gate_passed,
    run_submit_valid_docker_gate,
)


def test_docker_gate_passed_requires_validation_ok_true() -> None:
    assert docker_gate_passed({"validation_ok": True})
    assert not docker_gate_passed({"validation_ok": False})
    assert not docker_gate_passed({})
    assert not docker_gate_passed({"validation_ok": None})


@patch("src.artifacts.submit_valid_funnel.run_docker_validation_subprocess")
def test_run_submit_valid_docker_gate_forwards_defaults(mock_run) -> None:
    mock_run.return_value = {"validation_ok": True}
    result = run_submit_valid_docker_gate(
        checkpoint_path=Path("/tmp/ckpt.pkl"),
        output_dir=Path("/tmp/out"),
        repo_root=Path("/repo"),
    )
    assert result["validation_ok"] is True
    kwargs = mock_run.call_args.kwargs
    assert kwargs["docker_image"] == DEFAULT_DOCKER_IMAGE
    assert kwargs["seed"] == 42
    assert kwargs["episode_steps"] == 500
