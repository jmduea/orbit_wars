"""Unit tests for submit-valid Docker gate helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifacts.docker_validation import (
    DEFAULT_DOCKER_IMAGE,
    docker_gate_passed,
    run_docker_gate_for_job,
    run_docker_validation_subprocess,
)


def test_docker_gate_passed_requires_validation_ok_true() -> None:
    assert docker_gate_passed({"validation_ok": True}) is True
    assert docker_gate_passed({"validation_ok": False}) is False
    assert docker_gate_passed({}) is False
    assert docker_gate_passed({"validation_ok": None}) is False


@patch("src.artifacts.docker_validation.run_submit_valid_docker_gate")
def test_run_docker_gate_for_job_writes_manifest_and_forwards_job_fields(
    mock_gate: object,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt.pkl"
    checkpoint.write_bytes(b"stub")
    result_dir = tmp_path / "eval" / "checkpoint_eval_u10"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest = {
        "validation_ok": True,
        "output_dir": str(result_dir / "docker_validation"),
    }
    mock_gate.return_value = manifest

    returned_manifest, ok = run_docker_gate_for_job(
        {
            "checkpoint_path": str(checkpoint),
            "docker_image": "custom/image:tag",
            "seed": 7,
            "player_count": "2",
            "per_step_seconds": 0.5,
            "overage_budget_seconds": 30.0,
            "episode_steps": 100,
        },
        result_dir=result_dir,
        repo_root=repo_root,
    )

    assert returned_manifest == manifest
    assert ok is True
    mock_gate.assert_called_once_with(
        checkpoint_path=checkpoint,
        output_dir=result_dir / "docker_validation",
        repo_root=repo_root,
        docker_image="custom/image:tag",
        seed=7,
        player_count="2",
        per_step_seconds=0.5,
        overage_budget_seconds=30.0,
        episode_steps=100,
    )
    written = json.loads(
        (result_dir / "docker_manifest.json").read_text(encoding="utf-8")
    )
    assert written == manifest


@patch("src.artifacts.docker_validation.run_submit_valid_docker_gate")
def test_run_docker_gate_for_job_uses_defaults_for_missing_job_fields(
    mock_gate: object,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt.pkl"
    checkpoint.write_bytes(b"stub")
    result_dir = tmp_path / "result"
    repo_root = tmp_path
    mock_gate.return_value = {"validation_ok": False}

    _, ok = run_docker_gate_for_job(
        {"checkpoint_path": str(checkpoint)},
        result_dir=result_dir,
        repo_root=repo_root,
    )

    assert ok is False
    mock_gate.assert_called_once_with(
        checkpoint_path=checkpoint,
        output_dir=result_dir / "docker_validation",
        repo_root=repo_root,
        docker_image=DEFAULT_DOCKER_IMAGE,
        seed=42,
        player_count="both",
        per_step_seconds=1.0,
        overage_budget_seconds=60.0,
        episode_steps=500,
    )


def test_run_docker_validation_subprocess_raises_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "ckpt.pkl"
    checkpoint.write_bytes(b"stub")
    output_dir = tmp_path / "docker_out"
    repo_root = tmp_path

    def fake_run(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stderr.log").write_text("boom", encoding="utf-8")
        from types import SimpleNamespace

        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("src.artifacts.docker_validation.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 1"):
        run_docker_validation_subprocess(
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            repo_root=repo_root,
            docker_image=DEFAULT_DOCKER_IMAGE,
            seed=42,
            player_count="both",
            per_step_seconds=1.0,
            overage_budget_seconds=60.0,
            episode_steps=500,
        )
