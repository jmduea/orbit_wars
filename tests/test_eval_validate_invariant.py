"""#161: user-facing validate paths must invoke Docker validation, not layout-only."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifacts.kaggle_submission import package_checkpoint_submission
from src.cli import eval as eval_cli


def test_package_checkpoint_submission_calls_docker_when_validate_enabled(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"unused")
    output_dir = tmp_path / "out"
    package_path = output_dir / "submission.tar.gz"

    with (
        patch(
            "src.artifacts.kaggle_submission.build_submission_package",
            return_value=package_path,
        ) as build_mock,
        patch(
            "src.artifacts.kaggle_submission.run_docker_validation",
        ) as docker_mock,
    ):
        result = package_checkpoint_submission(
            checkpoint,
            output_dir,
            validate_docker=True,
        )

    assert result == package_path
    build_mock.assert_called_once()
    docker_mock.assert_called_once()
    assert docker_mock.call_args.args[0] == package_path


def test_package_checkpoint_submission_skips_docker_when_validate_disabled(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"unused")
    output_dir = tmp_path / "out"
    package_path = output_dir / "submission.tar.gz"

    with (
        patch(
            "src.artifacts.kaggle_submission.build_submission_package",
            return_value=package_path,
        ),
        patch(
            "src.artifacts.kaggle_submission.run_docker_validation",
        ) as docker_mock,
    ):
        result = package_checkpoint_submission(
            checkpoint,
            output_dir,
            validate_docker=False,
        )

    assert result == package_path
    docker_mock.assert_not_called()


def test_eval_package_cli_propagates_validate_docker(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt.pkl"
    checkpoint.write_bytes(b"unused")
    output_dir = tmp_path / "out"

    with patch(
        "src.cli.eval.package_checkpoint_submission",
        return_value=output_dir / "submission.tar.gz",
    ) as package_mock:
        exit_code = eval_cli.main(
            [
                "package",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--validate-docker",
            ]
        )

    assert exit_code == 0
    assert package_mock.call_args.kwargs["validate_docker"] is True


def test_eval_submit_cli_propagates_validate_docker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "jax_ckpt.pkl"
    checkpoint.write_bytes(b"unused")
    output_dir = tmp_path / "out"
    package_path = output_dir / "submission.tar.gz"

    with (
        patch(
            "src.cli.eval.package_checkpoint_submission",
            return_value=package_path,
        ) as package_mock,
        patch("src.cli.eval.submit_competition_package"),
    ):
        exit_code = eval_cli.main(
            [
                "submit",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--validate-docker",
                "--dry-run",
            ]
        )

    assert exit_code == 0
    assert package_mock.call_args.kwargs["validate_docker"] is True
    assert "docker_validation=skipped" not in capsys.readouterr().err
