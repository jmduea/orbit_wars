"""W&B run → checkpoint resolution for SSOT packaging."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.artifacts.tournament.resolve import (
    parse_wandb_run_ref,
    resolve_checkpoint_from_wandb_run,
)


def test_parse_wandb_run_ref_accepts_entity_project_run() -> None:
    assert parse_wandb_run_ref("ent/proj/run123") == ("ent", "proj", "run123")


def test_parse_wandb_run_ref_rejects_short_paths() -> None:
    with pytest.raises(ValueError, match="entity/project/run_id"):
        parse_wandb_run_ref("ent/run")


def test_resolve_checkpoint_from_wandb_run_downloads_newest(tmp_path: Path) -> None:
    artifact_old = MagicMock()
    artifact_old.type = "checkpoint"
    artifact_old.name = "checkpoint-u10"
    artifact_old.metadata = {"update": 10}

    artifact_new = MagicMock()
    artifact_new.type = "checkpoint"
    artifact_new.name = "checkpoint-u50"
    artifact_new.metadata = {"update": 50}

    run = MagicMock()
    run.logged_artifacts.return_value = [artifact_old, artifact_new]

    api = MagicMock()
    api.run.return_value = run

    expected = tmp_path / "jax_ckpt.pkl"
    expected.write_bytes(b"ok")

    with (
        patch("wandb.Api", return_value=api),
        patch(
            "src.artifacts.tournament.resolve.download_wandb_checkpoint_artifact",
            return_value=expected,
        ) as download_mock,
    ):
        path = resolve_checkpoint_from_wandb_run(
            "ent/proj/run123",
            tmp_path / "cache",
        )

    assert path == expected
    download_mock.assert_called_once()
    assert download_mock.call_args[0][0] == "checkpoint-u50"
