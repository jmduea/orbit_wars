"""SSOT packaging validation CLI flags."""

from __future__ import annotations

import pytest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.cli.eval import run_package_cli


def test_package_cli_forwards_ssot_packaging_flags(tmp_path: Path) -> None:
    ckpt = tmp_path / "jax_ckpt.pkl"
    ckpt.write_bytes(b"stub")
    out_dir = tmp_path / "out"
    args = Namespace(
        checkpoint=ckpt,
        wandb_run=None,
        output_dir=out_dir,
        validate_docker=True,
        packaging_seed=0,
        packaging_player_count="4",
        packaging_validation_marker=tmp_path / "ssot" / "packaging_validation.json",
        wandb_cache_dir=tmp_path / "cache",
    )
    with patch("src.cli.eval._eval_export") as export_mock:
        package_fn = export_mock.return_value
        package_fn.return_value = out_dir / "submission.tar.gz"
        assert run_package_cli(args) == 0
    package_fn.assert_called_once_with(
        ckpt.resolve(),
        out_dir.resolve(),
        validate_docker=True,
        seed=0,
        player_count="4",
    )


def test_package_cli_omits_packaging_kwargs_when_flags_unset(tmp_path: Path) -> None:
    ckpt = tmp_path / "jax_ckpt.pkl"
    ckpt.write_bytes(b"stub")
    out_dir = tmp_path / "out"
    args = Namespace(
        checkpoint=ckpt,
        wandb_run=None,
        output_dir=out_dir,
        validate_docker=False,
        packaging_seed=None,
        packaging_player_count=None,
        packaging_validation_marker=tmp_path / "ssot" / "packaging_validation.json",
        wandb_cache_dir=tmp_path / "cache",
    )
    with patch("src.cli.eval._eval_export") as export_mock:
        package_fn = export_mock.return_value
        package_fn.return_value = out_dir / "submission.tar.gz"
        assert run_package_cli(args) == 0
    package_fn.assert_called_once_with(
        ckpt.resolve(),
        out_dir.resolve(),
        validate_docker=False,
    )
