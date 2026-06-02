from __future__ import annotations

import json

from src.cli import sweep as sweep_cli


def test_sweep_parser_create_wandb() -> None:
    parser = sweep_cli.build_parser()
    args = parser.parse_args(
        ["create", "--backend", "wandb", "--yaml", "outputs/_meta/sweeps/test.yaml"]
    )
    assert args.command == "create"
    assert args.backend == "wandb"


def test_sweep_help_exits_zero(capsys) -> None:
    assert sweep_cli.main([]) == 0
    assert "ow sweep" in capsys.readouterr().out


def test_sweep_create_kaggle_dry_run(capsys) -> None:
    assert (
        sweep_cli.main(
            [
                "create",
                "--backend",
                "kaggle",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "kaggle"
    assert payload["dry_run"] is True
