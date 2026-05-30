from __future__ import annotations

import pytest
from hydra.errors import MissingConfigException

from src.cli import train_hosts


def test_parse_train_default_local() -> None:
    route = train_hosts.parse_train_argv([])

    assert route.host == "local"
    assert route.hydra_overrides == []


def test_parse_train_hydra_override_implies_local() -> None:
    route = train_hosts.parse_train_argv(["training=mixed_2p4p_16_total"])

    assert route.host == "local"
    assert route.hydra_overrides == ["training=mixed_2p4p_16_total"]


def test_parse_train_explicit_local() -> None:
    route = train_hosts.parse_train_argv(["local", "training.total_updates=10"])

    assert route.host == "local"
    assert route.hydra_overrides == ["training.total_updates=10"]


def test_parse_train_kaggle_launch_forwards_overrides() -> None:
    route = train_hosts.parse_train_argv(
        ["kaggle", "training=mixed_2p4p_16_total", "training.total_updates=500"]
    )

    assert route.host == "kaggle"
    assert route.subcommand is None
    assert route.hydra_overrides == [
        "training=mixed_2p4p_16_total",
        "training.total_updates=500",
    ]


def test_parse_train_kaggle_status_subcommand() -> None:
    route = train_hosts.parse_train_argv(["kaggle", "status", "owner/slug"])

    assert route.host == "kaggle"
    assert route.subcommand == "status"
    assert route.kaggle_argv == ["owner/slug"]


def test_parse_train_kaggle_sync_maps_to_sync_output() -> None:
    route = train_hosts.parse_train_argv(["kaggle", "sync", "owner/slug"])

    assert route.subcommand == "sync-output"
    assert route.kaggle_argv == ["owner/slug"]


def test_parse_train_kaggle_rejects_create_sweep() -> None:
    with pytest.raises(SystemExit, match="create-sweep"):
        train_hosts.parse_train_argv(["kaggle", "--create-sweep"])


def test_build_kaggle_argv_default_launch_standalone_p100() -> None:
    route = train_hosts.TrainRoute(
        host="kaggle",
        hydra_overrides=["training=mixed_2p4p_16_rotating"],
    )
    argv = train_hosts._build_kaggle_argv(route)

    assert argv[0] == "launch"
    assert "--no-wandb" in argv
    assert "--run-type" in argv
    assert argv[argv.index("--run-type") + 1] == "full"
    assert "--accelerator" in argv
    assert argv[argv.index("--accelerator") + 1] == "NvidiaTeslaP100"
    assert "--override" in argv
    assert argv[argv.index("--override") + 1] == "training=mixed_2p4p_16_rotating"


def test_dispatch_kaggle_calls_runner(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(argv: list[str] | None = None) -> int:
        captured.append(list(argv or []))
        return 0

    monkeypatch.setattr("src.cli.train_hosts.kaggle_cli.run", fake_run)

    train_hosts.dispatch(
        train_hosts.TrainRoute(
            host="kaggle",
            hydra_overrides=["training.total_updates=5"],
        )
    )

    assert captured == [
        [
            "launch",
            "--no-wandb",
            "--run-type",
            "full",
            "--accelerator",
            "NvidiaTeslaP100",
            "--override",
            "training.total_updates=5",
        ]
    ]


def test_dispatch_local_calls_hydra(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_train(args: list[str]) -> None:
        captured.append(args)

    monkeypatch.setattr("src.cli._run_hydra_train", fake_train)

    train_hosts.dispatch(
        train_hosts.TrainRoute(host="local", hydra_overrides=["print_resolved_config=true"])
    )

    assert captured == [["print_resolved_config=true"]]


def test_dispatch_kaggle_rejects_invalid_format_before_launch() -> None:
    route = train_hosts.parse_train_argv(
        ["kaggle", "--run-type", "smoke", "training=mixed_2p4p_16_totalb"]
    )

    with pytest.raises(MissingConfigException, match="mixed_2p4p_16_totalb"):
        train_hosts.dispatch(route)


def test_validate_hydra_overrides_lists_valid_training_options() -> None:
    with pytest.raises(MissingConfigException, match="mixed_2p4p_16_total"):
        from src.config import validate_hydra_overrides

        validate_hydra_overrides(["training=mixed_2p4p_16_totalb"])
