"""Host routing for ``ow train`` (local Hydra vs Kaggle launcher)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from src.cli import kaggle_runner as kaggle_cli

HOSTS = frozenset({"local", "kaggle"})
KAGGLE_SUBCOMMANDS = frozenset(
    {
        "preflight",
        "prepare",
        "status",
        "sync",
        "sync-output",
        "shortlist",
        "latest-checkpoint",
    }
)

KAGGLE_FLAGS = frozenset(
    {
        "--host",
        "--run-type",
        "--accelerator",
        "--kernel-id",
        "--dry-run",
        "--work-dir",
        "--title",
        "--sweep-yaml",
        "--timeout-seconds",
        "--ledger",
        "--force",
        "--no-accelerator-flag-fallback",
        "--calibration-max-variants",
        "--calibration-warmup",
        "--calibration-updates",
        "--calibration-timeout-seconds",
    }
)

KAGGLE_FLAG_VALUES = 1  # most flags take one value; --host, --run-type, etc.


def _is_hydra_override(arg: str) -> bool:
    return (
        "=" in arg or arg.startswith("+") or arg.startswith("~") or arg.startswith("-")
    )


@dataclass(slots=True)
class TrainRoute:
    """Parsed ``ow train`` route."""

    host: str = "local"
    subcommand: str | None = None
    kaggle_argv: list[str] = field(default_factory=list)
    hydra_overrides: list[str] = field(default_factory=list)


def parse_train_argv(args: list[str]) -> TrainRoute:
    """Parse train arguments into a host route."""

    if not args:
        return TrainRoute()

    index = 0
    host = "local"

    if args[0] == "--host":
        if len(args) < 2:
            raise SystemExit("--host requires local or kaggle")
        host = args[1]
        index = 2
    elif args[0] in HOSTS:
        host = args[0]
        index = 1
    elif _is_hydra_override(args[0]):
        return TrainRoute(hydra_overrides=list(args))

    if host not in HOSTS:
        raise SystemExit(
            f"Unknown train host {host!r}. Valid hosts: {', '.join(sorted(HOSTS))}"
        )

    if host == "local":
        return TrainRoute(host="local", hydra_overrides=args[index:])

    remaining = list(args[index:])
    if "--create-sweep" in remaining:
        raise SystemExit(
            "ow train kaggle does not support --create-sweep; use the standalone "
            "kaggle_runner script directly for W&B sweep creation."
        )
    subcommand: str | None = None
    if remaining and remaining[0] in KAGGLE_SUBCOMMANDS:
        subcommand = remaining[0]
        if subcommand == "sync":
            subcommand = "sync-output"
        remaining = remaining[1:]

    kaggle_argv, hydra_overrides = _split_kaggle_remaining(remaining, subcommand)

    return TrainRoute(
        host="kaggle",
        subcommand=subcommand,
        kaggle_argv=kaggle_argv,
        hydra_overrides=hydra_overrides,
    )


def _split_kaggle_remaining(
    remaining: list[str], subcommand: str | None
) -> tuple[list[str], list[str]]:
    """Split Kaggle CLI flags from Hydra overrides."""

    kaggle_argv: list[str] = []
    hydra_overrides: list[str] = []
    index = 0
    while index < len(remaining):
        token = remaining[index]
        if _is_hydra_override(token) and not token.startswith("--"):
            hydra_overrides.extend(remaining[index:])
            break
        if token == "--override":
            if index + 1 >= len(remaining):
                raise SystemExit("--override requires KEY=VALUE")
            kaggle_argv.extend([token, remaining[index + 1]])
            index += 2
            continue
        if token in {
            "--run-type",
            "--accelerator",
            "--kernel-id",
            "--work-dir",
            "--title",
            "--sweep-yaml",
            "--timeout-seconds",
            "--ledger",
            "--output-dir",
            "--calibration-max-variants",
            "--calibration-warmup",
            "--calibration-updates",
            "--calibration-timeout-seconds",
        }:
            if index + 1 >= len(remaining):
                raise SystemExit(f"{token} requires a value")
            kaggle_argv.extend([token, remaining[index + 1]])
            index += 2
            continue
        if token in {"--dry-run", "--force", "--no-accelerator-flag-fallback"}:
            kaggle_argv.append(token)
            index += 1
            continue
        if token.startswith("--"):
            raise SystemExit(f"Unknown kaggle flag for ow train kaggle: {token}")
        if subcommand in {"status", "sync-output"} and not kaggle_argv:
            kaggle_argv.append(token)
            index += 1
            continue
        if _is_hydra_override(token):
            hydra_overrides.append(token)
            index += 1
            continue
        hydra_overrides.append(token)
        index += 1
    return kaggle_argv, hydra_overrides


def _build_kaggle_argv(route: TrainRoute) -> list[str]:
    """Assemble argv for ``kaggle_runner.run`` from a train route."""

    argv: list[str] = []
    command = route.subcommand or "launch"
    argv.append(command)

    if command == "launch" and route.subcommand is None:
        argv.extend(["--no-wandb"])
        if "--run-type" not in route.kaggle_argv:
            argv.extend(["--run-type", "full"])
        if "--accelerator" not in route.kaggle_argv:
            argv.extend(["--accelerator", "NvidiaTeslaP100"])

    argv.extend(route.kaggle_argv)

    for override in route.hydra_overrides:
        argv.extend(["--override", override])

    if command in {"preflight", "prepare", "launch"} and "--no-wandb" not in argv:
        argv.append("--no-wandb")

    return argv


def dispatch(route: TrainRoute) -> None:
    """Dispatch a parsed train route to local Hydra or Kaggle runner."""

    if route.host == "local":
        from src.cli import _run_hydra_train

        _run_hydra_train(route.hydra_overrides)
        return

    argv = _build_kaggle_argv(route)
    code = kaggle_cli.run(argv)
    if code != 0:
        raise SystemExit(code)
