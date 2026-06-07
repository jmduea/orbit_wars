"""Orbit Wars CLI entrypoint (``ow``)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from src.cli.train_hosts import (
    contains_cli_help,
    is_cli_help_token,
    is_hydra_override,
    print_ow_help,
    print_train_help,
)


def _run_hydra_entry(entry: Callable[[], None], argv: list[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = argv
        entry()
    finally:
        sys.argv = original_argv


def _run_hydra_train(args: list[str]) -> None:
    from src.train import main as train_main

    _run_hydra_entry(train_main, ["ow train", *args])


def _run_train(args: list[str]) -> None:
    from src.cli import train_hosts

    route = train_hosts.parse_train_argv(args)
    train_hosts.dispatch(route)


def _run_make(args: list[str]) -> None:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "make_wandb_sweep.py"
    )
    if not script_path.exists():
        raise RuntimeError(f"Unable to find make script at {script_path}")

    cmd = [sys.executable, str(script_path), *args]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    args = sys.argv[1:]

    if args and is_cli_help_token(args[0]):
        print_ow_help()
        return

    if not args or is_hydra_override(args[0]):
        command = "train"
        command_args = args
    else:
        command = args[0]
        command_args = args[1:]

    match command:
        case "train":
            if contains_cli_help(command_args):
                print_train_help()
                return
            _run_train(command_args)

        case "make":
            _run_make(command_args)

        case "eval":
            from src.cli import eval as eval_cli

            raise SystemExit(eval_cli.main(command_args))

        case "benchmark":
            from src.cli.benchmark import main as benchmark_main

            raise SystemExit(benchmark_main(command_args))

        case "sweep":
            from src.cli.sweep import main as sweep_main

            raise SystemExit(sweep_main(command_args))

        case "runs":
            from src.cli import runs as runs_cli

            raise SystemExit(runs_cli.main(command_args))

        case "promote":
            from src.cli import promote as promote_cli

            raise SystemExit(promote_cli.main(command_args))

        case "help":
            print_ow_help()
        case _:
            raise SystemExit(
                f"Unknown ow command: {command!r}. "
                "Valid commands: train, eval, benchmark, sweep, make, runs, promote. "
                "Run: uv run ow --help"
            )
