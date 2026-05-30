from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def _is_hydra_override(arg: str) -> bool:
    return (
        "=" in arg or arg.startswith("+") or arg.startswith("~") or arg.startswith("-")
    )


def _run_hydra_entry(entry: Callable[[], None], argv: list[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = argv
        entry()
    finally:
        sys.argv = original_argv


def _run_train(args: list[str]) -> None:
    from src.train import main as train_main

    _run_hydra_entry(train_main, ["ow train", *args])


def _run_make(args: list[str]) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "make_wandb_sweep.py"
    )
    if not script_path.exists():
        raise RuntimeError(f"Unable to find make script at {script_path}")

    cmd = [sys.executable, str(script_path), *args]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    args = sys.argv[1:]

    # Allow:
    #   uv run ow train print_resolved_config=true
    # and:
    #   uv run ow print_resolved_config=true
    if not args or _is_hydra_override(args[0]):
        command = "train"
        command_args = args
    else:
        command = args[0]
        command_args = args[1:]

    match command:
        case "train":
            _run_train(command_args)

        case "make":
            _run_make(command_args)

        case "help" | "--help" | "-h":
            print(
                "Usage:\n"
                "  uv run ow train [HYDRA_OVERRIDES...]\n"
                "  uv run ow make [MAKE_SCRIPT_OVERRIDES...]\n"
                "  uv run ow [HYDRA_OVERRIDES...]\n\n"
                "Examples:\n"
                "  uv run ow train print_resolved_config=true\n"
                "  uv run ow make wandb_sweep=shield_cheap_history\n"
                "  uv run ow train task=shield_cheap training.total_updates=10\n"
            )
        case _:
            raise SystemExit(
                f"Unknown ow command: {command!r}. Valid commands: train, make"
            )
