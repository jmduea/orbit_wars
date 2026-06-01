"""Host routing for ``ow train`` (local Hydra vs Kaggle launcher)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.cli import kaggle_runner as kaggle_cli
from src.orchestration.accelerators import DEFAULT_KAGGLE_ACCELERATOR

HOSTS = frozenset({"local", "kaggle"})
CLI_HELP_TOKENS = frozenset({"help", "--help", "-h", "--hydra-help"})
PRIMARY_CONFIG_GROUPS = (
    "model",
    "task",
    "reward",
    "training",
    "curriculum",
    "opponents",
    "telemetry",
    "artifacts",
)

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


def is_cli_help_token(arg: str) -> bool:
    """Return True when ``arg`` is a CLI help flag rather than a Hydra override."""

    return arg in CLI_HELP_TOKENS


def contains_cli_help(args: list[str]) -> bool:
    """Return True when argv requests CLI help."""

    return any(is_cli_help_token(arg) for arg in args)


def is_hydra_override(arg: str) -> bool:
    """Return True when ``arg`` looks like a Hydra override rather than a CLI flag."""

    if is_cli_help_token(arg):
        return False
    return (
        "=" in arg or arg.startswith("+") or arg.startswith("~") or arg.startswith("-")
    )


def _conf_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "conf"


def _primary_config_group_lines() -> list[str]:
    """List primary Hydra config groups without composing a full config."""

    lines: list[str] = []
    conf_dir = _conf_dir()
    for group in PRIMARY_CONFIG_GROUPS:
        subdir = conf_dir / group
        if not subdir.is_dir():
            continue
        options = sorted(path.stem for path in subdir.glob("*.yaml"))
        if not options:
            continue
        lines.append(f"  {group}: {', '.join(options)}")
    return lines


def _hydra_override_section() -> str:
    group_lines = _primary_config_group_lines()
    groups_block = "\n".join(group_lines) if group_lines else "  (conf/ unavailable)"
    return (
        "Hydra overrides (for ow train / bare ow):\n"
        "  Select config groups: model=..., training=..., task=...\n"
        "  Set values: training.total_updates=1000, print_resolved_config=true\n"
        "  Add keys: +key=value    Remove groups/keys: ~group or -group\n\n"
        "Primary config groups (group=option):\n"
        f"{groups_block}\n"
    )


def print_ow_help() -> None:
    """Print top-level ``ow`` help without invoking Hydra."""

    print(
        "Orbit Wars CLI (ow)\n\n"
        "Commands:\n"
        "  train   Local or Kaggle JAX training via Hydra configuration\n"
        "  eval    Tournament eval, artifact worker, Kaggle competition submit\n"
        "  benchmark  Stability benchmarks and pre-flight learning gates\n"
        "  make    Generate W&B sweep YAML (scripts/make_wandb_sweep.py)\n\n"
        "Bare `ow` or `ow KEY=VALUE` defaults to `ow train`.\n\n"
        "Usage:\n"
        "  uv run ow [HYDRA_OVERRIDES...]\n"
        "  uv run ow train [local|kaggle] [SUBCMD] [FLAGS] [HYDRA_OVERRIDES...]\n"
        "  uv run ow eval tournament [OPTIONS]\n"
        "  uv run ow eval package --checkpoint outputs/.../jax_ckpt_last.pkl\n"
        "  uv run ow eval worker --run outputs/campaigns/<campaign>/runs/<run_id>\n"
        "  uv run ow eval submit --checkpoint outputs/.../jax_ckpt_last.pkl\n"
        "  uv run ow make [MAKE_SCRIPT_OVERRIDES...]\n\n"
        "Examples:\n"
        "  uv run ow train print_resolved_config=true\n"
        "  uv run ow train training=smoke training.total_updates=10\n"
        "  uv run ow eval tournament --checkpoint outputs/.../jax_ckpt_000100.pkl --vs-promoted\n"
        '  uv run ow eval submit --checkpoint outputs/.../jax_ckpt_last.pkl -m "update 100"\n'
        "  uv run ow train kaggle status owner/kernel-slug\n"
        "  uv run ow train kaggle training=2p4p_32_split\n"
        "  uv run ow make wandb_sweep=shield_cheap_history\n\n"
        f"{_hydra_override_section()}"
        "More help:\n"
        "  ow train --help\n"
        "  uv run ow eval tournament --help\n"
        "  uv run python -m src.cli.kaggle_runner --help\n"
    )


def print_train_help() -> None:
    """Print ``ow train`` help without invoking Hydra."""

    print(
        "ow train — JAX training (local Hydra or Kaggle remote)\n\n"
        "Usage:\n"
        "  uv run ow train [local] [HYDRA_OVERRIDES...]\n"
        "  uv run ow train kaggle [SUBCMD] [FLAGS] [HYDRA_OVERRIDES...]\n"
        "  uv run ow train --host local|kaggle ...\n\n"
        "Local (default):\n"
        "  Runs the Hydra + JAX training loop on this machine.\n"
        "  Any bare Hydra override implies local host (e.g. ow train training=smoke).\n\n"
        "Kaggle (ow train kaggle ...):\n"
        "  Packages and pushes a Kaggle kernel worker (default accelerator: P100).\n"
        "  Subcommands:\n"
        "    (default) launch         Push a standalone training kernel\n"
        "    preflight                Validate W&B + Kaggle credentials\n"
        "    prepare                  Build the worker bundle only\n"
        "    status KERNEL            Poll kernel status\n"
        "    sync|sync-output KERNEL  Download kernel outputs\n"
        "    shortlist                Export W&B sweep shortlist JSON\n"
        "    latest-checkpoint        Resolve latest checkpoint from a sweep\n\n"
        "Examples:\n"
        "  uv run ow train training=smoke training.total_updates=10\n"
        "  uv run ow train local print_resolved_config=true\n"
        "  uv run ow train kaggle preflight\n"
        "  uv run ow train kaggle status owner/kernel-slug\n"
        "  uv run ow train kaggle --run-type smoke training=2p4p_32_split\n\n"
        f"{_hydra_override_section()}"
        "Kaggle flag reference:\n"
        "  uv run python -m src.cli.kaggle_runner --help\n"
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

    if contains_cli_help(args):
        print_train_help()
        raise SystemExit(0)

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
    elif is_hydra_override(args[0]):
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
        if is_hydra_override(token) and not token.startswith("--"):
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
        if is_hydra_override(token):
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
            argv.extend(["--accelerator", DEFAULT_KAGGLE_ACCELERATOR])

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
