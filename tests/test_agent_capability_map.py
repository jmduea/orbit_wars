"""Regression: capability map rows are registered in ``ow`` CLI --help trees."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_DOC = REPO_ROOT / "docs" / "AGENT_CAPABILITIES.md"
CAPABILITY_MAP_HEADER = "## Capability map"

_TOP_LEVEL_OW_COMMANDS: frozenset[str] = frozenset(
    {"train", "eval", "runs", "promote", "benchmark", "sweep", "make"}
)

# Token-style subcommands not expressed as argparse ``{choices}`` in --help.
_EXTRA_NESTED_TOKENS: dict[tuple[str, ...], frozenset[str]] = {
    ("ow", "benchmark", "gate"): frozenset({"list", "run"}),
    (
        "ow",
        "benchmark",
        "gate",
        "run",
    ): frozenset(
        {
            "admission",
            "beat_noop",
            "beat_random",
            "curriculum_staged",
            "win_proof_tournament",
        }
    ),
    ("ow", "train"): frozenset({"kaggle", "local"}),
}


def _read_capability_map_section() -> str:
    text = CAPABILITIES_DOC.read_text(encoding="utf-8")
    start = text.find(CAPABILITY_MAP_HEADER)
    assert start >= 0, f"missing {CAPABILITY_MAP_HEADER!r} in AGENT_CAPABILITIES.md"
    rest = text[start + len(CAPABILITY_MAP_HEADER) :]
    end = rest.find("\n## ")
    return rest[:end] if end >= 0 else rest


def _normalize_command_cell(cell: str) -> str:
    match = re.search(r"`(ow [^`]+)`", cell)
    if match:
        return match.group(1)
    return cell.strip("`").split(" — ", 1)[0].strip("`").strip()


def _parse_ow_commands_from_table(section: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for line in section.splitlines():
        if (
            not line.startswith("|")
            or line.startswith("| Action")
            or line.startswith("|-")
        ):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if "(planned)" in cells[0].lower():
            continue
        raw = _normalize_command_cell(cells[1])
        if not raw.startswith("ow "):
            continue
        parts = tuple(raw.split())
        assert parts[0] == "ow", raw
        commands.append(parts)
    assert commands, "expected at least one ow command in capability map table"
    return commands


def _ow_help(argv: list[str]) -> str:
    proc = subprocess.run(
        ["uv", "run", "ow", *argv, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout + proc.stderr


def _choices_from_help(help_text: str) -> set[str]:
    choices: set[str] = set()
    for match in re.finditer(r"\{([^}]+)\}", help_text):
        for part in match.group(1).split(","):
            token = part.strip()
            if token:
                choices.add(token)
    return choices


def _nested_tokens(prefix: tuple[str, ...], help_text: str) -> set[str]:
    choices = _choices_from_help(help_text)
    if choices:
        return choices
    return set(_EXTRA_NESTED_TOKENS.get(prefix, ()))


def _collect_help_registry() -> dict[tuple[str, ...], str]:
    """Map command path prefixes to concatenated --help text for that node."""

    registry: dict[tuple[str, ...], str] = {}
    registry[("ow",)] = _ow_help([])
    for command in sorted(_TOP_LEVEL_OW_COMMANDS):
        if command == "train":
            registry[("ow", "train")] = _ow_help(["train"])
            continue
        body = _ow_help([command])
        registry[("ow", command)] = body
        for sub in _choices_from_help(body):
            sub_body = _ow_help([command, sub])
            registry[("ow", command, sub)] = sub_body
            for nested in _choices_from_help(sub_body):
                registry[("ow", command, sub, nested)] = _ow_help(
                    [command, sub, nested]
                )
    return registry


def _command_registered(
    path: tuple[str, ...], registry: dict[tuple[str, ...], str]
) -> bool:
    assert path[0] == "ow"
    if len(path) == 1:
        return True
    if path[1] not in _TOP_LEVEL_OW_COMMANDS:
        return False
    prefix: tuple[str, ...] = ("ow", path[1])
    if prefix not in registry:
        return False
    for index in range(2, len(path)):
        token = path[index]
        if prefix == ("ow", "train") and "=" in token:
            return index == len(path) - 1
        nested = _nested_tokens(prefix, registry.get(prefix, ""))
        if not nested:
            return False
        if token not in nested:
            return False
        prefix = (*prefix, token)
        if prefix not in registry and index < len(path) - 1:
            if not _EXTRA_NESTED_TOKENS.get(prefix):
                return False
    return True


@pytest.fixture(scope="module")
def help_registry() -> dict[tuple[str, ...], str]:
    return _collect_help_registry()


def test_shape_calibrate_not_registered_in_benchmark_help(
    help_registry: dict[tuple[str, ...], str],
) -> None:
    path = ("ow", "benchmark", "shape-calibrate")
    assert not _command_registered(path, help_registry)


def test_capability_map_ow_commands_in_help(
    help_registry: dict[tuple[str, ...], str],
) -> None:
    section = _read_capability_map_section()
    missing: list[str] = []
    for path in _parse_ow_commands_from_table(section):
        if not _command_registered(path, help_registry):
            missing.append(" ".join(path))
    assert not missing, "capability map commands missing from ow --help:\n" + "\n".join(
        missing
    )


def test_benchmark_help_commands_in_capability_map(
    help_registry: dict[tuple[str, ...], str],
) -> None:
    """Every ``ow benchmark`` subcommand in --help has a capability-map row."""
    benchmark_help = help_registry.get(("ow", "benchmark"), "")
    subcommands = _choices_from_help(benchmark_help)
    assert subcommands, "expected benchmark subcommands in --help"

    section = _read_capability_map_section()
    map_paths = _parse_ow_commands_from_table(section)
    mapped_benchmark = {
        path[2] for path in map_paths if len(path) >= 3 and path[1] == "benchmark"
    }
    missing = sorted(subcommands - mapped_benchmark)
    assert not missing, (
        "benchmark --help subcommands missing from capability map:\n"
        + "\n".join(f"  ow benchmark {name}" for name in missing)
    )
