"""Regression: capability map rows are registered in ``ow`` CLI --help trees."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_DOC = REPO_ROOT / "docs" / "AGENT_CAPABILITIES.md"
CAPABILITY_MAP_HEADER = "## Capability map"


def _read_capability_map_section() -> str:
    text = CAPABILITIES_DOC.read_text(encoding="utf-8")
    start = text.find(CAPABILITY_MAP_HEADER)
    assert start >= 0, f"missing {CAPABILITY_MAP_HEADER!r} in AGENT_CAPABILITIES.md"
    rest = text[start + len(CAPABILITY_MAP_HEADER) :]
    end = rest.find("\n## ")
    return rest[:end] if end >= 0 else rest


def _parse_ow_commands_from_table(section: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("| Action") or line.startswith("|-"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        raw = cells[1].strip("`").strip()
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


def _collect_help_registry() -> dict[tuple[str, ...], str]:
    """Map command path prefixes to concatenated --help text for that node."""

    registry: dict[tuple[str, ...], str] = {}
    top = _ow_help([])
    registry[("ow",)] = top
    for command in _choices_from_help(top):
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
    for depth in range(len(path), 0, -1):
        prefix = path[:depth]
        if prefix not in registry:
            continue
        help_text = registry[prefix]
        if depth == len(path):
            return True
        next_token = path[depth]
        if next_token in _choices_from_help(help_text):
            return True
        if next_token in help_text:
            return True
    return False


@pytest.fixture(scope="module")
def help_registry() -> dict[tuple[str, ...], str]:
    return _collect_help_registry()


def test_capability_map_ow_commands_in_help(help_registry: dict[tuple[str, ...], str]) -> None:
    section = _read_capability_map_section()
    missing: list[str] = []
    for path in _parse_ow_commands_from_table(section):
        if not _command_registered(path, help_registry):
            missing.append(" ".join(path))
    assert not missing, "capability map commands missing from ow --help:\n" + "\n".join(
        missing
    )
