from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.cli import sweep


def test_wandb_create_passes_project_and_entity(monkeypatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text("method: random\n", encoding="utf-8")
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sweep.subprocess, "run", _run)
    args = argparse.Namespace(
        backend="wandb",
        yaml=yaml_path,
        make=None,
        project="orbit_wars",
        entity="jmduea-jdueadev",
        dry_run=False,
    )

    assert sweep.run_create_cli(args) == 0
    assert calls == [
        [
            "uv",
            "run",
            "wandb",
            "sweep",
            "--project",
            "orbit_wars",
            "--entity",
            "jmduea-jdueadev",
            str(yaml_path),
        ]
    ]


def test_wandb_list_uses_project_name_and_entity(monkeypatch, capsys) -> None:
    class _Sweep:
        id = "abc123"
        name = "preflight"
        state = "RUNNING"

    class _Project:
        def sweeps(self):
            return [_Sweep()]

    class _Api:
        def __init__(self):
            self.calls: list[tuple[str, str | None]] = []

        def project(self, project: str, *, entity: str | None = None):
            self.calls.append((project, entity))
            return _Project()

    api = _Api()

    class _Wandb:
        Api = lambda self=None: api

    monkeypatch.setitem(__import__("sys").modules, "wandb", _Wandb())
    args = argparse.Namespace(
        backend="wandb",
        project="orbit_wars",
        entity="jmduea-jdueadev",
        limit=10,
    )

    assert sweep.run_list_cli(args) == 0
    assert api.calls == [("orbit_wars", "jmduea-jdueadev")]
    assert "abc123" in capsys.readouterr().out
