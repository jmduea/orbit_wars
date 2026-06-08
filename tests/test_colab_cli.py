from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.orchestration.colab_cli import (
    ColabCli,
    ColabCliError,
    parse_session_slug_from_text,
    parse_sessions_json,
)


class _FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], subprocess.CompletedProcess[str]]):
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(
        self,
        command,
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        self.calls.append(argv)
        key = tuple(argv)
        if key in self.responses:
            return self.responses[key]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_missing_colab_binary_raises_install_hint(monkeypatch) -> None:
    monkeypatch.setattr("src.orchestration.colab_cli.shutil.which", lambda _: None)
    cli = ColabCli()
    with pytest.raises(ColabCliError, match="uv tool install"):
        cli.resolve_executable()


def test_new_exec_upload_stop_argv_assembly(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.orchestration.colab_cli.shutil.which", lambda _: "/usr/bin/colab")
    runner = _FakeRunner(
        {
            ("/usr/bin/colab", "new", "-s", "ow-smoke-abc", "--gpu", "T4"): subprocess.CompletedProcess(
                [],
                0,
                stdout="session ow-smoke-abc ready",
                stderr="",
            ),
            (
                "/usr/bin/colab",
                "upload",
                "-s",
                "ow-smoke-abc",
                str(tmp_path / "orbit_wars.tgz"),
                "/content/orbit_wars.tgz",
            ): subprocess.CompletedProcess([], 0, stdout="ok", stderr=""),
            (
                "/usr/bin/colab",
                "exec",
                "-s",
                "ow-smoke-abc",
                "--timeout",
                "120",
                "-f",
                str(tmp_path / "bootstrap.sh"),
            ): subprocess.CompletedProcess([], 0, stdout="done", stderr=""),
            ("/usr/bin/colab", "stop", "-s", "ow-smoke-abc"): subprocess.CompletedProcess(
                [], 0, stdout="stopped", stderr=""
            ),
        }
    )
    cli = ColabCli(runner=runner)
    tarball = tmp_path / "orbit_wars.tgz"
    tarball.write_bytes(b"payload")
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text("#!/bin/bash\n", encoding="utf-8")

    assert cli.new("ow-smoke-abc", gpu="T4").returncode == 0
    assert cli.upload("ow-smoke-abc", tarball, "/content/orbit_wars.tgz").returncode == 0
    assert (
        cli.exec("ow-smoke-abc", command="", timeout=120, local_file=bootstrap).returncode
        == 0
    )
    assert cli.stop("ow-smoke-abc").returncode == 0
    assert runner.calls[0][:4] == ["/usr/bin/colab", "new", "-s", "ow-smoke-abc"]
    assert runner.calls[1][1:6] == [
        "upload",
        "-s",
        "ow-smoke-abc",
        str(tarball),
        "/content/orbit_wars.tgz",
    ]
    assert runner.calls[2][1:4] == ["exec", "-s", "ow-smoke-abc"]
    assert runner.calls[2][4:6] == ["--timeout", "120"]


def test_timeout_propagates_to_exec(monkeypatch) -> None:
    monkeypatch.setattr("src.orchestration.colab_cli.shutil.which", lambda _: "/usr/bin/colab")
    runner = _FakeRunner({})
    cli = ColabCli(runner=runner)
    cli.exec("ow-smoke", "echo hi", timeout=7200)
    assert runner.calls[-1][4:6] == ["--timeout", "7200"]


def test_parse_sessions_json_list_and_wrapped() -> None:
    assert parse_sessions_json('[{"slug": "ow-a"}]') == [{"slug": "ow-a"}]
    assert parse_sessions_json('{"sessions": [{"slug": "ow-b"}]}') == [{"slug": "ow-b"}]
    assert parse_sessions_json("not-json") == []


def test_parse_session_slug_from_text() -> None:
    assert parse_session_slug_from_text("Created session ow-colab_long-abc1234") == (
        "ow-colab_long-abc1234"
    )
