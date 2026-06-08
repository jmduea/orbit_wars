from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

INSTALL_HINT = "Install with: uv tool install google-colab-cli"
AUTH_HINT = "Authenticate with: colab auth"


class CommandRunner(Protocol):
    def __call__(
        self, command: Sequence[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]: ...


def _default_runner(
    command: Sequence[str], *, cwd: Path | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class ColabCliError(RuntimeError):
    """Raised when the Colab CLI is missing or a command fails."""


@dataclass(frozen=True, slots=True)
class ColabSessionInfo:
    slug: str
    raw: str
    returncode: int = 0

    @property
    def normalized(self) -> str:
        text = self.raw.strip().lower()
        if "running" in text or "active" in text:
            return "running"
        if "stopped" in text or "terminated" in text or "complete" in text:
            return "stopped"
        if self.returncode != 0 and not text:
            return "unknown"
        if self.returncode != 0:
            return "failed"
        return "unknown"


def parse_sessions_json(text: str) -> list[dict[str, object]]:
    """Parse ``colab sessions --json`` output when available."""

    text = text.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        return [item for item in payload["sessions"] if isinstance(item, dict)]
    return []


def parse_session_slug_from_text(text: str) -> str | None:
    """Best-effort session slug extraction from Colab CLI stdout."""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ow-"):
            return stripped.split()[0]
    match = re.search(r"\b(ow-[a-z0-9._-]+)\b", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


class ColabCli:
    """Thin wrapper around ``google-colab-cli`` subprocess commands."""

    def __init__(
        self,
        *,
        executable: str = "colab",
        runner: CommandRunner = _default_runner,
    ) -> None:
        self._executable = executable
        self._runner = runner

    def resolve_executable(self) -> str:
        """Return the Colab CLI path or raise with an install hint."""

        resolved = shutil.which(self._executable)
        if resolved is None:
            raise ColabCliError(
                f"colab CLI executable not found on PATH. {INSTALL_HINT}"
            )
        return resolved

    def version(self, *, timeout: int = 15) -> str:
        executable = self.resolve_executable()
        completed = self._runner([executable, "version"], timeout=timeout)
        if completed.returncode != 0:
            raise ColabCliError(
                _format_cli_failure("colab version", completed, auth_hint=False)
            )
        return (completed.stdout or completed.stderr).strip()

    def auth_check(self, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        return self._runner([executable, "sessions"], timeout=timeout)

    def sessions(self, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        for flag in ("--json",):
            completed = self._runner(
                [executable, "sessions", flag],
                timeout=timeout,
            )
            if completed.returncode == 0:
                return completed
        return self._runner([executable, "sessions"], timeout=timeout)

    def new(
        self,
        slug: str,
        *,
        gpu: str,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        return self._runner(
            [executable, "new", "-s", slug, "--gpu", gpu],
            timeout=timeout,
        )

    def upload(
        self,
        session: str,
        local_path: Path,
        remote_path: str,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        command = [
            executable,
            "upload",
            "-s",
            session,
            str(local_path),
            remote_path,
        ]
        return self._runner(command, timeout=timeout)

    def download(
        self,
        session: str,
        remote_path: str,
        local_path: Path,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            "download",
            "-s",
            session,
            remote_path,
            str(local_path),
        ]
        return self._runner(command, timeout=timeout)

    def exec(
        self,
        session: str,
        command: str,
        *,
        timeout: int | None = None,
        local_file: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        argv = [executable, "exec", "-s", session]
        if timeout is not None:
            argv.extend(["--timeout", str(int(timeout))])
        if local_file is not None:
            argv.extend(["-f", str(local_file)])
        else:
            argv.append(command)
        return self._runner(argv, timeout=timeout)

    def stop(self, session: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        executable = self.resolve_executable()
        return self._runner([executable, "stop", "-s", session], timeout=timeout)

    def status(self, session: str, *, timeout: int = 30) -> ColabSessionInfo:
        executable = self.resolve_executable()
        completed = self._runner([executable, "status", "-s", session], timeout=timeout)
        return ColabSessionInfo(
            slug=session,
            raw=(completed.stdout or completed.stderr or ""),
            returncode=completed.returncode,
        )


def _format_cli_failure(
    label: str,
    completed: subprocess.CompletedProcess[str],
    *,
    auth_hint: bool,
) -> str:
    detail = (completed.stderr or completed.stdout or "").strip()
    suffix = f" {AUTH_HINT}" if auth_hint and _looks_like_auth_failure(detail) else ""
    return f"{label} failed (exit {completed.returncode}): {detail}{suffix}".strip()


def _looks_like_auth_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("auth", "oauth", "credential", "permission", "403", "login")
    )
