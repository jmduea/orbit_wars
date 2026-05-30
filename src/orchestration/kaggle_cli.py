from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


def kaggle_push_supports_secret_flag(*, executable: str = "kaggle") -> bool:
    """Return True when the installed Kaggle CLI exposes ``--secret`` on kernels push."""

    try:
        completed = subprocess.run(
            [executable, "kernels", "push", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    help_text = completed.stdout or completed.stderr or ""
    return "--secret" in help_text


class CommandRunner(Protocol):
    def __call__(
        self, command: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]: ...


def _default_runner(
    command: Sequence[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True, slots=True)
class KaggleKernelRef:
    owner: str
    slug: str

    @classmethod
    def parse(cls, value: str) -> "KaggleKernelRef":
        if "/" not in value:
            raise ValueError("Kaggle kernel ref must use owner/slug format.")
        owner, slug = value.split("/", 1)
        if not owner or not slug:
            raise ValueError("Kaggle kernel ref must include owner and slug.")
        return cls(owner=owner, slug=slug)

    def __str__(self) -> str:
        return f"{self.owner}/{self.slug}"


@dataclass(frozen=True, slots=True)
class KaggleKernelStatus:
    ref: KaggleKernelRef
    raw: str
    returncode: int = 0

    @property
    def normalized(self) -> str:
        text = self.raw.strip().lower()
        if "complete" in text or "succeed" in text:
            return "complete"
        if "fail" in text or "error" in text:
            return "failed"
        if "running" in text or "queued" in text:
            return "running"
        if self.returncode != 0:
            return "failed"
        return "unknown"


class KaggleCli:
    """Small wrapper around Kaggle kernel commands."""

    def __init__(
        self,
        *,
        executable: str = "kaggle",
        runner: CommandRunner = _default_runner,
    ) -> None:
        self._executable = executable
        self._runner = runner

    def push(
        self,
        package_dir: Path,
        *,
        accelerator: str | None = None,
        timeout_seconds: int | None = None,
        secrets: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [self._executable, "kernels", "push", "-p", str(package_dir)]
        if accelerator:
            command.extend(["--accelerator", accelerator])
        if timeout_seconds is not None:
            command.extend(["--timeout", str(int(timeout_seconds))])
        if secrets and kaggle_push_supports_secret_flag(executable=self._executable):
            for secret in secrets:
                command.extend(["--secret", secret])
        return self._runner(command, cwd=package_dir)

    def status(self, ref: KaggleKernelRef) -> KaggleKernelStatus:
        result = self._runner(
            [self._executable, "kernels", "status", str(ref)],
            cwd=None,
        )
        return KaggleKernelStatus(
            ref=ref,
            raw=(result.stdout or result.stderr),
            returncode=result.returncode,
        )

    def files(self, ref: KaggleKernelRef) -> subprocess.CompletedProcess[str]:
        return self._runner([self._executable, "kernels", "files", str(ref)], cwd=None)

    def output(
        self,
        ref: KaggleKernelRef,
        output_dir: Path,
        *,
        force: bool = False,
        file_pattern: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            self._executable,
            "kernels",
            "output",
            str(ref),
            "-p",
            str(output_dir),
        ]
        if force:
            command.append("--force")
        if file_pattern:
            command.extend(["--file-pattern", file_pattern])
        return self._runner(command, cwd=None)
