from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from src.orchestration.colab_cli import ColabSessionInfo
from src.orchestration.colab_runner import (
    ColabRequest,
    _merge_shortlist_overrides,
    default_session_slug,
    launch,
    prepare_package,
    preflight,
    shortlist,
    status,
    stop,
    sync,
)


class _FakeColabCli:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, Path, str]] = []
        self.exec_calls: list[tuple[str, int | None]] = []
        self.downloads: list[tuple[str, str, Path]] = []
        self.stopped = False

    def resolve_executable(self) -> str:
        return "/usr/bin/colab"

    def version(self, *, timeout: int = 15) -> str:
        return "0.5.9"

    def auth_check(self, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout="[]", stderr="")

    def new(self, slug: str, *, gpu: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=f"session {slug}", stderr="")

    def upload(
        self,
        session: str,
        local_path: Path,
        remote_path: str,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.uploads.append((session, local_path, remote_path))
        return subprocess.CompletedProcess([], 0, stdout="uploaded", stderr="")

    def exec(
        self,
        session: str,
        command: str,
        *,
        timeout: int | None = None,
        local_file: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.exec_calls.append((session, timeout))
        return subprocess.CompletedProcess([], 0, stdout="worker done", stderr="")

    def download(
        self,
        session: str,
        remote_path: str,
        local_path: Path,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.downloads.append((session, remote_path, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_path.endswith("_sync.tgz"):
            payload = b"campaign payload"
            with tarfile.open(local_path, "w:gz") as archive:
                import io

                info = tarfile.TarInfo(name="runs/run-a/logs/smoke_jax.jsonl")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        else:
            local_path.write_text('{"status":"colab_complete","exit_code":0}\n', encoding="utf-8")
        return subprocess.CompletedProcess([], 0, stdout="downloaded", stderr="")

    def status(self, session: str, *, timeout: int = 30) -> ColabSessionInfo:
        if self.stopped:
            return ColabSessionInfo(slug=session, raw="stopped", returncode=0)
        return ColabSessionInfo(slug=session, raw="running on T4", returncode=0)

    def stop(self, session: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        self.stopped = True
        return subprocess.CompletedProcess([], 0, stdout="stopped", stderr="")


def test_preflight_reports_colab_and_gpu(monkeypatch, tmp_path: Path) -> None:
    request = ColabRequest(work_dir=tmp_path / "kernel", gpu="T4")
    payload = preflight(request, cli=_FakeColabCli())
    names = {check["name"] for check in payload["checks"]}
    assert payload["ok"] is True
    assert {"colab_cli", "colab_auth", "gpu_type", "package_dir_writable"} <= names


def test_prepare_renders_tarball_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "outputs" / "colab_runner" / "kernel"
    request = ColabRequest(
        work_dir=work_dir,
        gpu="T4",
        hydra_overrides=["training.total_updates=10", "output.campaign=colab_test"],
    )
    payload = prepare_package(request)
    assert Path(payload["tarball_path"]).is_file()
    assert Path(payload["summary_path"]).is_file()
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    assert summary["gpu"] == "T4"
    assert "training.total_updates=10" in summary["hydra_overrides"]


def test_prepare_uses_wandb_key_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "env-key")
    work_dir = tmp_path / "kernel"
    request = ColabRequest(
        work_dir=work_dir,
        hydra_overrides=["training.total_updates=10"],
    )
    prepare_package(request)
    env = json.loads((work_dir / "worker-env.json").read_text(encoding="utf-8"))
    summary = json.loads((work_dir / "package-summary.json").read_text(encoding="utf-8"))
    assert env["WANDB_API_KEY"] == "env-key"
    assert summary["generated_env"]["WANDB_API_KEY"] == "<redacted>"


def test_prepare_uses_wandb_key_from_netrc_when_env_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(
        "machine api.wandb.ai login user password netrc-key\n",
        encoding="utf-8",
    )
    netrc_path.chmod(0o600)
    work_dir = tmp_path / "kernel"
    request = ColabRequest(
        work_dir=work_dir,
        hydra_overrides=["training.total_updates=10"],
    )
    prepare_package(request)
    env = json.loads((work_dir / "worker-env.json").read_text(encoding="utf-8"))
    assert env["WANDB_API_KEY"] == "netrc-key"


def test_launch_dry_run_emits_json_without_subprocess(tmp_path: Path, capsys) -> None:
    request = ColabRequest(
        work_dir=tmp_path / "kernel",
        dry_run=True,
        hydra_overrides=["training.total_updates=3", "output.campaign=colab_smoke"],
    )
    payload = launch(request, cli=_FakeColabCli())
    assert payload["dry_run"] is True
    assert payload["session_slug"].startswith("ow-colab_smoke-")
    captured = capsys.readouterr()
    assert "dry_run" in captured.out


def test_launch_records_ledger_and_sessions(tmp_path: Path, capsys) -> None:
    work_dir = tmp_path / "kernel"
    ledger = tmp_path / "launches.jsonl"
    sessions_path = tmp_path / "sessions.json"
    cli = _FakeColabCli()
    request = ColabRequest(
        work_dir=work_dir,
        ledger=ledger,
        sessions_path=sessions_path,
        timeout=120,
        hydra_overrides=["training.total_updates=3", "output.campaign=colab_launch"],
    )
    launch(request, cli=cli)
    assert cli.uploads
    assert cli.exec_calls
    ledger_lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(ledger_lines[-1])["event"] == "launch"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    assert any(key.startswith("ow-colab_launch-") for key in sessions)


def test_shortlist_merges_from_json_rank(tmp_path: Path) -> None:
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "run-1",
                    "hydra_overrides": ["training.lr=0.001", "model=attention"],
                    "config": {},
                },
                {
                    "run_id": "run-2",
                    "hydra_overrides": ["training.lr=0.002"],
                    "config": {},
                },
            ]
        ),
        encoding="utf-8",
    )
    request = ColabRequest(
        from_shortlist=shortlist_path,
        rank=1,
        hydra_overrides=["training.total_updates=100"],
    )
    merged = _merge_shortlist_overrides(request)
    assert "training.lr=0.002" in merged
    assert "training.total_updates=100" in merged


def test_status_and_sync_use_session(tmp_path: Path, capsys) -> None:
    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(
        json.dumps(
            {
                "ow-colab_smoke-abc": {
                    "session": "ow-colab_smoke-abc",
                    "campaign": "colab_smoke",
                    "hydra_overrides": ["output.campaign=colab_smoke"],
                }
            }
        ),
        encoding="utf-8",
    )
    cli = _FakeColabCli()
    ledger = tmp_path / "launches.jsonl"
    request = ColabRequest(
        session="ow-colab_smoke-abc",
        ledger=ledger,
        sessions_path=sessions_path,
        sync_dir=tmp_path / "synced",
    )
    status(request, cli=cli)
    assert sync(request, cli=cli) == 0
    assert cli.downloads[-1][1].endswith("worker-summary.json")
    synced_log = tmp_path / "synced" / "colab_smoke" / "runs" / "run-a" / "logs" / "smoke_jax.jsonl"
    assert synced_log.is_file()


def test_stop_idempotent_when_already_stopped(tmp_path: Path, capsys) -> None:
    cli = _FakeColabCli()
    cli.stopped = True
    request = ColabRequest(session="ow-colab_smoke-abc", ledger=tmp_path / "ledger.jsonl")
    assert stop(request, cli=cli) == 0
    captured = capsys.readouterr()
    assert "idempotent" in captured.out


def test_default_session_slug_uses_campaign() -> None:
    slug = default_session_slug(["output.campaign=colab_long", "training.total_updates=10"])
    assert slug.startswith("ow-colab_long-")


def test_shortlist_writes_json(monkeypatch, tmp_path: Path, capsys) -> None:
    class _Row:
        run_id = "abc"
        name = "run-a"
        state = "finished"
        checkpoint_artifact = None
        checkpoint_artifact_version = None
        checkpoint_artifact_aliases = ()
        metrics = {"episode_reward_mean": 1.0}
        config = {"training.lr": 0.001}

        @property
        def score(self) -> float:
            return 1.0

    monkeypatch.setattr(
        "src.orchestration.colab_runner.shortlist_from_api",
        lambda **kwargs: [_Row()],
    )
    output = tmp_path / "shortlist.json"
    request = ColabRequest(sweep_id="sweep123", output=output, limit=1)
    shortlist(request)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["hydra_overrides"] == ["training.lr=0.001"]
