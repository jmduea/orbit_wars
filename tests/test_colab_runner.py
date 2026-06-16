from __future__ import annotations

import json
import subprocess
import tarfile
import time
from pathlib import Path

from src.orchestration.colab_cli import ColabSessionInfo
from src.orchestration.colab_runner import (
    ColabRequest,
    _merge_shortlist_overrides,
    _bootstrap_script,
    default_session_slug,
    launch,
    monitor,
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

    def new(
        self, slug: str, *, gpu: str, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
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
            with tarfile.open(local_path, "w:gz") as archive:
                import io

                log_payload = (
                    b'{"update":1,"overall_win_rate":0.5,'
                    b'"preflight_sweep_score":-1.0}\n'
                )
                info = tarfile.TarInfo(name="runs/run-a/logs/smoke_jax.jsonl")
                info.mtime = time.time()
                info.size = len(log_payload)
                archive.addfile(info, io.BytesIO(log_payload))
                checkpoint_payload = b"checkpoint"
                ckpt = tarfile.TarInfo(
                    name="runs/run-a/checkpoints/jax_ckpt_000010.pkl"
                )
                ckpt.mtime = time.time()
                ckpt.size = len(checkpoint_payload)
                archive.addfile(ckpt, io.BytesIO(checkpoint_payload))
                last = tarfile.TarInfo(name="runs/run-a/checkpoints/jax_ckpt_last.pkl")
                last.mtime = time.time()
                last.size = len(checkpoint_payload)
                archive.addfile(last, io.BytesIO(checkpoint_payload))
        else:
            local_path.write_text(
                '{"status":"colab_complete","exit_code":0}\n', encoding="utf-8"
            )
        return subprocess.CompletedProcess([], 0, stdout="downloaded", stderr="")

    def status(self, session: str, *, timeout: int = 30) -> ColabSessionInfo:
        if self.stopped:
            return ColabSessionInfo(slug=session, raw="stopped", returncode=0)
        return ColabSessionInfo(slug=session, raw="running on T4", returncode=0)

    def stop(
        self, session: str, *, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        self.stopped = True
        return subprocess.CompletedProcess([], 0, stdout="stopped", stderr="")


class _StaleOutputColabCli(_FakeColabCli):
    def download(
        self,
        session: str,
        remote_path: str,
        local_path: Path,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if remote_path.endswith("_sync.tgz"):
            import io

            self.downloads.append((session, remote_path, local_path))
            local_path.parent.mkdir(parents=True, exist_ok=True)
            old_mtime = time.time() - 3600
            with tarfile.open(local_path, "w:gz") as archive:
                log_payload = (
                    b'{"update":1,"overall_win_rate":0.5,'
                    b'"preflight_sweep_score":-1.0}\n'
                )
                info = tarfile.TarInfo(name="runs/run-a/logs/smoke_jax.jsonl")
                info.mtime = old_mtime
                info.size = len(log_payload)
                archive.addfile(info, io.BytesIO(log_payload))
                checkpoint_payload = b"checkpoint"
                ckpt = tarfile.TarInfo(
                    name="runs/run-a/checkpoints/jax_ckpt_000010.pkl"
                )
                ckpt.mtime = old_mtime
                ckpt.size = len(checkpoint_payload)
                archive.addfile(ckpt, io.BytesIO(checkpoint_payload))
                last = tarfile.TarInfo(name="runs/run-a/checkpoints/jax_ckpt_last.pkl")
                last.mtime = old_mtime
                last.size = len(checkpoint_payload)
                archive.addfile(last, io.BytesIO(checkpoint_payload))
            return subprocess.CompletedProcess([], 0, stdout="downloaded", stderr="")
        return super().download(
            session,
            remote_path,
            local_path,
            timeout=timeout,
        )


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
    summary = json.loads(
        (work_dir / "package-summary.json").read_text(encoding="utf-8")
    )
    assert env["WANDB_API_KEY"] == "env-key"
    assert env["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
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


def test_launch_can_start_monitor_after_worker_bootstrap(
    tmp_path: Path, capsys
) -> None:
    work_dir = tmp_path / "kernel"
    ledger = tmp_path / "launches.jsonl"
    sessions_path = tmp_path / "sessions.json"
    cli = _FakeColabCli()
    request = ColabRequest(
        work_dir=work_dir,
        ledger=ledger,
        sessions_path=sessions_path,
        sync_dir=tmp_path / "synced",
        monitor_dir=tmp_path / "monitor",
        timeout=120,
        hydra_overrides=["training.total_updates=3", "output.campaign=colab_launch"],
        monitor_after_launch=True,
        once=True,
        eval_checkpoints=False,
        stale_seconds=9999,
    )

    payload = launch(request, cli=cli)

    assert payload["monitor_hint"].startswith("ow train colab monitor --session")
    assert payload["monitor_payload"]["stale"] is False
    assert payload["monitor_payload"]["progress"]["latest_update"] == 1
    assert len(cli.exec_calls) >= 2


def test_bootstrap_starts_worker_detached_for_keepalive_safety() -> None:
    bootstrap = _bootstrap_script()
    assert "subprocess.Popen" in bootstrap
    assert "start_new_session=True" in bootstrap
    assert "colab-worker.pid" in bootstrap
    assert "subprocess.call" not in bootstrap


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
    assert cli.exec_calls[-1][1] == 600
    assert cli.downloads[-1][1].endswith("worker-summary.json")
    synced_log = (
        tmp_path
        / "synced"
        / "colab_smoke"
        / "runs"
        / "run-a"
        / "logs"
        / "smoke_jax.jsonl"
    )
    assert synced_log.is_file()


def test_monitor_syncs_and_evaluates_new_checkpoints(tmp_path: Path, capsys) -> None:
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
    eval_commands: list[list[str]] = []

    def evaluator(
        command: list[str], output_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        eval_commands.append(command)
        matches = output_dir / "matches"
        matches.mkdir(parents=True, exist_ok=True)
        (matches / "2p_vs_baseline_0000.json").write_text(
            json.dumps(
                {
                    "agent_ids": ["jax_ckpt_000010", "baseline:noop"],
                    "baseline_name": "noop",
                    "format_name": "2p_vs_baseline",
                    "results": {"jax_ckpt_000010": "win", "baseline:noop": "loss"},
                    "placements": {"jax_ckpt_000010": 1, "baseline:noop": 2},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    request = ColabRequest(
        session="ow-colab_smoke-abc",
        ledger=tmp_path / "launches.jsonl",
        sessions_path=sessions_path,
        sync_dir=tmp_path / "synced",
        monitor_dir=tmp_path / "monitor",
        once=True,
        stale_seconds=9999,
    )
    payload = monitor(
        request, cli=_FakeColabCli(), evaluator=evaluator, sleep=lambda _: None
    )
    assert payload["stale"] is False
    assert payload["progress"]["latest_update"] == 1
    assert len(payload["evaluated"]) == 1
    assert payload["evaluated"][0]["summary"]["by_baseline"]["noop"]["win_rate"] == 1.0
    assert eval_commands
    state = json.loads((tmp_path / "monitor" / "ow-colab_smoke-abc.json").read_text())
    assert len(state["evaluated_checkpoints"]) == 1


def test_monitor_skips_previously_evaluated_checkpoints(tmp_path: Path, capsys) -> None:
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
    eval_count = 0

    def evaluator(
        command: list[str], output_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        nonlocal eval_count
        eval_count += 1
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    request = ColabRequest(
        session="ow-colab_smoke-abc",
        ledger=tmp_path / "launches.jsonl",
        sessions_path=sessions_path,
        sync_dir=tmp_path / "synced",
        monitor_dir=tmp_path / "monitor",
        once=True,
        stale_seconds=9999,
    )
    monitor(request, cli=_FakeColabCli(), evaluator=evaluator, sleep=lambda _: None)
    second_payload = monitor(
        request, cli=_FakeColabCli(), evaluator=evaluator, sleep=lambda _: None
    )

    assert eval_count == 1
    assert second_payload["evaluated"] == []


def test_monitor_marks_running_session_stale_when_synced_activity_is_old(
    tmp_path: Path, capsys
) -> None:
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
    request = ColabRequest(
        session="ow-colab_smoke-abc",
        ledger=tmp_path / "launches.jsonl",
        sessions_path=sessions_path,
        sync_dir=tmp_path / "synced",
        monitor_dir=tmp_path / "monitor",
        once=True,
        stale_seconds=30,
        eval_checkpoints=False,
    )

    payload = monitor(
        request, cli=_StaleOutputColabCli(), evaluator=None, sleep=lambda _: None
    )

    assert payload["remote_status"] == "running"
    assert payload["stale"] is True
    assert any(
        reason.startswith("activity_stale_") for reason in payload["stale_reasons"]
    )


def test_stop_idempotent_when_already_stopped(tmp_path: Path, capsys) -> None:
    cli = _FakeColabCli()
    cli.stopped = True
    request = ColabRequest(
        session="ow-colab_smoke-abc", ledger=tmp_path / "ledger.jsonl"
    )
    assert stop(request, cli=cli) == 0
    captured = capsys.readouterr()
    assert "idempotent" in captured.out


def test_default_session_slug_uses_campaign() -> None:
    slug = default_session_slug(
        ["output.campaign=colab_long", "training.total_updates=10"]
    )
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
