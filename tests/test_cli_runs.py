from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import runs as runs_cli


def test_runs_list_and_show(tmp_path: Path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run-001",
        "campaign": "cap",
        "created_at": "2026-06-01T00:00:00Z",
        "job_type": "train",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    (log_dir / "cap_jax.jsonl").write_text(
        '{"update": 1}\n{"update": 2}\n', encoding="utf-8"
    )

    outputs_root = tmp_path / "outputs"

    assert runs_cli.main(
        ["list", "--limit", "5", "--outputs-root", str(outputs_root)]
    ) == 0
    listed = capsys.readouterr().out
    assert "run-001" in listed

    assert runs_cli.main(["show", "--run", str(run_dir)]) == 0
    shown = capsys.readouterr().out
    assert '"run_id": "run-001"' in shown

    assert runs_cli.main(["logs", "--run", str(run_dir), "--tail", "1"]) == 0
    logs = capsys.readouterr().out
    assert '"update": 2' in logs


def test_runs_watch_exits_when_idle(tmp_path: Path, capsys, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-001", "campaign": "cap"}),
        encoding="utf-8",
    )
    (run_dir / "queue" / "optional_jobs").mkdir(parents=True)

    sleeps: list[float] = []
    monkeypatch.setattr(runs_cli.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert (
        runs_cli.main(
            [
                "watch",
                "--run",
                str(run_dir),
                "--idle-exit-seconds",
                "0",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.count('"run_id": "run-001"') >= 1
    assert sleeps == []


def test_runs_archive_dry_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-001", "campaign": "cap"}),
        encoding="utf-8",
    )
    (run_dir / "queue" / "optional_jobs").mkdir(parents=True)
    outputs_root = tmp_path / "outputs"
    assert (
        runs_cli.main(
            [
                "archive",
                "--run",
                str(run_dir),
                "--outputs-root",
                str(outputs_root),
                "--dry-run",
            ]
        )
        == 0
    )
    assert run_dir.is_dir()


def test_runs_archive_moves_tree(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-001", "campaign": "cap"}),
        encoding="utf-8",
    )
    (run_dir / "queue" / "optional_jobs").mkdir(parents=True)
    outputs_root = tmp_path / "outputs"
    assert (
        runs_cli.main(
            [
                "archive",
                "--run",
                str(run_dir),
                "--outputs-root",
                str(outputs_root),
                "--confirm",
            ]
        )
        == 0
    )
    archived = (
        outputs_root / "archived" / "campaigns" / "cap" / "runs" / "run-001"
    )
    assert archived.is_dir()
    assert not run_dir.exists()


def test_runs_checkpoint_delete(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-001"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    ckpt = checkpoints / "jax_ckpt_000010.pkl"
    ckpt.write_bytes(b"stub")
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-001", "campaign": "cap"}),
        encoding="utf-8",
    )
    assert (
        runs_cli.main(
            [
                "checkpoint",
                "delete",
                "--run",
                str(run_dir),
                "--checkpoint",
                "jax_ckpt_000010.pkl",
                "--confirm",
            ]
        )
        == 0
    )
    assert not ckpt.exists()


def test_runs_show_missing_manifest_exits(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-manifest"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="No manifest.json"):
        runs_cli.main(["show", "--run", str(run_dir)])


def test_runs_logs_missing_log_exits(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-001", "campaign": "cap"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="No \\*_jax.jsonl log"):
        runs_cli.main(["logs", "--run", str(run_dir)])


def test_runs_list_json_format(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "cap" / "runs" / "run-json"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-json",
                "campaign": "cap",
                "created_at": "2026-06-06T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    outputs_root = tmp_path / "outputs"
    assert (
        runs_cli.main(
            ["list", "--format", "json", "--outputs-root", str(outputs_root)]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["run_id"] == "run-json"
