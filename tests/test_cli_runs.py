from __future__ import annotations

import json
from pathlib import Path

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
