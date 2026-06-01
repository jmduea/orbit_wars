from __future__ import annotations

import json
from pathlib import Path

from src.cli import eval as eval_cli
from src.cli.run_status import summarize_run_status


def test_summarize_run_status_with_queued_job(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "campaigns" / "c" / "runs" / "r1"
    queue_dir = run_dir / "queue" / "optional_jobs"
    queue_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "campaign": "c"}),
        encoding="utf-8",
    )
    job = {
        "job_file": str(queue_dir / "tournament_u10.json"),
        "kind": "tournament",
        "status": "queued",
        "update": 10,
    }
    (queue_dir / "tournament_u10.json").write_text(
        json.dumps(job), encoding="utf-8"
    )

    summary = summarize_run_status(run_dir)
    assert summary["run_id"] == "r1"
    assert len(summary["jobs"]) == 1
    assert summary["jobs"][0]["status"] == "queued"


def test_eval_status_cli(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "campaign": "c"}),
        encoding="utf-8",
    )
    (run_dir / "queue" / "optional_jobs").mkdir(parents=True)

    assert eval_cli.main(["status", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert '"run_id": "r1"' in out


def test_eval_empty_argv_prints_help(capsys) -> None:
    assert eval_cli.main([]) == 0
    assert "ow eval" in capsys.readouterr().out
