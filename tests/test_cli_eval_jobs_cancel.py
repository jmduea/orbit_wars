from __future__ import annotations

import json
from pathlib import Path

from src.artifacts.pipeline import cancel_optional_jobs, load_optional_jobs
from src.cli import eval as eval_cli


def _write_job(queue_dir: Path, *, job_id: str, status: str = "queued") -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / f"tournament_u10_{job_id}.json"
    payload = {
        "job_id": job_id,
        "kind": "tournament",
        "status": status,
        "update": 10,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cancel_optional_jobs_dry_run(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    _write_job(queue_dir, job_id="abc123")

    result = cancel_optional_jobs(
        queue_dir,
        all_queued=True,
        dry_run=True,
    )
    assert result["cancelled_count"] == 1
    assert result["cancelled"][0]["would_cancel"] is True
    jobs = load_optional_jobs(queue_dir, statuses={"queued"})
    assert len(jobs) == 1


def test_cancel_optional_jobs_marks_cancelled(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    _write_job(queue_dir, job_id="abc123")

    result = cancel_optional_jobs(queue_dir, job_ids={"abc123"})
    assert result["cancelled_count"] == 1
    jobs = load_optional_jobs(queue_dir, statuses={"cancelled"})
    assert len(jobs) == 1
    assert jobs[0]["status"] == "cancelled"


def test_eval_jobs_cancel_cli(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs" / "r1"
    queue_dir = run_dir / "queue" / "optional_jobs"
    _write_job(queue_dir, job_id="job1")

    assert (
        eval_cli.main(
            [
                "jobs",
                "cancel",
                "--run",
                str(run_dir),
                "--all-queued",
                "--dry-run",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert '"dry_run": true' in out
    assert load_optional_jobs(queue_dir, statuses={"queued"})


def test_eval_jobs_cancel_requires_selector(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "queue" / "optional_jobs").mkdir(parents=True)
    try:
        eval_cli.main(["jobs", "cancel", "--run", str(run_dir)])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit")
