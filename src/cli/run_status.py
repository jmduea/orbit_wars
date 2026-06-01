"""Shared helpers for run directory / queue introspection."""

from __future__ import annotations

import json
from pathlib import Path

from src.artifacts.pipeline import load_optional_jobs
from src.artifacts.worker_runner import resolve_run_worker_dirs


def summarize_run_status(run_dir: Path) -> dict[str, object]:
    """Build a JSON-serializable status summary for a campaign run directory."""

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, object] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    queue_dir, evaluations_dir = resolve_run_worker_dirs(run_dir)
    jobs = load_optional_jobs(
        queue_dir,
        statuses={"queued", "running", "failed", "completed"},
    )
    job_rows = [
        {
            "job_file": job.get("job_file"),
            "kind": job.get("kind"),
            "status": job.get("status"),
            "update": job.get("update"),
            "error": job.get("error"),
        }
        for job in jobs
    ]

    campaign = None
    if manifest is not None:
        campaign = manifest.get("campaign")
    promoted_manifest: str | None = None
    if campaign:
        promoted_path = (
            run_dir.parent.parent / "promoted" / "current_best" / "manifest.json"
        )
        if promoted_path.is_file():
            promoted_manifest = str(promoted_path)

    last_event: str | None = None
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        log_files = sorted(logs_dir.glob("*_jax.jsonl"))
        if log_files:
            last_line = ""
            for line in log_files[-1].read_text(encoding="utf-8").splitlines():
                if line.strip():
                    last_line = line
            if last_line:
                try:
                    record = json.loads(last_line)
                    last_event = str(record.get("event") or record.get("update"))
                except json.JSONDecodeError:
                    last_event = "unparseable_last_line"

    return {
        "run_dir": str(run_dir),
        "manifest_present": manifest is not None,
        "campaign": campaign,
        "run_id": manifest.get("run_id") if manifest else None,
        "queue_dir": str(queue_dir),
        "evaluations_dir": str(evaluations_dir),
        "jobs": job_rows,
        "promoted_manifest": promoted_manifest,
        "last_log_marker": last_event,
        "worker_stdout_log": str(queue_dir / "worker.stdout.log"),
        "worker_stderr_log": str(queue_dir / "worker.stderr.log"),
    }
