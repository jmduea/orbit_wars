"""Run queued optional artifact jobs (replay, docker, tournament, checkpoint_eval)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from src.artifacts.pipeline import load_optional_jobs

ProcessJobFn = Callable[[dict[str, object]], None]


def resolve_run_worker_dirs(run_dir: Path) -> tuple[Path, Path]:
    """Return (queue_dir, evaluations_dir) for a standard campaign run directory."""

    run_dir = run_dir.resolve()
    queue_dir = run_dir / "queue" / "optional_jobs"
    evaluations_dir = run_dir / "evaluations"
    return queue_dir, evaluations_dir


def run_optional_job_worker(
    queue_dir: Path,
    process_job: ProcessJobFn,
    write_status: Callable[..., None],
    *,
    result_root: Path | None = None,
    once: bool = False,
    poll_seconds: float = 5.0,
    idle_exit_seconds: float | None = None,
    recover_running: bool = False,
    retry_failed: bool = False,
    verbose: bool = False,
) -> int:
    """Poll ``queue_dir`` and process optional jobs until idle or ``once`` completes."""

    last_work_time = time.monotonic()
    while True:
        statuses = {"queued"}
        if recover_running:
            statuses.add("running")
        if retry_failed:
            statuses.add("failed")
        jobs = load_optional_jobs(queue_dir, statuses=statuses)
        if jobs:
            last_work_time = time.monotonic()
        had_failure = False
        for job in jobs:
            job_file = Path(str(job["job_file"]))
            if result_root is not None:
                job["_trusted_result_root"] = str(result_root)
            kind = job.get("kind", "unknown")
            if verbose:
                print(f"artifact_job_start kind={kind} file={job_file.name}")
            try:
                write_status(job_file, "running")
                process_job(job)
                if verbose:
                    print(f"artifact_job_done kind={kind} file={job_file.name} status=completed")
            except Exception as exc:
                write_status(job_file, "failed", error=str(exc))
                had_failure = True
                if verbose:
                    print(
                        f"artifact_job_failed kind={kind} file={job_file.name} "
                        f"error={exc}"
                    )
        if once:
            return 1 if had_failure else 0
        if idle_exit_seconds is not None and time.monotonic() - last_work_time >= max(
            float(idle_exit_seconds), 0.0
        ):
            return 0
        time.sleep(max(float(poll_seconds), 0.1))
