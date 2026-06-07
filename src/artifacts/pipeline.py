from __future__ import annotations

import json
import os
import pickle
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.artifacts.run_paths import append_jsonl_atomic

CheckpointStatus = Literal["committed", "failed", "skipped"]
OptionalJobKind = Literal[
    "replay",
    "docker_validation",
    "tournament",
    "checkpoint_eval",
    "qualifier_eval",
    "bracket_match",
]


@dataclass(slots=True)
class CheckpointJob:
    update: int
    run_dir: Path
    build_payload: Callable[[], Mapping[str, object]]
    final: bool = False
    created_at_unix: float = field(default_factory=time.time)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def numbered_path(self) -> Path:
        return self.run_dir / f"jax_ckpt_{self.update:06d}.pkl"

    @property
    def latest_path(self) -> Path:
        return self.run_dir / "jax_ckpt_last.pkl"


@dataclass(slots=True)
class CheckpointResult:
    job_id: str
    update: int
    status: CheckpointStatus
    numbered_path: Path | None = None
    latest_path: Path | None = None
    final: bool = False
    reason: str | None = None
    error: str | None = None
    started_at_unix: float | None = None
    finished_at_unix: float = field(default_factory=time.time)

    @property
    def committed(self) -> bool:
        return self.status == "committed"


@dataclass(slots=True)
class ArtifactPipelineStats:
    latest_requested_update: int = 0
    latest_committed_update: int = 0
    skipped_checkpoints: int = 0
    failed_checkpoints: int = 0
    queue_depth: int = 0
    unhealthy: bool = False
    last_error: str | None = None

    @property
    def latest_lag_updates(self) -> int:
        return max(0, self.latest_requested_update - self.latest_committed_update)


class ArtifactPipelineError(RuntimeError):
    pass


class AsyncArtifactPipeline:
    """Bounded in-process checkpoint worker with explicit result draining."""

    def __init__(
        self,
        *,
        checkpoint_queue_size: int,
        ledger_path: Path | None = None,
        coalesce_intermediate_checkpoints: bool = True,
        autostart: bool = True,
    ) -> None:
        if checkpoint_queue_size <= 0:
            raise ValueError("checkpoint_queue_size must be positive")
        self._max_pending = int(checkpoint_queue_size)
        self._coalesce_intermediate = bool(coalesce_intermediate_checkpoints)
        self._pending: deque[CheckpointJob] = deque()
        self._active = 0
        self._active_paths: set[Path] = set()
        self._closed = False
        self._unhealthy = False
        self._condition = threading.Condition()
        self._results: queue.SimpleQueue[CheckpointResult] = queue.SimpleQueue()
        self._stats = ArtifactPipelineStats()
        self._ledger_path = ledger_path
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="orbit-wars-artifact-checkpoint-worker",
            daemon=True,
        )
        if autostart:
            self.start()

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    @property
    def stats(self) -> ArtifactPipelineStats:
        with self._condition:
            return ArtifactPipelineStats(
                latest_requested_update=self._stats.latest_requested_update,
                latest_committed_update=self._stats.latest_committed_update,
                skipped_checkpoints=self._stats.skipped_checkpoints,
                failed_checkpoints=self._stats.failed_checkpoints,
                queue_depth=len(self._pending),
                unhealthy=self._unhealthy,
                last_error=self._stats.last_error,
            )

    def submit_checkpoint(self, job: CheckpointJob) -> None:
        """Submit a checkpoint job, coalescing intermediate pending work if needed.

        Results are delivered via ``drain_results()`` after the worker commits.
        """

        with self._condition:
            if self._closed:
                raise ArtifactPipelineError("artifact pipeline is closed")
            if self._unhealthy and not job.final:
                raise ArtifactPipelineError("artifact pipeline is unhealthy")
            self._stats.latest_requested_update = max(
                self._stats.latest_requested_update, int(job.update)
            )

            if job.final:
                self._skip_all_pending_locked("final_checkpoint_priority")
            elif (
                len(self._pending) >= self._max_pending and self._coalesce_intermediate
            ):
                self._skip_one_intermediate_locked("queue_pressure")

            if len(self._pending) >= self._max_pending and not job.final:
                result = _skipped_result(job, "queue_pressure")
                self._record_result_locked(result)
                return

            self._pending.append(job)
            self._stats.queue_depth = len(self._pending)
            self._append_ledger(
                {
                    "event": "checkpoint_queued",
                    "job_id": job.job_id,
                    "update": job.update,
                    "final": job.final,
                    "created_at_unix": job.created_at_unix,
                }
            )
            self._condition.notify_all()

    def drain_results(self) -> list[CheckpointResult]:
        results: list[CheckpointResult] = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            results.append(result)
            with self._condition:
                if result.status == "committed":
                    self._stats.latest_committed_update = max(
                        self._stats.latest_committed_update, result.update
                    )
                elif result.status == "skipped":
                    self._stats.skipped_checkpoints += 1
                elif result.status == "failed":
                    self._stats.failed_checkpoints += 1
                    self._stats.last_error = result.error
                    self._unhealthy = True
        return results

    def flush(self, *, timeout_seconds: float | None = None) -> list[CheckpointResult]:
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        with self._condition:
            while self._pending or self._active:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError("timed out waiting for artifact pipeline flush")
                self._condition.wait(timeout=remaining)
        return self.drain_results()

    def close(self, *, timeout_seconds: float | None = None) -> list[CheckpointResult]:
        results = self.flush(timeout_seconds=timeout_seconds)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout_seconds)
        results.extend(self.drain_results())
        return results

    def protected_paths(self) -> set[Path]:
        with self._condition:
            paths: set[Path] = set()
            for job in self._pending:
                paths.add(job.numbered_path)
                paths.add(job.latest_path)
            paths.update(self._active_paths)
            return paths

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                job = self._pending.popleft()
                self._active += 1
                self._active_paths.add(job.numbered_path)
                self._active_paths.add(job.latest_path)
                self._stats.queue_depth = len(self._pending)
            started = time.time()
            try:
                payload = job.build_payload()
                commit_checkpoint_payload(job.run_dir, job.update, payload)
                result = CheckpointResult(
                    job_id=job.job_id,
                    update=job.update,
                    status="committed",
                    numbered_path=job.numbered_path,
                    latest_path=job.latest_path,
                    final=job.final,
                    started_at_unix=started,
                )
            except Exception as exc:  # pragma: no cover - tested via behavior
                result = CheckpointResult(
                    job_id=job.job_id,
                    update=job.update,
                    status="failed",
                    final=job.final,
                    error=str(exc),
                    started_at_unix=started,
                )
            with self._condition:
                self._active -= 1
                self._active_paths.discard(job.numbered_path)
                self._active_paths.discard(job.latest_path)
                self._record_result_locked(result)
                self._condition.notify_all()

    def _skip_all_pending_locked(self, reason: str) -> list[CheckpointResult]:
        skipped: list[CheckpointResult] = []
        while self._pending:
            result = _skipped_result(self._pending.popleft(), reason)
            self._record_result_locked(result)
            skipped.append(result)
        self._stats.queue_depth = 0
        return skipped

    def _skip_one_intermediate_locked(self, reason: str) -> list[CheckpointResult]:
        skipped: list[CheckpointResult] = []
        for _ in range(len(self._pending)):
            candidate = self._pending.popleft()
            if candidate.final:
                self._pending.append(candidate)
                continue
            result = _skipped_result(candidate, reason)
            self._record_result_locked(result)
            skipped.append(result)
            break
        self._stats.queue_depth = len(self._pending)
        return skipped

    def _record_result_locked(self, result: CheckpointResult) -> None:
        self._results.put(result)
        self._append_ledger(_result_event(result))

    def _append_ledger(self, record: Mapping[str, object]) -> None:
        if self._ledger_path is None:
            return
        append_jsonl_atomic(
            self._ledger_path, {**dict(record), "time_unix": time.time()}
        )


def _skipped_result(job: CheckpointJob, reason: str) -> CheckpointResult:
    return CheckpointResult(
        job_id=job.job_id,
        update=job.update,
        status="skipped",
        final=job.final,
        reason=reason,
    )


def _result_event(result: CheckpointResult) -> dict[str, object]:
    return {
        "event": "checkpoint_result",
        "job_id": result.job_id,
        "update": result.update,
        "status": result.status,
        "final": result.final,
        "numbered_path": str(result.numbered_path) if result.numbered_path else None,
        "latest_path": str(result.latest_path) if result.latest_path else None,
        "reason": result.reason,
        "error": result.error,
        "started_at_unix": result.started_at_unix,
        "finished_at_unix": result.finished_at_unix,
    }


def commit_checkpoint_payload(
    run_dir: Path,
    update: int,
    payload: Mapping[str, object],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    update_path = run_dir / f"jax_ckpt_{update:06d}.pkl"
    latest_path = run_dir / "jax_ckpt_last.pkl"
    _atomic_pickle_dump(payload, update_path)
    _atomic_pickle_dump(payload, latest_path)
    return update_path


def _atomic_pickle_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as file:
            pickle.dump(dict(payload), file)
            file.flush()
            os.fsync(file.fileno())
        tmp_path.replace(path)
        _fsync_dir(path.parent)
    finally:
        tmp_path.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_optional_job(
    queue_dir: Path,
    *,
    kind: OptionalJobKind,
    update: int,
    checkpoint_path: Path,
    payload: Mapping[str, object],
    result_root: Path | None = None,
) -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    result_dir = None
    if result_root is not None:
        result_dir = result_root / f"{kind}_u{update:06d}_{job_id}"
    path = queue_dir / f"{kind}_u{update:06d}_{job_id}.json"
    record = {
        "job_id": job_id,
        "kind": kind,
        "update": update,
        "checkpoint_path": str(checkpoint_path),
        "status": "queued",
        "created_at_unix": time.time(),
        **dict(payload),
    }
    if result_dir is not None:
        record["result_root"] = str(result_root)
        record["result_dir"] = str(result_dir)
        record["result_manifest_path"] = str(result_dir / "manifest.json")
    tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    _fsync_dir(path.parent)
    return path


def load_optional_jobs(
    queue_dir: Path, *, statuses: set[str] | None = None
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    if not queue_dir.exists():
        return jobs
    for path in sorted(queue_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and (
            statuses is None or str(raw.get("status")) in statuses
        ):
            raw["job_file"] = str(path)
            jobs.append(raw)
    return jobs


def load_pending_optional_jobs(queue_dir: Path) -> list[dict[str, object]]:
    return load_optional_jobs(queue_dir, statuses={"queued"})


def load_active_optional_jobs(queue_dir: Path) -> list[dict[str, object]]:
    return load_optional_jobs(queue_dir, statuses={"queued", "running"})


def protected_paths_from_jobs(jobs: Iterable[Mapping[str, object]]) -> set[Path]:
    protected: set[Path] = set()
    for job in jobs:
        checkpoint_path = job.get("checkpoint_path")
        if isinstance(checkpoint_path, str) and checkpoint_path:
            protected.add(Path(checkpoint_path))
    return protected


def _write_optional_job_status(
    job_file: Path,
    status: str,
    *,
    error: str | None = None,
    cancelled_reason: str | None = None,
) -> dict[str, object]:
    payload = json.loads(job_file.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["updated_at_unix"] = time.time()
    if error is not None:
        payload["error"] = error
    elif status not in {"failed"}:
        payload.pop("error", None)
    if cancelled_reason is not None:
        payload["cancelled_reason"] = cancelled_reason
    tmp_path = job_file.with_suffix(job_file.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(job_file)
    payload["job_file"] = str(job_file)
    return payload


def cancel_optional_jobs(
    queue_dir: Path,
    *,
    job_ids: set[str] | None = None,
    all_queued: bool = False,
    include_running: bool = False,
    dry_run: bool = False,
    reason: str = "operator_cancel",
) -> dict[str, object]:
    """Cancel queued optional jobs by marking their JSON status as ``cancelled``."""

    if not all_queued and not job_ids:
        raise ValueError(
            "Provide --all-queued and/or --job-id to select jobs to cancel."
        )

    cancellable_statuses = {"queued"}
    if include_running:
        cancellable_statuses.add("running")

    jobs = load_optional_jobs(queue_dir, statuses=cancellable_statuses)
    if job_ids:
        jobs = [job for job in jobs if str(job.get("job_id")) in job_ids]

    cancelled: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for job in jobs:
        job_file = Path(str(job["job_file"]))
        row = {
            "job_id": job.get("job_id"),
            "kind": job.get("kind"),
            "status": job.get("status"),
            "job_file": str(job_file),
        }
        if dry_run:
            row["would_cancel"] = True
            cancelled.append(row)
            continue
        updated = _write_optional_job_status(
            job_file,
            "cancelled",
            cancelled_reason=reason,
        )
        cancelled.append(
            {
                "job_id": updated.get("job_id"),
                "kind": updated.get("kind"),
                "status": updated.get("status"),
                "job_file": str(job_file),
            }
        )

    if job_ids:
        matched = {str(item.get("job_id")) for item in cancelled}
        for job_id in sorted(job_ids - matched):
            skipped.append({"job_id": job_id, "reason": "not_found_or_not_cancellable"})

    return {
        "queue_dir": str(queue_dir.resolve()),
        "dry_run": dry_run,
        "cancelled": cancelled,
        "skipped": skipped,
        "cancelled_count": len(cancelled),
    }
