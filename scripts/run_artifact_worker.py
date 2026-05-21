from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifact_pipeline import (  # noqa: E402
    load_optional_jobs,
    load_pending_optional_jobs,
)
from src.replay import maybe_write_jax_checkpoint_replay  # noqa: E402


def _load_checkpoint_config(checkpoint_path: Path) -> Any:
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"checkpoint does not contain config: {checkpoint_path}")
    return checkpoint["config"]


def _write_status(job_file: Path, status: str, **fields: object) -> None:
    payload = json.loads(job_file.read_text(encoding="utf-8"))
    payload.update(fields)
    payload["status"] = status
    payload["updated_at_unix"] = time.time()
    tmp_path = job_file.with_suffix(job_file.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(job_file)


def _run_replay_job(job: dict[str, object]) -> None:
    job_file = Path(str(job["job_file"]))
    checkpoint_path = Path(str(job["checkpoint_path"]))
    log_path = Path(str(job["log_path"]))
    cfg = _load_checkpoint_config(checkpoint_path)
    artifact_cfg = getattr(cfg, "artifact_pipeline", None)
    backend = str(job.get("backend", getattr(artifact_cfg, "replay_backend", "docker")))
    if backend == "docker":
        job.setdefault("docker_image", getattr(artifact_cfg, "docker_image", "gcr.io/kaggle-images/python-simulations"))
        job.setdefault("player_count", getattr(artifact_cfg, "docker_player_count", "both"))
        job.setdefault("timeout_seconds", getattr(artifact_cfg, "docker_timeout_seconds", 1.0))
        job.setdefault("episode_steps", getattr(cfg.replay, "max_steps", 500))
        job.setdefault("seed", int(getattr(cfg, "seed", 42)) + int(job["update"]))
        _run_docker_validation_job(job)
        return
    if backend != "local":
        raise ValueError(f"unsupported replay backend: {backend!r}")
    metadata_path = maybe_write_jax_checkpoint_replay(
        cfg,
        update=int(job["update"]),
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )
    _write_status(
        job_file,
        "completed",
        metadata_path=str(metadata_path) if metadata_path is not None else None,
    )


def _run_docker_validation_job(job: dict[str, object]) -> None:
    job_file = Path(str(job["job_file"]))
    checkpoint_path = Path(str(job["checkpoint_path"]))
    output_dir = job_file.parent / f"docker_u{int(job['update']):06d}_{job['job_id']}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_kaggle_docker_submission.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--docker-image",
        str(job.get("docker_image", "gcr.io/kaggle-images/python-simulations")),
        "--seed",
        str(job.get("seed", 42)),
        "--player-count",
        str(job.get("player_count", "both")),
        "--timeout-seconds",
        str(job.get("timeout_seconds", 1.0)),
        "--episode-steps",
        str(job.get("episode_steps", 500)),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Docker validation failed with exit code {completed.returncode}; "
            f"see {output_dir / 'stderr.log'}"
        )
    _write_status(
        job_file,
        "completed",
        backend="docker",
        output_dir=str(output_dir),
        stdout_path=str(output_dir / "stdout.log"),
        stderr_path=str(output_dir / "stderr.log"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run queued Orbit Wars replay and Docker artifact jobs."
    )
    parser.add_argument("queue_dir", type=Path, help="Run-local artifact job directory")
    parser.add_argument("--once", action="store_true", help="Process current queue once")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=None,
        help="Exit after this many seconds without queued jobs.",
    )
    parser.add_argument(
        "--recover-running",
        action="store_true",
        help="Also process jobs left in running status by a dead worker.",
    )
    args = parser.parse_args()

    last_work_time = time.monotonic()
    while True:
        jobs = (
            load_optional_jobs(args.queue_dir, statuses={"queued", "running"})
            if args.recover_running
            else load_pending_optional_jobs(args.queue_dir)
        )
        if jobs:
            last_work_time = time.monotonic()
        for job in jobs:
            job_file = Path(str(job["job_file"]))
            try:
                _write_status(job_file, "running")
                if job.get("kind") == "replay":
                    _run_replay_job(job)
                elif job.get("kind") == "docker_validation":
                    _run_docker_validation_job(job)
                else:
                    raise ValueError(f"unsupported job kind: {job.get('kind')!r}")
            except Exception as exc:
                _write_status(job_file, "failed", error=str(exc))
                return 1
        if args.once:
            return 0
        if (
            args.idle_exit_seconds is not None
            and time.monotonic() - last_work_time >= max(float(args.idle_exit_seconds), 0.0)
        ):
            return 0
        time.sleep(max(float(args.poll_seconds), 0.1))


if __name__ == "__main__":
    raise SystemExit(main())