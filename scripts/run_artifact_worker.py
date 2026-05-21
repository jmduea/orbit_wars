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

from src.artifact_pipeline import load_pending_optional_jobs  # noqa: E402
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
    output_dir = job_file.parent / f"docker_u{int(job['update']):06d}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_kaggle_docker_submission.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--player-count",
        str(job.get("player_count", "both")),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Docker validation failed with exit code {completed.returncode}")
    _write_status(job_file, "completed", output_dir=str(output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run queued Orbit Wars replay and Docker artifact jobs."
    )
    parser.add_argument("queue_dir", type=Path, help="Run-local artifact job directory")
    parser.add_argument("--once", action="store_true", help="Process current queue once")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    while True:
        for job in load_pending_optional_jobs(args.queue_dir):
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
        time.sleep(max(float(args.poll_seconds), 0.1))


if __name__ == "__main__":
    raise SystemExit(main())