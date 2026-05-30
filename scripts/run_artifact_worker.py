from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.worker_env import bootstrap_artifact_worker_jax_env  # noqa: E402

bootstrap_artifact_worker_jax_env()

from src.artifacts.checkpoint_compat import (  # noqa: E402
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
)
from src.artifacts.pipeline import (  # noqa: E402
    load_optional_jobs,
)
from src.artifacts.replay import maybe_write_jax_checkpoint_replay  # noqa: E402
from src.artifacts.tournament.worker import run_tournament_promotion_job  # noqa: E402
from src.artifacts.run_paths import atomic_write_json  # noqa: E402


def _load_checkpoint_config(checkpoint_path: Path) -> Any:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"checkpoint does not contain config: {checkpoint_path}")
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )
    return checkpoint["config"]


def _write_status(job_file: Path, status: str, **fields: object) -> None:
    payload = json.loads(job_file.read_text(encoding="utf-8"))
    payload.update(fields)
    payload["status"] = status
    if status in {"running", "completed"}:
        payload.pop("error", None)
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
    result_dir = _job_result_dir(job, job_file)
    cfg = _load_checkpoint_config(checkpoint_path)
    artifact_cfg = cfg.artifacts.artifact_pipeline
    backend = str(job.get("backend", getattr(artifact_cfg, "replay_backend", "docker")))
    if backend == "docker":
        job.setdefault("docker_image", getattr(artifact_cfg, "docker_image", "gcr.io/kaggle-images/python-simulations"))
        job.setdefault("player_count", getattr(artifact_cfg, "docker_player_count", "both"))
        job.setdefault("timeout_seconds", getattr(artifact_cfg, "docker_timeout_seconds", 1.0))
        job.setdefault("episode_steps", getattr(cfg.artifacts.replay, "max_steps", 500))
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
        output_dir=result_dir / "replay",
    )
    manifest_path = _job_manifest_path(job, result_dir)
    atomic_write_json(
        manifest_path,
        {
            "job_id": job["job_id"],
            "kind": job.get("kind"),
            "update": job["update"],
            "checkpoint_path": str(checkpoint_path),
            "metadata_path": str(metadata_path) if metadata_path is not None else None,
            "output_dir": str(result_dir / "replay"),
            "status": "completed",
        },
    )
    _write_status(
        job_file,
        "completed",
        result_dir=str(result_dir),
        result_manifest_path=str(manifest_path),
        metadata_path=str(metadata_path) if metadata_path is not None else None,
    )


def _run_docker_validation_job(job: dict[str, object]) -> None:
    job_file = Path(str(job["job_file"]))
    checkpoint_path = Path(str(job["checkpoint_path"]))
    result_dir = _job_result_dir(job, job_file)
    output_dir = result_dir / "docker_validation"
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
        "--per-step-seconds",
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
        env=dict(os.environ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Docker validation failed with exit code {completed.returncode}; "
            f"see {output_dir / 'stderr.log'}"
        )
    replay_html_paths = sorted(str(path) for path in (output_dir / "replays").glob("*.html"))
    manifest_path = _job_manifest_path(job, result_dir)
    atomic_write_json(
        manifest_path,
        {
            "job_id": job["job_id"],
            "kind": job.get("kind"),
            "update": job["update"],
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "replay_html_paths": replay_html_paths,
            "status": "completed",
        },
    )
    _write_status(
        job_file,
        "completed",
        backend="docker",
        result_dir=str(result_dir),
        result_manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        stdout_path=str(output_dir / "stdout.log"),
        stderr_path=str(output_dir / "stderr.log"),
        replay_html_paths=replay_html_paths,
    )



def _run_tournament_job(job: dict[str, object]) -> None:
    job_file = Path(str(job["job_file"]))
    result_dir = _job_result_dir(job, job_file)
    tournament, promotion_attempt = run_tournament_promotion_job(job, result_dir=result_dir)
    manifest_path = _job_manifest_path(job, result_dir)
    atomic_write_json(
        manifest_path,
        {
            "job_id": job["job_id"],
            "kind": job.get("kind"),
            "update": job["update"],
            "checkpoint_path": str(job["checkpoint_path"]),
            "tournament_id": tournament.tournament_id,
            "leaderboard_path": str(result_dir / "leaderboard.json"),
            "promoted": bool(promotion_attempt and promotion_attempt.promoted),
            "promotion_reason": promotion_attempt.reason if promotion_attempt else "no_passing_row",
            "status": "completed",
        },
    )
    _write_status(
        job_file,
        "completed",
        result_dir=str(result_dir),
        result_manifest_path=str(manifest_path),
        tournament_id=tournament.tournament_id,
        promoted=bool(promotion_attempt and promotion_attempt.promoted),
    )

def _job_result_dir(
    job: dict[str, object],
    job_file: Path,
) -> Path:
    trusted_root = _trusted_result_root(job, job_file)
    raw_result_dir = job.get("result_dir")
    if raw_result_dir is None:
        job_id = str(job["job_id"])
        if not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
            raise ValueError(f"unsafe job_id: {job_id!r}")
        kind = str(job.get("kind", "job"))
        result_dir = trusted_root / f"{kind}_u{int(job['update']):06d}_{job_id}"
    else:
        result_dir = Path(str(raw_result_dir))
    if not result_dir.resolve().is_relative_to(trusted_root.resolve()):
        raise ValueError(f"job result_dir escapes evaluations directory: {result_dir}")
    return result_dir


def _job_manifest_path(job: dict[str, object], result_dir: Path) -> Path:
    manifest_path = Path(str(job.get("result_manifest_path") or result_dir / "manifest.json"))
    if not manifest_path.resolve().is_relative_to(result_dir.resolve()):
        raise ValueError(f"job result_manifest_path escapes result directory: {manifest_path}")
    return manifest_path


def _trusted_result_root(job: dict[str, object], job_file: Path) -> Path:
    explicit = getattr(_trusted_result_root, "explicit", None)
    if explicit is not None:
        return Path(str(explicit))
    queue_dir = job_file.parent
    if queue_dir.name == "optional_jobs" and queue_dir.parent.name == "queue":
        return queue_dir.parent.parent / "evaluations"
    return queue_dir.parent / "evaluations"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run queued Orbit Wars replay and Docker artifact jobs."
    )
    parser.add_argument("queue_dir", type=Path, help="Run-local artifact job directory")
    parser.add_argument("--result-root", type=Path, default=None, help="Trusted result artifact root")
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
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also process failed jobs, for explicit one-off retry workflows.",
    )
    args = parser.parse_args()
    if args.result_root is not None:
        setattr(_trusted_result_root, "explicit", args.result_root)

    last_work_time = time.monotonic()
    while True:
        statuses = {"queued"}
        if args.recover_running:
            statuses.add("running")
        if args.retry_failed:
            statuses.add("failed")
        jobs = load_optional_jobs(args.queue_dir, statuses=statuses)
        if jobs:
            last_work_time = time.monotonic()
        for job in jobs:
            job_file = Path(str(job["job_file"]))
            if args.result_root is not None:
                job["_trusted_result_root"] = str(args.result_root)
            try:
                _write_status(job_file, "running")
                if job.get("kind") == "replay":
                    _run_replay_job(job)
                elif job.get("kind") == "docker_validation":
                    _run_docker_validation_job(job)
                elif job.get("kind") == "tournament":
                    _run_tournament_job(job)
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
