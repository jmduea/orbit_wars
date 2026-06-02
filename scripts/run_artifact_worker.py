from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    print(
        "prefer: uv run ow eval worker --run <run_dir> (agent path for artifact jobs)",
        file=sys.stderr,
    )

from src.artifacts.worker_env import bootstrap_artifact_worker_jax_env  # noqa: E402

bootstrap_artifact_worker_jax_env()

from src.artifacts.checkpoint_compat import (  # noqa: E402
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
)
from src.artifacts.checkpoint_eval import run_checkpoint_eval_job  # noqa: E402
from src.artifacts.docker_validation import (
    run_docker_validation_subprocess,  # noqa: E402
)
from src.artifacts.replay import maybe_write_jax_checkpoint_replay  # noqa: E402
from src.artifacts.run_paths import atomic_write_json  # noqa: E402
from src.artifacts.tournament.worker import run_tournament_promotion_job  # noqa: E402
from src.artifacts.worker_runner import run_optional_job_worker  # noqa: E402


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


def _docker_job_params(job: dict[str, object], cfg: Any) -> dict[str, object]:
    artifact_cfg = cfg.artifacts.artifact_pipeline
    tournament_cfg = cfg.artifacts.tournament
    return {
        "docker_image": str(
            job.get(
                "docker_image",
                getattr(
                    artifact_cfg,
                    "docker_image",
                    "gcr.io/kaggle-images/python-simulations",
                ),
            )
        ),
        "seed": int(
            job.get("seed", int(getattr(cfg, "seed", 42)) + int(job["update"]))
        ),
        "player_count": str(
            job.get(
                "player_count", getattr(artifact_cfg, "docker_player_count", "both")
            )
        ),
        "per_step_seconds": float(
            job.get(
                "per_step_seconds",
                getattr(
                    tournament_cfg,
                    "per_step_seconds",
                    getattr(artifact_cfg, "docker_timeout_seconds", 1.0),
                ),
            )
        ),
        "overage_budget_seconds": float(
            job.get(
                "overage_budget_seconds",
                getattr(tournament_cfg, "overage_budget_seconds", 60.0),
            )
        ),
        "episode_steps": int(
            job.get("episode_steps", getattr(cfg.artifacts.replay, "max_steps", 500))
        ),
    }


def _run_docker_validation_job(job: dict[str, object]) -> None:
    job_file = Path(str(job["job_file"]))
    checkpoint_path = Path(str(job["checkpoint_path"]))
    result_dir = _job_result_dir(job, job_file)
    output_dir = result_dir / "docker_validation"
    cfg = _load_checkpoint_config(checkpoint_path)
    params = _docker_job_params(job, cfg)
    docker_manifest = run_docker_validation_subprocess(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        docker_image=str(params["docker_image"]),
        seed=int(params["seed"]),
        player_count=str(params["player_count"]),
        per_step_seconds=float(params["per_step_seconds"]),
        overage_budget_seconds=float(params["overage_budget_seconds"]),
        episode_steps=int(params["episode_steps"]),
    )
    manifest_path = _job_manifest_path(job, result_dir)
    atomic_write_json(
        manifest_path,
        {
            "job_id": job["job_id"],
            "kind": job.get("kind"),
            "update": job["update"],
            "checkpoint_path": str(checkpoint_path),
            **docker_manifest,
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
        stdout_path=docker_manifest["stdout_path"],
        stderr_path=docker_manifest["stderr_path"],
        replay_html_paths=docker_manifest["replay_html_paths"],
        validation_ok=True,
    )


def _run_checkpoint_eval_job(job: dict[str, object]) -> None:
    job_file = Path(str(job["job_file"]))
    result_dir = _job_result_dir(job, job_file)
    summary = run_checkpoint_eval_job(job, result_dir=result_dir)
    manifest_path = _job_manifest_path(job, result_dir)
    atomic_write_json(
        manifest_path,
        {
            "job_id": job["job_id"],
            "kind": job.get("kind"),
            "update": job["update"],
            "checkpoint_path": str(job["checkpoint_path"]),
            "status": "completed",
            **summary,
        },
    )
    _write_status(
        job_file,
        "completed",
        result_dir=str(result_dir),
        result_manifest_path=str(manifest_path),
        validation_ok=True,
        tournament_id=summary["tournament_id"],
        promoted=bool(summary["promoted"]),
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
    job_root = job.get("_trusted_result_root")
    if job_root is not None:
        return Path(str(job_root))
    explicit = getattr(_trusted_result_root, "explicit", None)
    if explicit is not None:
        return Path(str(explicit))
    queue_dir = job_file.parent
    if queue_dir.name == "optional_jobs" and queue_dir.parent.name == "queue":
        return queue_dir.parent.parent / "evaluations"
    return queue_dir.parent / "evaluations"


def _process_job(job: dict[str, object]) -> None:
    kind = job.get("kind")
    if kind == "replay":
        _run_replay_job(job)
    elif kind == "docker_validation":
        _run_docker_validation_job(job)
    elif kind == "tournament":
        _run_tournament_job(job)
    elif kind == "checkpoint_eval":
        _run_checkpoint_eval_job(job)
    else:
        raise ValueError(f"unsupported job kind: {kind!r}")


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
    return run_optional_job_worker(
        args.queue_dir,
        _process_job,
        _write_status,
        result_root=args.result_root,
        once=args.once,
        poll_seconds=float(args.poll_seconds),
        idle_exit_seconds=args.idle_exit_seconds,
        recover_running=args.recover_running,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
