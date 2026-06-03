from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.artifacts.pipeline import write_optional_job
from src.artifacts.replay_schedule import checkpoint_replay_due
from src.config import TrainConfig


def queue_optional_jobs_if_due(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    log_path: Path,
    queue_dir: Path,
    result_root: Path | None = None,
    queue_replay: bool,
    queue_docker_validation: bool,
) -> list[Path]:
    job_paths: list[Path] = []
    if queue_replay and checkpoint_replay_due(cfg, update):
        job_paths.append(
            write_optional_job(
                queue_dir,
                kind="replay",
                update=update,
                checkpoint_path=checkpoint_path,
                payload={
                    "backend": cfg.artifacts.artifact_pipeline.replay_backend,
                    "log_path": str(log_path),
                    "replay_output_dir": cfg.artifacts.replay.output_dir,
                    "docker_image": cfg.artifacts.artifact_pipeline.docker_image,
                    "player_count": cfg.artifacts.artifact_pipeline.docker_player_count,
                    "timeout_seconds": cfg.artifacts.artifact_pipeline.docker_timeout_seconds,
                    "episode_steps": cfg.artifacts.replay.max_steps,
                    "seed": cfg.seed + update,
                },
                result_root=result_root,
            )
        )
    if queue_docker_validation:
        job_paths.append(
            write_optional_job(
                queue_dir,
                kind="docker_validation",
                update=update,
                checkpoint_path=checkpoint_path,
                payload={
                    "docker_image": cfg.artifacts.artifact_pipeline.docker_image,
                    "player_count": cfg.artifacts.artifact_pipeline.docker_player_count,
                    "timeout_seconds": cfg.artifacts.artifact_pipeline.docker_timeout_seconds,
                    "episode_steps": cfg.artifacts.replay.max_steps,
                    "seed": cfg.seed + update,
                },
                result_root=result_root,
            )
        )
    return job_paths


def queue_tournament_job_if_eligible(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    queue_dir: Path,
    result_root: Path | None,
    promotion_attempt_reason: str,
) -> Path | None:
    tournament_reasons = {"metric_eligible_queue_tournament", "tournament_only"}
    if promotion_attempt_reason not in tournament_reasons:
        return None
    if cfg.artifacts.promotion.strategy in {"hybrid", "tournament"}:
        cfg.artifacts.tournament.enabled = True
    if not cfg.artifacts.tournament.enabled:
        return None
    artifact_cfg = cfg.artifacts.artifact_pipeline
    kind = "checkpoint_eval" if artifact_cfg.checkpoint_eval_async else "tournament"
    return write_optional_job(
        queue_dir,
        kind=kind,
        update=update,
        checkpoint_path=checkpoint_path,
        payload={
            "campaign": cfg.output.campaign,
            "run_id": cfg.output.run_id,
            "docker_image": artifact_cfg.docker_image,
            "player_count": artifact_cfg.docker_player_count,
            "per_step_seconds": cfg.artifacts.tournament.per_step_seconds,
            "overage_budget_seconds": cfg.artifacts.tournament.overage_budget_seconds,
            "episode_steps": cfg.artifacts.replay.max_steps,
            "seed": cfg.seed + update,
        },
        result_root=result_root,
    )


def start_artifact_worker_if_needed(
    cfg: TrainConfig,
    *,
    queue_dir: Path,
    result_root: Path | None = None,
    worker_state: dict[str, subprocess.Popen[object]],
) -> None:
    if not cfg.artifacts.artifact_pipeline.worker_autostart:
        return
    worker = worker_state.get("process")
    if worker is not None:
        exit_code = worker.poll()
        if exit_code is None:
            return
        print(
            f"artifact_worker_exited code={exit_code}; restarting",
            flush=True,
        )
        worker_state.pop("process", None)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = queue_dir / "worker.stdout.log"
    stderr_path = queue_dir / "worker.stderr.log"
    repo_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_artifact_worker.py"),
        str(queue_dir),
        "--poll-seconds",
        str(cfg.artifacts.artifact_pipeline.worker_poll_seconds),
        "--idle-exit-seconds",
        str(cfg.artifacts.artifact_pipeline.worker_idle_exit_seconds),
    ]
    if result_root is not None:
        command.extend(["--result-root", str(result_root)])
    command.append("--recover-running")
    from src.artifacts.worker_env import artifact_worker_subprocess_env

    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    worker_state["process"] = subprocess.Popen(
        command,
        cwd=repo_root / "src",
        stdout=stdout,
        stderr=stderr,
        env=artifact_worker_subprocess_env(),
        start_new_session=True,
    )
    print(
        f"artifact_worker_started stdout_log={stdout_path} stderr_log={stderr_path}"
    )
