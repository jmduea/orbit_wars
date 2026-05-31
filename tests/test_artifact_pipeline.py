from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.artifacts.checkpoint_retention import prune_checkpoints
from src.artifacts.pipeline import (
    AsyncArtifactPipeline,
    CheckpointJob,
    commit_checkpoint_payload,
    load_active_optional_jobs,
    load_pending_optional_jobs,
    protected_paths_from_jobs,
    write_optional_job,
)
from src.config.schema import TrainConfig
from src.telemetry.metric_registry import filter_update_record


def _payload(update: int) -> dict[str, object]:
    return {"update": update, "params": {"value": update}}


def test_atomic_checkpoint_commit_writes_numbered_then_latest(tmp_path: Path):
    checkpoint_path = commit_checkpoint_payload(tmp_path, 7, _payload(7))

    latest_path = tmp_path / "jax_ckpt_last.pkl"
    assert checkpoint_path == tmp_path / "jax_ckpt_000007.pkl"
    assert latest_path.exists()
    with checkpoint_path.open("rb") as file:
        numbered = pickle.load(file)
    with latest_path.open("rb") as file:
        latest = pickle.load(file)

    assert numbered["update"] == 7
    assert latest["update"] == 7


def test_checkpoint_queue_coalesces_intermediate_but_keeps_final(tmp_path: Path):
    pipeline = AsyncArtifactPipeline(checkpoint_queue_size=1, autostart=False)

    first = CheckpointJob(update=1, run_dir=tmp_path, build_payload=lambda: _payload(1))
    second = CheckpointJob(update=2, run_dir=tmp_path, build_payload=lambda: _payload(2))
    final = CheckpointJob(
        update=3,
        run_dir=tmp_path,
        build_payload=lambda: _payload(3),
        final=True,
    )

    assert pipeline.submit_checkpoint(first) == []
    pipeline.submit_checkpoint(second)
    pipeline.submit_checkpoint(final)
    pipeline.start()
    results = pipeline.close(timeout_seconds=5.0)

    skipped_updates = {result.update for result in results if result.status == "skipped"}
    committed_updates = {result.update for result in results if result.status == "committed"}
    assert skipped_updates == {1, 2}
    assert committed_updates == {3}
    assert (tmp_path / "jax_ckpt_000003.pkl").exists()
    with (tmp_path / "jax_ckpt_last.pkl").open("rb") as file:
        latest = pickle.load(file)
    assert latest["update"] == 3


def test_checkpoint_worker_failure_is_drained(tmp_path: Path):
    pipeline = AsyncArtifactPipeline(checkpoint_queue_size=1)

    def fail() -> dict[str, object]:
        raise RuntimeError("boom")

    pipeline.submit_checkpoint(
        CheckpointJob(update=4, run_dir=tmp_path, build_payload=fail, final=True)
    )

    results = pipeline.close(timeout_seconds=5.0)
    failures = [result for result in results if result.status == "failed"]
    assert len(failures) == 1
    assert failures[0].update == 4
    assert "boom" in str(failures[0].error)


def test_retention_keeps_explicitly_protected_checkpoint(tmp_path: Path):
    log_path = tmp_path / "metrics.jsonl"
    log_path.write_text("", encoding="utf-8")
    for update in range(1, 5):
        commit_checkpoint_payload(tmp_path, update, _payload(update))

    protected = {tmp_path / "jax_ckpt_000002.pkl"}
    decision = prune_checkpoints(
        tmp_path,
        log_path=log_path,
        keep_last_n=1,
        keep_every_n_updates=0,
        keep_best_k_by_metric=0,
        best_metric_name="episode_reward_mean",
        best_metric_mode="max",
        min_update_for_pruning=0,
        dry_run_pruning=False,
        protected_paths=protected,
    )

    deleted_names = {path.name for path in decision.deleted}
    assert "jax_ckpt_000002.pkl" not in deleted_names
    assert (tmp_path / "jax_ckpt_000002.pkl").exists()
    assert (tmp_path / "jax_ckpt_000004.pkl").exists()


def test_checkpoint_pruning_can_read_preserved_metric_from_filtered_jsonl(
    tmp_path: Path,
):
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.losses = False
    cfg.artifacts.checkpoint_retention.best_metric_name = "total_loss"
    cfg.artifacts.checkpoint_retention.best_metric_mode = "max"

    log_path = tmp_path / "metrics.jsonl"
    records = [
        {
            "update": 1,
            "total_env_steps": 100,
            "completed_episodes": 2,
            "samples": 64,
            "overall_win_rate": 0.25,
            "win_rate_2p": 0.25,
            "first_place_rate_4p": 0.0,
            "episode_reward_mean": 0.1,
            "env_steps_per_sec": 500.0,
            "total_loss": 1.0,
        },
        {
            "update": 2,
            "total_env_steps": 200,
            "completed_episodes": 4,
            "samples": 64,
            "overall_win_rate": 0.5,
            "win_rate_2p": 0.5,
            "first_place_rate_4p": 0.0,
            "episode_reward_mean": 0.3,
            "env_steps_per_sec": 550.0,
            "total_loss": 2.0,
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(filter_update_record(record, cfg)) for record in records)
        + "\n",
        encoding="utf-8",
    )
    for update in (1, 2):
        (tmp_path / f"jax_ckpt_{update:06d}.pkl").write_bytes(b"checkpoint")

    decision = prune_checkpoints(
        tmp_path,
        log_path=log_path,
        keep_last_n=0,
        keep_every_n_updates=0,
        keep_best_k_by_metric=1,
        best_metric_name=cfg.artifacts.checkpoint_retention.best_metric_name,
        best_metric_mode=cfg.artifacts.checkpoint_retention.best_metric_mode,
        min_update_for_pruning=0,
        dry_run_pruning=False,
        protected_paths=None,
    )

    deleted_names = {path.name for path in decision.deleted}
    assert "jax_ckpt_000001.pkl" in deleted_names
    assert (tmp_path / "jax_ckpt_000002.pkl").exists()


def test_optional_job_file_roundtrip(tmp_path: Path):
    checkpoint_path = tmp_path / "jax_ckpt_000009.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="replay",
        update=9,
        checkpoint_path=checkpoint_path,
        payload={"log_path": str(tmp_path / "metrics.jsonl")},
    )

    jobs = load_pending_optional_jobs(tmp_path / "jobs")
    assert job_path.exists()
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "replay"
    assert jobs[0]["checkpoint_path"] == str(checkpoint_path)
    assert jobs[0]["log_path"] == str(tmp_path / "metrics.jsonl")


def test_optional_job_records_result_paths_when_result_root_is_provided(tmp_path: Path):
    checkpoint_path = tmp_path / "jax_ckpt_000009.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_path = write_optional_job(
        tmp_path / "queue" / "optional_jobs",
        kind="docker_validation",
        update=9,
        checkpoint_path=checkpoint_path,
        payload={"log_path": str(tmp_path / "metrics.jsonl")},
        result_root=tmp_path / "evaluations",
    )

    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert Path(job["result_dir"]).parent == tmp_path / "evaluations"
    assert job["result_manifest_path"] == str(Path(job["result_dir"]) / "manifest.json")


def test_running_optional_job_protects_checkpoint_from_retention(tmp_path: Path):
    log_path = tmp_path / "metrics.jsonl"
    log_path.write_text("", encoding="utf-8")
    checkpoint_path = commit_checkpoint_payload(tmp_path, 1, _payload(1))
    commit_checkpoint_payload(tmp_path, 2, _payload(2))
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="replay",
        update=1,
        checkpoint_path=checkpoint_path,
        payload={"log_path": str(log_path)},
    )
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_pending_optional_jobs(tmp_path / "jobs") == []
    protected = protected_paths_from_jobs(load_active_optional_jobs(tmp_path / "jobs"))
    prune_checkpoints(
        tmp_path,
        log_path=log_path,
        keep_last_n=0,
        keep_every_n_updates=0,
        keep_best_k_by_metric=0,
        best_metric_name="episode_reward_mean",
        best_metric_mode="max",
        min_update_for_pruning=0,
        dry_run_pruning=False,
        protected_paths=protected,
    )

    assert checkpoint_path.exists()


def test_replay_job_defaults_to_docker_backend(tmp_path: Path):
    from src.config import TrainConfig
    from src.jax.train.queue import queue_optional_jobs_if_due

    cfg = TrainConfig()
    cfg.artifacts.replay.enabled = True
    cfg.artifacts.replay.max_steps = 20
    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_paths = queue_optional_jobs_if_due(
        cfg,
        update=1,
        checkpoint_path=checkpoint_path,
        log_path=tmp_path / "metrics.jsonl",
        queue_dir=tmp_path / "jobs",
        result_root=tmp_path / "evaluations",
        queue_replay=True,
        queue_docker_validation=False,
    )

    jobs = load_pending_optional_jobs(tmp_path / "jobs")
    assert len(job_paths) == 1
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "replay"
    assert jobs[0]["backend"] == "docker"
    assert jobs[0]["episode_steps"] == 20
    assert jobs[0]["checkpoint_path"] == str(checkpoint_path)
    assert Path(jobs[0]["result_dir"]).parent == tmp_path / "evaluations"


def test_docker_job_can_be_queued_when_replay_is_disabled(tmp_path: Path):
    from src.config import TrainConfig
    from src.jax.train.queue import queue_optional_jobs_if_due

    cfg = TrainConfig()
    cfg.artifacts.replay.enabled = False
    cfg.artifacts.artifact_pipeline.docker_validation_async = True
    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_paths = queue_optional_jobs_if_due(
        cfg,
        update=1,
        checkpoint_path=checkpoint_path,
        log_path=tmp_path / "metrics.jsonl",
        queue_dir=tmp_path / "jobs",
        result_root=tmp_path / "evaluations",
        queue_replay=False,
        queue_docker_validation=True,
    )

    jobs = load_pending_optional_jobs(tmp_path / "jobs")
    assert len(job_paths) == 1
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "docker_validation"
    assert jobs[0]["checkpoint_path"] == str(checkpoint_path)
    assert Path(jobs[0]["result_dir"]).parent == tmp_path / "evaluations"


def test_docker_worker_records_replay_html_paths(tmp_path: Path, monkeypatch):
    from scripts import run_artifact_worker
    from src.config import TrainConfig

    cfg = TrainConfig()
    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    with checkpoint_path.open("wb") as file:
        pickle.dump({"params": {}, "config": cfg}, file)
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="replay",
        update=1,
        checkpoint_path=checkpoint_path,
        payload={"backend": "docker", "log_path": str(tmp_path / "metrics.jsonl")},
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        assert command[command.index("--per-step-seconds") + 1] == "1.0"
        assert command[command.index("--overage-budget-seconds") + 1] == "60.0"
        assert "--timeout-seconds" not in command
        replay_dir = output_dir / "replays"
        replay_dir.mkdir(parents=True, exist_ok=True)
        (replay_dir / "replay_u000001_2p.html").write_text("<html></html>", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.artifacts.docker_validation.subprocess.run", fake_run)

    run_artifact_worker._run_docker_validation_job(job)

    status = json.loads(job_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["backend"] == "docker"
    assert "error" not in status
    assert Path(status["output_dir"]).parent == Path(status["result_dir"])
    assert Path(status["result_manifest_path"]).exists()
    assert len(status["replay_html_paths"]) == 1
    assert status["replay_html_paths"][0].endswith("replay_u000001_2p.html")


def test_artifact_worker_retry_failed_processes_failed_job(
    tmp_path: Path, monkeypatch
):
    from scripts import run_artifact_worker

    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="replay",
        update=1,
        checkpoint_path=checkpoint_path,
        payload={"backend": "docker", "log_path": str(tmp_path / "metrics.jsonl")},
        result_root=tmp_path / "evaluations",
    )
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["status"] = "failed"
    job["error"] = "stale failure"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    processed: list[dict[str, object]] = []

    def fake_run_replay(job: dict[str, object]) -> None:
        processed.append(job)

    monkeypatch.setattr(run_artifact_worker, "_run_replay_job", fake_run_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_artifact_worker.py", str(tmp_path / "jobs"), "--once", "--retry-failed"],
    )

    assert run_artifact_worker.main() == 0
    assert len(processed) == 1
    assert processed[0]["status"] == "failed"
    status = json.loads(job_path.read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert "error" not in status


def test_local_replay_worker_writes_to_result_dir(tmp_path: Path, monkeypatch):
    from scripts import run_artifact_worker
    from src.config import TrainConfig

    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="replay",
        update=1,
        checkpoint_path=checkpoint_path,
        payload={"backend": "local", "log_path": str(tmp_path / "metrics.jsonl")},
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]

    monkeypatch.setattr(run_artifact_worker, "_load_checkpoint_config", lambda _path: TrainConfig())

    captured: dict[str, Path | None] = {}

    def fake_replay(cfg, *, update, checkpoint_path, log_path, output_dir=None):
        captured["output_dir"] = output_dir
        output_dir.mkdir(parents=True)
        metadata_path = output_dir / "replay.json"
        metadata_path.write_text("{}", encoding="utf-8")
        return metadata_path

    monkeypatch.setattr(run_artifact_worker, "maybe_write_jax_checkpoint_replay", fake_replay)

    run_artifact_worker._run_replay_job(job)

    status = json.loads(job_path.read_text(encoding="utf-8"))
    assert captured["output_dir"] == Path(status["result_dir"]) / "replay"
    assert Path(status["result_manifest_path"]).exists()
    assert status["metadata_path"] == str(Path(status["result_dir"]) / "replay" / "replay.json")


def test_worker_rejects_result_dir_escape(tmp_path: Path):
    from scripts import run_artifact_worker

    job_file = tmp_path / "jobs" / "job.json"
    job_file.parent.mkdir()
    job = {"job_id": "bad", "kind": "replay", "update": 1, "result_dir": str(tmp_path / "elsewhere")}

    with pytest.raises(ValueError, match="escapes"):
        run_artifact_worker._job_result_dir(job, job_file)


def test_worker_rejects_unsafe_legacy_job_id(tmp_path: Path):
    from scripts import run_artifact_worker

    job_file = tmp_path / "run" / "queue" / "optional_jobs" / "job.json"
    job_file.parent.mkdir(parents=True)
    job = {"job_id": "x/../../escape", "kind": "replay", "update": 1}

    with pytest.raises(ValueError, match="unsafe job_id"):
        run_artifact_worker._job_result_dir(job, job_file)


def test_worker_accepts_custom_result_root_from_trusted_worker_option(
    tmp_path: Path, monkeypatch
):
    from scripts import run_artifact_worker

    job_file = tmp_path / "custom_queue" / "job.json"
    job_file.parent.mkdir()
    trusted_root = tmp_path / "custom_evaluations"
    job = {
        "job_id": "ok",
        "kind": "replay",
        "update": 1,
        "result_root": str(tmp_path / "custom_evaluations"),
        "result_dir": str(trusted_root / "replay_u000001_ok"),
    }

    monkeypatch.setattr(
        run_artifact_worker._trusted_result_root,
        "explicit",
        trusted_root,
        raising=False,
    )

    assert run_artifact_worker._job_result_dir(job, job_file) == Path(job["result_dir"])


def test_artifact_worker_autostart_launches_background_process(tmp_path: Path, monkeypatch):
    from src.config import TrainConfig
    from src.jax.train import queue as train_queue

    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")
    launched: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    cfg = TrainConfig()
    cfg.artifacts.artifact_pipeline.worker_poll_seconds = 0.5
    cfg.artifacts.artifact_pipeline.worker_idle_exit_seconds = 1.0
    monkeypatch.setattr(train_queue.subprocess, "Popen", fake_popen)

    worker_state: dict[str, object] = {}
    train_queue.start_artifact_worker_if_needed(
        cfg,
        queue_dir=tmp_path,
        worker_state=worker_state,
    )

    command = launched["command"]
    assert command[1] == str(Path("scripts/run_artifact_worker.py").resolve())
    assert str(tmp_path) in command
    assert launched["kwargs"]["start_new_session"] is True
    assert launched["kwargs"]["env"]["JAX_PLATFORMS"] == "cpu"
    assert worker_state["process"].poll() is None


def test_artifact_worker_subprocess_env_honors_cpu_override(monkeypatch) -> None:
    from src.artifacts.worker_env import artifact_worker_subprocess_env

    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")

    env = artifact_worker_subprocess_env()

    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA"] == "1"


def test_checkpoint_queue_rejects_invalid_size():
    with pytest.raises(ValueError, match="checkpoint_queue_size"):
        AsyncArtifactPipeline(checkpoint_queue_size=0)
