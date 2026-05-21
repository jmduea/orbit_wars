from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from src.artifact_pipeline import (
    AsyncArtifactPipeline,
    CheckpointJob,
    commit_checkpoint_payload,
    load_active_optional_jobs,
    load_pending_optional_jobs,
    protected_paths_from_jobs,
    write_optional_job,
)
from src.checkpoint_retention import prune_checkpoints


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
    from src.jax_train import _queue_optional_jobs_if_due

    cfg = TrainConfig()
    cfg.replay.enabled = True
    cfg.replay.max_steps = 20
    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_paths = _queue_optional_jobs_if_due(
        cfg,
        update=1,
        checkpoint_path=checkpoint_path,
        log_path=tmp_path / "metrics.jsonl",
        queue_dir=tmp_path / "jobs",
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


def test_docker_job_can_be_queued_when_replay_is_disabled(tmp_path: Path):
    from src.config import TrainConfig
    from src.jax_train import _queue_optional_jobs_if_due

    cfg = TrainConfig()
    cfg.replay.enabled = False
    cfg.artifact_pipeline.docker_validation_async = True
    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_paths = _queue_optional_jobs_if_due(
        cfg,
        update=1,
        checkpoint_path=checkpoint_path,
        log_path=tmp_path / "metrics.jsonl",
        queue_dir=tmp_path / "jobs",
        queue_replay=False,
        queue_docker_validation=True,
    )

    jobs = load_pending_optional_jobs(tmp_path / "jobs")
    assert len(job_paths) == 1
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "docker_validation"
    assert jobs[0]["checkpoint_path"] == str(checkpoint_path)


def test_artifact_worker_autostart_launches_background_process(tmp_path: Path, monkeypatch):
    from src import jax_train
    from src.config import TrainConfig

    launched: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    cfg = TrainConfig()
    cfg.artifact_pipeline.worker_poll_seconds = 0.5
    cfg.artifact_pipeline.worker_idle_exit_seconds = 1.0
    monkeypatch.setattr(jax_train.subprocess, "Popen", fake_popen)

    worker_state: dict[str, object] = {}
    jax_train._start_artifact_worker_if_needed(
        cfg,
        queue_dir=tmp_path,
        worker_state=worker_state,
    )

    command = launched["command"]
    assert "scripts/run_artifact_worker.py" in str(command)
    assert str(tmp_path) in command
    assert launched["kwargs"]["start_new_session"] is True
    assert worker_state["process"].poll() is None


def test_checkpoint_queue_rejects_invalid_size():
    with pytest.raises(ValueError, match="checkpoint_queue_size"):
        AsyncArtifactPipeline(checkpoint_queue_size=0)