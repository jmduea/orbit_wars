from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.pipeline import load_pending_optional_jobs, write_optional_job
from src.artifacts.tournament.types import LeaderboardRow, TournamentResult
from src.config import TrainConfig
from src.jax.train.queue import queue_tournament_job_if_eligible


def test_queue_tournament_job_uses_checkpoint_eval_when_async_enabled(
    tmp_path: Path,
) -> None:
    cfg = TrainConfig()
    cfg.artifacts.promotion.strategy = "hybrid"
    cfg.artifacts.tournament.enabled = True
    cfg.artifacts.artifact_pipeline.checkpoint_eval_async = True

    job_path = queue_tournament_job_if_eligible(
        cfg,
        update=10,
        checkpoint_path=tmp_path / "ckpt.pkl",
        queue_dir=tmp_path / "queue",
        result_root=tmp_path / "evaluations",
        promotion_attempt_reason="metric_eligible_queue_tournament",
    )
    assert job_path is not None
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "checkpoint_eval"
    assert payload["per_step_seconds"] == cfg.artifacts.tournament.per_step_seconds
    assert (
        payload["overage_budget_seconds"]
        == cfg.artifacts.tournament.overage_budget_seconds
    )


def test_checkpoint_eval_worker_runs_docker_then_tournament(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import run_artifact_worker

    checkpoint_path = tmp_path / "jax_ckpt_000010.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="checkpoint_eval",
        update=10,
        checkpoint_path=checkpoint_path,
        payload={"campaign": "c", "run_id": "r", "seed": 52},
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]

    docker_calls: list[dict[str, object]] = []
    tournament_calls: list[Path] = []

    def fake_docker_gate(
        job: dict[str, object], *, result_dir: Path, repo_root: Path
    ) -> tuple[dict[str, object], bool]:
        docker_calls.append(
            {"job": job, "result_dir": result_dir, "repo_root": repo_root}
        )
        output_dir = result_dir / "docker_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "submission.tar.gz").write_bytes(b"tar")
        manifest = {
            "validation_ok": True,
            "output_dir": str(output_dir),
            "package_path": str(output_dir / "submission.tar.gz"),
            "replay_html_paths": [],
            "stdout_path": str(output_dir / "stdout.log"),
            "stderr_path": str(output_dir / "stderr.log"),
        }
        (result_dir / "docker_manifest.json").write_text("{}", encoding="utf-8")
        return manifest, True

    fake_tournament = TournamentResult(
        tournament_id="t-1",
        output_dir=tmp_path / "tournament",
        outcomes=(),
        leaderboard=(
            LeaderboardRow(
                agent_id="candidate",
                checkpoint_path=str(checkpoint_path),
                games_played=1,
                win_rate_vs_sniper=0.2,
                gates_passed=False,
            ),
        ),
    )

    def fake_tournament_job(job: dict[str, object], *, result_dir: Path):
        tournament_calls.append(result_dir)
        return fake_tournament, None

    monkeypatch.setattr(
        "src.artifacts.checkpoint_eval.run_docker_gate_for_job",
        fake_docker_gate,
    )
    monkeypatch.setattr(
        "src.artifacts.checkpoint_eval.run_tournament_promotion_job",
        fake_tournament_job,
    )

    run_artifact_worker._run_checkpoint_eval_job(job)

    status = json.loads(job_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["validation_ok"] is True
    assert status["tournament_id"] == "t-1"
    assert status["promoted"] is False
    assert len(docker_calls) == 1
    assert len(tournament_calls) == 1
    assert tournament_calls[0].name == "tournament"
    manifest = json.loads(
        Path(status["result_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["validation_ok"] is True
    assert manifest["docker_manifest_path"]


def test_checkpoint_eval_worker_marks_failed_when_docker_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import run_artifact_worker

    checkpoint_path = tmp_path / "jax_ckpt_000010.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="checkpoint_eval",
        update=10,
        checkpoint_path=checkpoint_path,
        payload={"campaign": "c", "run_id": "r"},
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]

    def fake_docker_gate(
        _job: dict[str, object], *, result_dir: Path, repo_root: Path
    ) -> tuple[dict[str, object], bool]:
        raise RuntimeError("docker down")

    tournament_called = {"value": False}

    def fake_tournament_job(_job: dict[str, object], *, result_dir: Path):
        tournament_called["value"] = True
        raise AssertionError("tournament should not run")

    monkeypatch.setattr(
        "src.artifacts.checkpoint_eval.run_docker_gate_for_job",
        fake_docker_gate,
    )
    monkeypatch.setattr(
        "src.artifacts.checkpoint_eval.run_tournament_promotion_job",
        fake_tournament_job,
    )

    with pytest.raises(RuntimeError, match="docker down"):
        run_artifact_worker._run_checkpoint_eval_job(job)

    assert tournament_called["value"] is False


def test_checkpoint_eval_worker_completes_when_docker_soft_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import run_artifact_worker

    checkpoint_path = tmp_path / "jax_ckpt_000010.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="checkpoint_eval",
        update=10,
        checkpoint_path=checkpoint_path,
        payload={"campaign": "c", "run_id": "r"},
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]
    tournament_called = {"value": False}

    def fake_docker_gate(
        _job: dict[str, object], *, result_dir: Path, repo_root: Path
    ) -> tuple[dict[str, object], bool]:
        return {"validation_ok": False}, False

    def fake_tournament_job(_job: dict[str, object], *, result_dir: Path):
        tournament_called["value"] = True
        raise AssertionError("tournament should not run")

    monkeypatch.setattr(
        "src.artifacts.checkpoint_eval.run_docker_gate_for_job",
        fake_docker_gate,
    )
    monkeypatch.setattr(
        "src.artifacts.checkpoint_eval.run_tournament_promotion_job",
        fake_tournament_job,
    )

    run_artifact_worker._run_checkpoint_eval_job(job)

    assert tournament_called["value"] is False
    status = json.loads(job_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["validation_ok"] is False
    manifest = json.loads(
        Path(status["result_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["tournament_skipped"] is True
    assert manifest["promotion_reason"] == "docker_validation_failed"
