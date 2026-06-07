from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.promotion import (
    promote_if_better,
    resolve_from_promoted,
)
from src.artifacts.promotion_manifest import commit_promotion, promoted_manifest_path
from src.artifacts.run_paths import RunContext, resolve_run_paths, write_run_manifests
from src.config.schema import TrainConfig


def _run_context(tmp_path: Path, *, campaign: str = "capacity") -> RunContext:
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.output.campaign = campaign
    cfg.output.run_id = "run-001"
    _, context = resolve_run_paths(cfg)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    context.logs_dir.mkdir(parents=True, exist_ok=True)
    context.indexes_dir.mkdir(parents=True, exist_ok=True)
    return context


def _write_metric_log(log_path: Path, records: list[dict[str, object]]) -> None:
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_commit_promotion_writes_manifest_campaign_and_index(tmp_path: Path) -> None:
    context = _run_context(tmp_path, campaign="commit_helper")
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    manifest_out = commit_promotion(
        campaign_dir=context.campaign_dir,
        campaign_manifest_path=context.campaign_manifest_path,
        indexes_dir=context.indexes_dir,
        promoted_payload={
            "campaign": context.campaign_slug,
            "checkpoint_path": str(checkpoint_path),
            "metric_name": "episode_reward_mean",
            "metric_value": 0.5,
        },
        campaign_updates={
            "current_best_value": 0.5,
            "current_best_run_id": context.run_id,
        },
        index_record={
            "campaign": context.campaign_slug,
            "run_id": context.run_id,
            "update": 1,
            "metric_name": "episode_reward_mean",
            "metric_value": 0.5,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    assert manifest_out == promoted_manifest_path(context.campaign_dir)
    campaign_manifest = json.loads(context.campaign_manifest_path.read_text())
    assert campaign_manifest["current_best_value"] == pytest.approx(0.5)
    index_lines = (context.indexes_dir / "promoted.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 1
    assert "promoted_manifest_path" in json.loads(index_lines[0])


def test_promote_if_better_uses_injected_metrics_without_log_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _run_context(tmp_path)
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.artifacts.promotion.metric_name = "episode_reward_mean"
    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000002.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    def fail_scan(*_args: object, **_kwargs: object) -> dict[int, float]:
        raise AssertionError("collect_metric_by_update should not run when metrics are injected")

    monkeypatch.setattr(
        "src.artifacts.promotion.collect_metric_by_update",
        fail_scan,
    )

    attempt, run_best = promote_if_better(
        cfg,
        context,
        checkpoint_path=checkpoint_path,
        update=2,
        log_path=context.log_path,
        run_best_value=None,
        metrics_by_update={2: 0.7},
    )

    assert attempt.promoted is True
    assert run_best == pytest.approx(0.7)


def test_promote_if_better_cas_max_updates_manifest_and_index(tmp_path: Path) -> None:
    context = _run_context(tmp_path)
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.output.campaign = context.campaign_slug
    cfg.artifacts.promotion.metric_name = "episode_reward_mean"
    cfg.artifacts.promotion.metric_mode = "max"
    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})

    log_path = context.log_path
    _write_metric_log(
        log_path,
        [
            {"update": 1, "episode_reward_mean": 0.2},
            {"update": 2, "episode_reward_mean": 0.5},
        ],
    )
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000002.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    first, run_best = promote_if_better(
        cfg,
        context,
        checkpoint_path=checkpoint_path,
        update=2,
        log_path=log_path,
        run_best_value=None,
    )
    assert first.promoted is True
    assert run_best == pytest.approx(0.5)
    manifest = json.loads(promoted_manifest_path(context.campaign_dir).read_text())
    assert manifest["checkpoint_path"] == str(checkpoint_path.resolve())
    assert manifest["metric_value"] == pytest.approx(0.5)
    assert (context.indexes_dir / "promoted.jsonl").exists()

    checkpoint_path_2 = context.checkpoints_dir / "jax_ckpt_000003.pkl"
    checkpoint_path_2.write_bytes(b"checkpoint-2")
    _write_metric_log(log_path, [{"update": 3, "episode_reward_mean": 0.4}])
    second, _ = promote_if_better(
        cfg,
        context,
        checkpoint_path=checkpoint_path_2,
        update=3,
        log_path=log_path,
        run_best_value=0.5,
    )
    assert second.promoted is False
    assert second.reason == "run_best_unchanged"


def test_promote_if_better_hybrid_defers_to_tournament(tmp_path: Path) -> None:
    context = _run_context(tmp_path, campaign="hybrid")
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.artifacts.promotion.metric_name = "episode_reward_mean"
    cfg.artifacts.promotion.metric_mode = "max"
    cfg.artifacts.promotion.strategy = "hybrid"
    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})

    log_path = context.log_path
    _write_metric_log(log_path, [{"update": 1, "episode_reward_mean": 0.9}])
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    attempt, run_best = promote_if_better(
        cfg,
        context,
        checkpoint_path=checkpoint_path,
        update=1,
        log_path=log_path,
        run_best_value=None,
    )

    assert attempt.promoted is False
    assert attempt.reason == "metric_eligible_queue_tournament"
    assert run_best == pytest.approx(0.9)
    assert not promoted_manifest_path(context.campaign_dir).exists()


def test_promote_if_better_tournament_only_defers_to_worker(tmp_path: Path) -> None:
    context = _run_context(tmp_path, campaign="tournament_only")
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.artifacts.promotion.metric_name = "episode_reward_mean"
    cfg.artifacts.promotion.strategy = "tournament"
    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})

    log_path = context.log_path
    _write_metric_log(log_path, [{"update": 1, "episode_reward_mean": 0.9}])
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    attempt, run_best = promote_if_better(
        cfg,
        context,
        checkpoint_path=checkpoint_path,
        update=1,
        log_path=log_path,
        run_best_value=None,
    )

    assert attempt.promoted is False
    assert attempt.reason == "tournament_only"
    assert run_best == pytest.approx(0.9)


def test_promote_if_better_respects_min_mode(tmp_path: Path) -> None:
    context = _run_context(tmp_path, campaign="latency")
    cfg = TrainConfig()
    cfg.output.root = str(tmp_path / "outputs")
    cfg.artifacts.promotion.metric_name = "total_loss"
    cfg.artifacts.promotion.metric_mode = "min"
    write_run_manifests(cfg, context, {"job_type": "train", "backend": "jax"})

    log_path = context.log_path
    _write_metric_log(log_path, [{"update": 1, "total_loss": 2.0}])
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    attempt, run_best = promote_if_better(
        cfg,
        context,
        checkpoint_path=checkpoint_path,
        update=1,
        log_path=log_path,
        run_best_value=None,
    )
    assert attempt.promoted is True
    assert run_best == pytest.approx(2.0)


def test_resolve_from_promoted_sets_checkpoint_path(tmp_path: Path) -> None:
    context = _run_context(tmp_path, campaign="resume_me")
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000010.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path = promoted_manifest_path(context.campaign_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": "resume_me",
                "checkpoint_path": str(checkpoint_path),
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_from_promoted("resume_me", str(tmp_path / "outputs"))

    assert resolved["campaign"] == "resume_me"
    assert resolved["checkpoint_path"] == str(checkpoint_path)


def test_from_promoted_compose_sets_resume_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import compose_hydra_train_config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    context = _run_context(tmp_path, campaign="promoted_campaign")
    checkpoint_path = context.checkpoints_dir / "jax_ckpt_000001.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path = promoted_manifest_path(context.campaign_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"checkpoint_path": str(checkpoint_path)}),
        encoding="utf-8",
    )

    cfg = compose_hydra_train_config(
        overrides=[
            "output.root=outputs",
            "from_promoted=promoted_campaign",
            "print_resolved_config=true",
        ]
    )

    assert cfg.output.campaign == "promoted_campaign"
    assert cfg.resume_checkpoint == str(checkpoint_path)
    assert cfg.from_promoted is None


def test_append_produced_artifact_is_idempotent(tmp_path: Path) -> None:
    from src.artifacts.run_paths import append_produced_artifact

    context = _run_context(tmp_path)
    context.manifest_path.write_text(
        json.dumps({"produced_artifacts": []}),
        encoding="utf-8",
    )
    entry = {
        "kind": "checkpoint",
        "update": 1,
        "path": str(context.checkpoints_dir / "jax_ckpt_000001.pkl"),
        "final": False,
    }
    append_produced_artifact(context.manifest_path, entry)
    append_produced_artifact(context.manifest_path, entry)
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["produced_artifacts"]) == 1
    assert manifest["produced_artifacts"][0]["update"] == 1
