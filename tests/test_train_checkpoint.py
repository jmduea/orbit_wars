from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
from src.jax.train.checkpoint import (
    CheckpointHandler,
    CheckpointResult,
    checkpoint_payload_builder,
    restore_historical_snapshot_pool,
)


def test_checkpoint_handler_skips_artifact_jobs_when_pipeline_disabled(
    tmp_path: Path,
) -> None:
    """artifact_pipeline.enabled=false must not queue docker replay or sync replay."""
    log_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "jax_ckpt_000100.pkl"
    checkpoint_path.write_bytes(b"ckpt")
    cfg = SimpleNamespace(
        artifacts=SimpleNamespace(
            artifact_pipeline=SimpleNamespace(
                enabled=False,
                replay_async=True,
                docker_validation_async=False,
            ),
            checkpoint_retention=SimpleNamespace(
                keep_last_n=1,
                keep_every_n_updates=0,
                keep_best_k_by_metric=0,
                best_metric_name="overall_win_rate",
                best_metric_mode="max",
                min_update_for_pruning=0,
                dry_run_pruning=True,
            ),
            promotion=SimpleNamespace(
                enabled=False,
                strategy="metric",
                metric_name="episode_reward_mean",
            ),
            tournament=SimpleNamespace(enabled=False),
            replay=SimpleNamespace(enabled=True, output_dir="replays", max_steps=500),
        ),
        telemetry=SimpleNamespace(wandb=SimpleNamespace(log_artifacts=False)),
    )
    handler = CheckpointHandler(
        cfg=cfg,
        run_dir=tmp_path,
        log_path=log_path,
        run_context=SimpleNamespace(
            evaluations_dir=tmp_path / "eval",
            manifest_path=tmp_path / "manifest.json",
        ),
        telemetry=MagicMock(),
        artifact_queue_dir=tmp_path / "queue",
        checkpoint_pipeline=None,
    )

    with (
        patch(
            "src.jax.train.checkpoint.queue_optional_jobs_if_due",
        ) as queue_jobs,
        patch(
            "src.jax.train.checkpoint.maybe_write_jax_checkpoint_replay",
        ) as sync_replay,
        patch(
            "src.jax.train.checkpoint.queue_tournament_job_if_eligible",
        ) as queue_tournament,
    ):
        handler.handle_results(
            [
                CheckpointResult(
                    job_id="sync-100",
                    update=100,
                    status="committed",
                    numbered_path=checkpoint_path,
                    latest_path=tmp_path / "jax_ckpt_last.pkl",
                    final=False,
                )
            ]
        )

    queue_jobs.assert_not_called()
    sync_replay.assert_not_called()
    queue_tournament.assert_not_called()


def test_checkpoint_handler_parses_shared_metric_once(tmp_path: Path) -> None:
    import json

    log_path = tmp_path / "metrics.jsonl"
    log_path.write_text(
        json.dumps({"update": 5, "episode_reward_mean": 0.8}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "jax_ckpt_000005.pkl"
    checkpoint_path.write_bytes(b"ckpt")
    cfg = SimpleNamespace(
        artifacts=SimpleNamespace(
            artifact_pipeline=SimpleNamespace(
                enabled=False,
                replay_async=False,
                docker_validation_async=False,
            ),
            checkpoint_retention=SimpleNamespace(
                keep_last_n=1,
                keep_every_n_updates=0,
                keep_best_k_by_metric=2,
                best_metric_name="episode_reward_mean",
                best_metric_mode="max",
                min_update_for_pruning=0,
                dry_run_pruning=True,
            ),
            promotion=SimpleNamespace(
                enabled=True,
                strategy="hybrid",
                metric_name="episode_reward_mean",
                metric_mode="max",
            ),
            tournament=SimpleNamespace(enabled=False),
            replay=SimpleNamespace(enabled=False),
        ),
        telemetry=SimpleNamespace(wandb=SimpleNamespace(log_artifacts=False)),
    )
    handler = CheckpointHandler(
        cfg=cfg,
        run_dir=tmp_path,
        log_path=log_path,
        run_context=SimpleNamespace(
            campaign_slug="shared_metric",
            campaign_dir=tmp_path / "campaign",
            campaign_manifest_path=tmp_path / "campaign" / "campaign_manifest.json",
            indexes_dir=tmp_path / "indexes",
            run_dir=tmp_path,
            run_id="run-1",
            evaluations_dir=tmp_path / "eval",
            manifest_path=tmp_path / "manifest.json",
        ),
        telemetry=MagicMock(),
        artifact_queue_dir=tmp_path / "queue",
        checkpoint_pipeline=None,
    )
    (tmp_path / "campaign").mkdir(parents=True, exist_ok=True)
    (tmp_path / "indexes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"produced_artifacts": []}),
        encoding="utf-8",
    )

    with patch(
        "src.jax.train.checkpoint.collect_metric_by_update",
        wraps=__import__(
            "src.artifacts.checkpoint_retention",
            fromlist=["collect_metric_by_update"],
        ).collect_metric_by_update,
    ) as collect_metrics:
        handler.handle_results(
            [
                CheckpointResult(
                    job_id="sync-5",
                    update=5,
                    status="committed",
                    numbered_path=checkpoint_path,
                    latest_path=tmp_path / "jax_ckpt_last.pkl",
                    final=False,
                )
            ]
        )

    assert collect_metrics.call_count == 1
    assert collect_metrics.call_args.args == (log_path, "episode_reward_mean")


def test_checkpoint_handler_records_failed_results(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    cfg = SimpleNamespace(
        artifacts=SimpleNamespace(
            artifact_pipeline=SimpleNamespace(
                replay_async=False,
                docker_validation_async=False,
            ),
            checkpoint_retention=SimpleNamespace(
                keep_last_n=1,
                keep_every_n_updates=0,
                keep_best_k_by_metric=0,
                best_metric_name="overall_win_rate",
                best_metric_mode="max",
                min_update_for_pruning=0,
                dry_run_pruning=True,
            ),
            promotion=SimpleNamespace(strategy="metric"),
            tournament=SimpleNamespace(enabled=False),
            replay=SimpleNamespace(enabled=False),
        ),
        telemetry=SimpleNamespace(wandb=SimpleNamespace(log_artifacts=False)),
    )
    telemetry = MagicMock()
    handler = CheckpointHandler(
        cfg=cfg,
        run_dir=tmp_path,
        log_path=log_path,
        run_context=SimpleNamespace(evaluations_dir=tmp_path / "eval"),
        telemetry=telemetry,
        artifact_queue_dir=tmp_path / "queue",
        checkpoint_pipeline=None,
    )

    handler.handle_results(
        [
            SimpleNamespace(
                update=2,
                status="failed",
                final=False,
                reason="worker_error",
                error="disk full",
                committed=False,
                numbered_path=None,
                latest_path=None,
            )
        ]
    )

    assert handler.first_failure() is not None
    assert log_path.exists()
    telemetry.log.assert_called_once()


def test_restore_historical_snapshot_pool_returns_fallback_on_bad_payload() -> None:
    import jax.numpy as jnp

    from src.jax.train.checkpoint import HistoricalSnapshotPool

    fallback = HistoricalSnapshotPool(
        params={"w": jnp.zeros((2, 2))},
        snapshot_ids=jnp.zeros((2,), dtype=jnp.int32),
        snapshot_updates=jnp.zeros((2,), dtype=jnp.int32),
        valid_mask=jnp.zeros((2,), dtype=bool),
    )

    restored = restore_historical_snapshot_pool("not-a-dict", fallback)

    assert restored is fallback


def test_restore_historical_snapshot_pool_returns_fallback_on_shape_mismatch() -> None:
    import jax.numpy as jnp

    from src.jax.train.checkpoint import HistoricalSnapshotPool

    fallback = HistoricalSnapshotPool(
        params={"w": jnp.zeros((1, 2, 2))},
        snapshot_ids=jnp.zeros((1,), dtype=jnp.int32),
        snapshot_updates=jnp.zeros((1,), dtype=jnp.int32),
        valid_mask=jnp.zeros((1,), dtype=bool),
    )
    checkpoint_pool = {
        "params": {"w": jnp.ones((2, 2, 2))},
        "snapshot_ids": jnp.array([1, 2], dtype=jnp.int32),
        "snapshot_updates": jnp.array([3, 4], dtype=jnp.int32),
        "valid_mask": jnp.array([True, True]),
        "next_slot": 0,
        "next_id": 3,
    }

    restored = restore_historical_snapshot_pool(checkpoint_pool, fallback)

    assert restored is fallback


def test_checkpoint_payload_builder_includes_curriculum_state() -> None:
    from src.config import TrainConfig
    from src.jax.policy import build_jax_policy
    from src.jax.train import init_train_state
    from src.training.curriculum import CurriculumController

    cfg = TrainConfig()
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 8
    cfg.model.attention_heads = 2
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)

    payload = checkpoint_payload_builder(
        train_state,
        cfg,
        key=jax.random.PRNGKey(2),
        update=1,
        total_env_steps=10,
        completed_episodes=2,
        curriculum=controller,
        historical_pool=None,
    )()

    assert "curriculum_state" in payload
    assert payload["update"] == 1
