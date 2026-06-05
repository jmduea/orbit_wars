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
