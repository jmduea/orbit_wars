from pathlib import Path
from types import SimpleNamespace

from src.jax.train.queue import (
    checkpoint_replay_due,
    queue_tournament_job_if_eligible,
)


def test_checkpoint_replay_due_on_final_update() -> None:
    cfg = SimpleNamespace(
        artifacts=SimpleNamespace(
            replay=SimpleNamespace(enabled=True, every_n_checkpoints=1),
            checkpoint_every=10,
        ),
        training=SimpleNamespace(total_updates=5),
    )
    assert checkpoint_replay_due(cfg, 5) is True


def test_queue_tournament_job_skips_non_tournament_reasons(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        artifacts=SimpleNamespace(
            promotion=SimpleNamespace(strategy="metric"),
            tournament=SimpleNamespace(enabled=True),
        ),
        output=SimpleNamespace(campaign="c", run_id="r"),
    )
    job = queue_tournament_job_if_eligible(
        cfg,
        update=1,
        checkpoint_path=tmp_path / "ckpt.pkl",
        queue_dir=tmp_path / "queue",
        result_root=None,
        promotion_attempt_reason="metric_only",
    )
    assert job is None


def test_queue_tournament_job_writes_job_for_tournament_only(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        artifacts=SimpleNamespace(
            promotion=SimpleNamespace(strategy="tournament"),
            tournament=SimpleNamespace(enabled=True),
        ),
        output=SimpleNamespace(campaign="c", run_id="r"),
    )
    queue_dir = tmp_path / "queue"
    job = queue_tournament_job_if_eligible(
        cfg,
        update=3,
        checkpoint_path=tmp_path / "ckpt.pkl",
        queue_dir=queue_dir,
        result_root=None,
        promotion_attempt_reason="tournament_only",
    )
    assert job is not None
    assert job.exists()
