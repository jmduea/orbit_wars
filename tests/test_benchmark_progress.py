from __future__ import annotations

from src.cli import benchmark as benchmark_cli
from src.jax.benchmark_progress import (
    emit_benchmark_progress,
    total_updates_from_overrides,
)


def test_total_updates_from_overrides() -> None:
    assert total_updates_from_overrides(["training.total_updates=500"]) == 500
    assert total_updates_from_overrides(["model=foo"]) is None


def test_gate_dry_run_emits_stderr_start(capsys) -> None:
    assert benchmark_cli.main(["gate", "beat_noop", "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert "preflight gate 'beat_noop'" in captured.err
    assert "uv run ow train" in captured.err


def test_emit_benchmark_progress_writes_stderr(capsys) -> None:
    emit_benchmark_progress("hello progress")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hello progress" in captured.err
