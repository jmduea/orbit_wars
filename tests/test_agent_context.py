from __future__ import annotations

import json
from pathlib import Path

from scripts.agent_context import build_context


def test_build_context_includes_preflight_and_roadmap() -> None:
    payload = build_context(limit_runs=0)
    assert payload["preflight"]["present"] is True
    assert "min_win_rate_delta" in payload["preflight"]["learning_signal"]
    assert payload["preflight"]["gates"]["present"] is True
    assert "beat_noop" in payload["preflight"]["gates"]["gate_ids"]
    assert payload["resolved_config"]["print_command"]
    assert "single_gpu_note" in payload["gpu_contention"]
    assert "list_command" in payload["wandb_sweeps"]
    assert payload["roadmap"]["present"] is True
    assert "docs" in payload


def test_build_context_recent_runs_from_fixture(tmp_path: Path, monkeypatch) -> None:
    indexes = tmp_path / "outputs" / "indexes"
    indexes.mkdir(parents=True)
    run_dir = tmp_path / "outputs" / "campaigns" / "c1" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "campaign": "c1"}),
        encoding="utf-8",
    )
    (run_dir / "queue" / "optional_jobs").mkdir(parents=True)
    line = json.dumps({"campaign": "c1", "run_id": "r1", "run_dir": str(run_dir)})
    (indexes / "runs.jsonl").write_text(line + "\n", encoding="utf-8")

    import scripts.agent_context as mod

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_read_git_branch", lambda _root: {"present": True, "branch": "main"})
    payload = build_context(limit_runs=5)
    assert len(payload["recent_runs_index"]) == 1
    assert payload["latest_run_eval"]["present"] is True
    assert payload["latest_run_eval"]["run_id"] == "r1"


def test_build_context_wandb_sweeps_from_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.agent_context as mod

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_read_git_branch", lambda _root: {"present": True, "branch": "main"})

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 0
            stdout = json.dumps(
                {
                    "backend": "wandb",
                    "sweeps": [{"id": "abc", "name": "s1", "state": "RUNNING"}],
                }
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    payload = mod.build_context(limit_runs=0)
    assert payload["wandb_sweeps"]["present"] is True
    assert payload["wandb_sweeps"]["active_count"] == 1
