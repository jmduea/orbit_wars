from __future__ import annotations

import json
from pathlib import Path

from scripts.agent_context import build_context


def test_build_context_includes_preflight_and_roadmap() -> None:
    payload = build_context(limit_runs=0)
    assert payload["preflight"]["present"] is True
    assert "min_win_rate_delta" in payload["preflight"]["learning_signal"]
    assert payload["roadmap"]["present"] is True
    assert "docs" in payload


def test_build_context_recent_runs_from_fixture(tmp_path: Path, monkeypatch) -> None:
    indexes = tmp_path / "outputs" / "indexes"
    indexes.mkdir(parents=True)
    line = json.dumps({"campaign": "c1", "run_id": "r1"})
    (indexes / "runs.jsonl").write_text(line + "\n", encoding="utf-8")

    import scripts.agent_context as mod

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    payload = build_context(limit_runs=5)
    assert len(payload["recent_runs_index"]) == 1
