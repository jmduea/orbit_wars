"""Tests for qualifier_eval artifact worker handler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.artifacts.pipeline import load_pending_optional_jobs, write_optional_job
from src.artifacts.tournament.bracket.qualifier import QualifierVerdict
from src.artifacts.tournament.bracket.state import load_bracket_state, save_bracket_state, BracketState, BracketEntry
from src.artifacts.tournament.unified.reporting import UnifiedLadderVerdict, UnifiedStageResult
from src.artifacts.tournament.unified.scoring import UnifiedOpponentScore


def test_qualifier_eval_worker_skips_ladder_when_docker_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import run_artifact_worker

    checkpoint_path = tmp_path / "jax_ckpt_000010.pkl"
    checkpoint_path.write_bytes(b"checkpoint")
    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="qualifier_eval",
        update=10,
        checkpoint_path=checkpoint_path,
        payload={"campaign": "c", "output_root": str(tmp_path)},
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]
    ladder_called: list[object] = []

    def fake_ladder(*args: object, **kwargs: object) -> UnifiedLadderVerdict:
        ladder_called.append(1)
        raise AssertionError("ladder should not run")

    monkeypatch.setattr(
        "src.artifacts.qualifier_eval.run_submit_valid_docker_gate",
        lambda **kwargs: {"validation_ok": False},
    )
    monkeypatch.setattr("src.artifacts.qualifier_eval.run_unified_ladder", fake_ladder)

    run_artifact_worker._run_qualifier_eval_job(job)
    assert ladder_called == []
    status = json.loads(job_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["validation_ok"] is False
    manifest = json.loads(Path(status["result_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["ladder_skipped"] is True


def _write_fake_checkpoint(tmp_path: Path) -> Path:
    import pickle

    from src.config import TrainConfig

    ckpt = tmp_path / "ckpt.pkl"
    cfg = TrainConfig()
    cfg.output.campaign = "c"
    cfg.output.root = str(tmp_path)
    payload = {"config": cfg, "params": {}}
    ckpt.write_bytes(pickle.dumps(payload))
    return ckpt


def test_qualifier_eval_worker_updates_bracket_state(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import run_artifact_worker

    checkpoint_path = _write_fake_checkpoint(tmp_path)
    state_path = tmp_path / "campaigns" / "c" / "bracket" / "state.json"
    save_bracket_state(
        state_path,
        BracketState(entries={"u10": BracketEntry(agent_id="u10", checkpoint_path=str(checkpoint_path))}),
    )

    job_path = write_optional_job(
        tmp_path / "jobs",
        kind="qualifier_eval",
        update=10,
        checkpoint_path=checkpoint_path,
        payload={
            "campaign": "c",
            "output_root": str(tmp_path),
            "agent_id": "u10",
        },
        result_root=tmp_path / "evaluations",
    )
    job = load_pending_optional_jobs(tmp_path / "jobs")[0]

    monkeypatch.setattr(
        "src.artifacts.qualifier_eval.run_submit_valid_docker_gate",
        lambda **kwargs: {"validation_ok": True},
    )
    monkeypatch.setattr(
        "src.artifacts.qualifier_eval.load_train_config_from_checkpoint",
        lambda path: __import__("src.config", fromlist=["TrainConfig"]).TrainConfig(),
    )
    monkeypatch.setattr(
        "src.artifacts.qualifier_eval.load_unified_tournament_spec",
        lambda *args, **kwargs: MagicMock(),
    )

    noop_row = UnifiedOpponentScore(
        opponent="noop",
        win_rate_2p=1.0,
        win_rate_4p=1.0,
        combined=1.0,
        passed=True,
    )
    random_row = UnifiedOpponentScore(
        opponent="random",
        win_rate_2p=1.0,
        win_rate_4p=1.0,
        combined=1.0,
        passed=True,
    )
    sniper_row = UnifiedOpponentScore(
        opponent="nearest_sniper",
        win_rate_2p=1.0,
        win_rate_4p=1.0,
        combined=1.0,
        passed=True,
    )
    verdict = UnifiedLadderVerdict(
        passed=True,
        reason="qualifier_cleared",
        stages=(
            UnifiedStageResult(
                name="stage1",
                passed=True,
                opponents=(noop_row, random_row),
            ),
            UnifiedStageResult(
                name="sniper",
                passed=True,
                opponents=(sniper_row,),
            ),
        ),
        challenger_checkpoint=str(checkpoint_path),
    )
    monkeypatch.setattr(
        "src.artifacts.qualifier_eval.run_unified_ladder",
        lambda *args, **kwargs: verdict,
    )

    run_artifact_worker._run_qualifier_eval_job(job)
    state = load_bracket_state(state_path)
    assert state.entries["u10"].qualifier_cleared is True
    assert state.phase == "main"
