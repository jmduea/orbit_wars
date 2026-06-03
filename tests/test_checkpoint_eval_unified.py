from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.artifacts.checkpoint_eval import run_checkpoint_eval_job
from src.artifacts.tournament.unified.reporting import (
    UnifiedLadderVerdict,
    UnifiedStageResult,
)


@patch("src.artifacts.checkpoint_eval.run_docker_validation_subprocess")
@patch("src.artifacts.tournament.worker.run_unified_ladder")
@patch("src.artifacts.tournament.worker.load_train_config_from_checkpoint")
@patch("src.artifacts.tournament.worker.agent_from_checkpoint")
def test_checkpoint_eval_unified_pass_promotes(
    mock_agent,
    mock_cfg,
    mock_ladder,
    mock_docker,
    tmp_path: Path,
) -> None:
    from src.artifacts.tournament.types import AgentEntry
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.artifacts.unified_tournament.enabled = True
    cfg.artifacts.promotion.strategy = "hybrid"
    mock_cfg.return_value = cfg
    mock_agent.return_value = AgentEntry(
        agent_id="cand",
        checkpoint_path=tmp_path / "ckpt.pkl",
        cfg=cfg,
        act_fn=lambda _obs: [],
    )
    mock_docker.return_value = {"ok": True}
    mock_ladder.return_value = UnifiedLadderVerdict(
        passed=True,
        reason="pass",
        stages=(UnifiedStageResult(name="stage2_incumbent", passed=True),),
        challenger_checkpoint=str(tmp_path / "ckpt.pkl"),
        incumbent_swap=True,
    )

    with patch(
        "src.artifacts.tournament.worker.promote_from_unified_ladder"
    ) as mock_promote:
        from src.artifacts.promotion import PromotionAttempt

        mock_promote.return_value = PromotionAttempt(
            promoted=True,
            reason="unified_ladder_promoted",
            metric_name="unified_combined_noop",
            metric_value=0.8,
        )
        result = run_checkpoint_eval_job(
            {
                "checkpoint_path": str(tmp_path / "ckpt.pkl"),
                "update": 10,
                "campaign": "test",
            },
            result_dir=tmp_path / "result",
        )

    assert result["promoted"] is True
    assert result["validation_ok"] is True


@patch("src.artifacts.checkpoint_eval.run_docker_validation_subprocess")
@patch("src.artifacts.tournament.worker.run_unified_ladder")
@patch("src.artifacts.tournament.worker.load_train_config_from_checkpoint")
@patch("src.artifacts.tournament.worker.agent_from_checkpoint")
def test_prerequisite_fail_does_not_promote(
    mock_agent,
    mock_cfg,
    mock_ladder,
    mock_docker,
    tmp_path: Path,
) -> None:
    from src.artifacts.tournament.types import AgentEntry
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.artifacts.unified_tournament.enabled = True
    mock_cfg.return_value = cfg
    mock_agent.return_value = AgentEntry(
        agent_id="cand",
        checkpoint_path=tmp_path / "ckpt.pkl",
        cfg=cfg,
        act_fn=lambda _obs: [],
    )
    mock_docker.return_value = {"ok": True}
    mock_ladder.return_value = UnifiedLadderVerdict(
        passed=False,
        reason="failed_prerequisite_random",
        stages=(UnifiedStageResult(name="stage1_prerequisites", passed=False),),
        challenger_checkpoint=str(tmp_path / "ckpt.pkl"),
    )

    result = run_checkpoint_eval_job(
        {"checkpoint_path": str(tmp_path / "ckpt.pkl"), "update": 5},
        result_dir=tmp_path / "result",
    )
    assert result["promoted"] is False
