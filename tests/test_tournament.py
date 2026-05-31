"""Tests for tournament ranking and promotion gates."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.artifacts.run_paths import resolve_run_paths
from src.artifacts.tournament.promotion import (
    promote_from_tournament,
    tournament_improves_incumbent,
)
from src.artifacts.tournament.ranking import (
    aggregate_pairwise_win_rates,
    build_leaderboard,
    evaluate_gates,
)
from src.artifacts.tournament.resolve import (
    run_context_for_agent,
    validate_agents_feature_compatible,
)
from src.artifacts.tournament.runner import run_match
from src.artifacts.tournament.types import (
    AgentEntry,
    LeaderboardRow,
    MatchOutcome,
    TournamentResult,
)
from src.cli import eval as eval_cli
from src.config import TrainConfig
from src.config.schema import PromotionTournamentConfig


def _candidate(agent_id: str, path: str = "/tmp/ckpt.pkl") -> AgentEntry:
    return AgentEntry(
        agent_id=agent_id,
        checkpoint_path=__import__("pathlib").Path(path),
        cfg=TrainConfig(),
        act_fn=lambda _obs: [],
    )


def test_build_leaderboard_aggregates_sniper_and_incumbent_rates() -> None:
    candidates = (_candidate("cand_a"),)
    outcomes = (
        MatchOutcome(
            match_id="m1",
            format_name="2p_vs_baseline",
            seed=0,
            agent_ids=("cand_a", "baseline:sniper"),
            rewards={"cand_a": 1.0, "baseline:sniper": -1.0},
            results={"cand_a": "win", "baseline:sniper": "loss"},
        ),
        MatchOutcome(
            match_id="m2",
            format_name="2p_vs_baseline",
            seed=1,
            agent_ids=("cand_a", "baseline:sniper"),
            rewards={"cand_a": -1.0, "baseline:sniper": 1.0},
            results={"cand_a": "loss", "baseline:sniper": "win"},
        ),
        MatchOutcome(
            match_id="m3",
            format_name="2p_head_to_head",
            seed=2,
            agent_ids=("cand_a", "incumbent"),
            rewards={"cand_a": 1.0, "incumbent": -1.0},
            results={"cand_a": "win", "incumbent": "loss"},
        ),
    )

    rows = build_leaderboard(
        candidates,
        outcomes,
        incumbent_id="incumbent",
        baseline_name="sniper",
    )

    assert len(rows) == 1
    assert rows[0].win_rate_vs_sniper == 0.5
    assert rows[0].win_rate_vs_incumbent == 1.0


def test_evaluate_gates_requires_incumbent_when_configured() -> None:
    row = build_leaderboard(
        (_candidate("cand_a"),),
        (
            MatchOutcome(
                match_id="m1",
                format_name="2p_vs_baseline",
                seed=0,
                agent_ids=("cand_a", "baseline:sniper"),
                rewards={"cand_a": 1.0, "baseline:sniper": -1.0},
                results={"cand_a": "win", "baseline:sniper": "loss"},
            ),
            MatchOutcome(
                match_id="m2",
                format_name="2p_vs_baseline",
                seed=1,
                agent_ids=("cand_a", "baseline:sniper"),
                rewards={"cand_a": -1.0, "baseline:sniper": 1.0},
                results={"cand_a": "loss", "baseline:sniper": "win"},
            ),
        ),
        incumbent_id="incumbent",
    )[0]
    gates = PromotionTournamentConfig(
        min_win_rate_vs_sniper=0.55,
        min_win_rate_vs_incumbent=0.51,
        require_head_to_head=True,
    )

    passed, reasons = evaluate_gates(row, gates, incumbent_present=True)

    assert passed is False
    assert "below_min_win_rate_vs_sniper" in reasons
    assert "missing_vs_incumbent" in reasons


def test_aggregate_pairwise_win_rates() -> None:
    outcomes = (
        MatchOutcome(
            match_id="m1",
            format_name="2p_head_to_head",
            seed=0,
            agent_ids=("a", "b"),
            rewards={"a": 1.0, "b": -1.0},
            results={"a": "win", "b": "loss"},
        ),
        MatchOutcome(
            match_id="m2",
            format_name="2p_head_to_head",
            seed=1,
            agent_ids=("a", "b"),
            rewards={"a": -1.0, "b": 1.0},
            results={"a": "loss", "b": "win"},
        ),
    )

    matrix = aggregate_pairwise_win_rates(outcomes)

    assert matrix["a"]["b"] == 0.5
    assert matrix["b"]["a"] == 0.5


def test_tournament_improves_incumbent_requires_higher_sniper_rate(
    tmp_path: Path,
) -> None:
    from src.artifacts.promotion import promoted_manifest_path

    campaign_dir = tmp_path / "outputs" / "campaigns" / "demo"
    manifest = promoted_manifest_path(campaign_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '{"tournament_win_rate_vs_sniper": 0.7}',
        encoding="utf-8",
    )
    row = LeaderboardRow(
        agent_id="cand",
        checkpoint_path="/tmp/ckpt.pkl",
        games_played=5,
        win_rate_vs_sniper=0.65,
        gates_passed=True,
    )

    improves, reason = tournament_improves_incumbent(row, campaign_dir=campaign_dir)

    assert improves is False
    assert reason == "incumbent_win_rate_vs_sniper_unchanged"


def test_run_context_for_agent_uses_checkpoint_run_dir(tmp_path: Path) -> None:
    cfg = TrainConfig()
    cfg.output.campaign = "demo"
    cfg.output.root = str(tmp_path / "outputs")
    run_dir = tmp_path / "outputs" / "campaigns" / "demo" / "runs" / "run_a"
    checkpoint = run_dir / "checkpoints" / "jax_ckpt_000010.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as file:
        pickle.dump({"params": {}, "config": cfg}, file)
    agent = AgentEntry(
        agent_id="run_a",
        checkpoint_path=checkpoint,
        cfg=cfg,
        act_fn=lambda _obs: [],
    )

    context = run_context_for_agent(agent, campaign="demo", output_root=str(tmp_path / "outputs"))

    assert context.run_id == "run_a"
    assert context.run_dir == run_dir


def test_run_context_for_agent_wandb_cache_paths_match_resolve_run_paths(
    tmp_path: Path,
) -> None:
    cfg = TrainConfig()
    output_root = tmp_path / "outputs"
    cfg.output.root = str(output_root)
    cfg.output.campaign = "demo"
    cfg.output.run_id = "run_a"
    run_dir = output_root / "campaigns" / "demo" / "runs" / "run_a"
    checkpoint = run_dir / "checkpoints" / "jax_ckpt_000010.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as file:
        pickle.dump({"params": {}, "config": cfg}, file)
    agent = AgentEntry(
        agent_id="run_a",
        checkpoint_path=checkpoint,
        cfg=cfg,
        act_fn=lambda _obs: [],
    )

    _, resolved_context = resolve_run_paths(cfg)
    agent_context = run_context_for_agent(
        agent, campaign="demo", output_root=str(output_root)
    )

    assert agent_context.wandb_dir == resolved_context.wandb_dir
    assert agent_context.wandb_artifact_dir == resolved_context.wandb_artifact_dir
    assert agent_context.wandb_data_dir == resolved_context.wandb_data_dir
    assert agent_context.wandb_artifact_dir == output_root / "cache" / "wandb-artifacts"
    assert agent_context.wandb_data_dir == output_root / "cache" / "wandb-data"


def test_validate_agents_feature_compatible_rejects_mismatch() -> None:
    left = _candidate("left")
    right = _candidate("right")
    with patch(
        "src.artifacts.tournament.resolve.feature_metadata_for_agent",
        side_effect=[
            {"schema_version": 5, "planet_feature_dim": 13, "edge_feature_dim": 12,
             "global_feature_dim": 46, "edge_k": 3, "encoder_backbone": "planet_graph"},
            {"schema_version": 5, "planet_feature_dim": 99, "edge_feature_dim": 12,
             "global_feature_dim": 46, "edge_k": 3, "encoder_backbone": "planet_graph"},
        ],
    ):
        with pytest.raises(ValueError, match="planet_feature_dim"):
            validate_agents_feature_compatible((left, right))


def test_run_match_smoke_with_mock_env() -> None:
    state = MagicMock()
    state.observation = {"player": 0, "planets": []}
    state.reward = 1.0
    state.status = "DONE"

    env = MagicMock()
    env.reset.return_value = [state, state]
    env.step.return_value = [state, state]

    with patch("kaggle_environments.make", return_value=env):
        outcome, returned_env, _timing = run_match(
            match_id="mock",
            format_name="2p_vs_baseline",
            seed=7,
            agent_ids=("a", "b"),
            agents=[lambda _obs: [], lambda _obs: []],
            max_steps=3,
        )

    assert outcome.results["a"] == "win"
    assert returned_env is env


def test_queue_tournament_job_accepts_tournament_only_reason(tmp_path: Path) -> None:
    from src.config import TrainConfig
    from src.jax.train.queue import queue_tournament_job_if_eligible

    cfg = TrainConfig()
    cfg.artifacts.promotion.strategy = "tournament"
    checkpoint_path = tmp_path / "jax_ckpt_000001.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    job_path = queue_tournament_job_if_eligible(
        cfg,
        update=1,
        checkpoint_path=checkpoint_path,
        queue_dir=tmp_path / "jobs",
        result_root=tmp_path / "evaluations",
        promotion_attempt_reason="tournament_only",
    )

    assert job_path is not None
    assert job_path.exists()


def test_eval_cli_dry_run(tmp_path: Path) -> None:
    cfg = TrainConfig()
    checkpoint = tmp_path / "jax_ckpt.pkl"
    with checkpoint.open("wb") as file:
        pickle.dump({"params": {}, "config": cfg}, file)

    with patch(
        "src.cli.eval.agent_from_checkpoint",
        return_value=AgentEntry(
            agent_id="cand",
            checkpoint_path=checkpoint,
            cfg=cfg,
            act_fn=lambda _obs: [],
        ),
    ):
        exit_code = eval_cli.main(
            ["tournament", "--checkpoint", str(checkpoint), "--dry-run"]
        )

    assert exit_code == 0


def test_promote_from_tournament_rejects_weaker_incumbent(tmp_path: Path) -> None:
    from src.artifacts.promotion import promoted_manifest_path

    cfg = TrainConfig()
    campaign_dir = tmp_path / "outputs" / "campaigns" / "demo"
    manifest = promoted_manifest_path(campaign_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '{"tournament_win_rate_vs_sniper": 0.9}',
        encoding="utf-8",
    )
    run_dir = campaign_dir / "runs" / "run_a"
    checkpoint = run_dir / "checkpoints" / "jax_ckpt.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as file:
        pickle.dump({"params": {}, "config": cfg}, file)

    context = run_context_for_agent(
        AgentEntry("run_a", checkpoint, cfg, act_fn=lambda _obs: []),
        campaign="demo",
        output_root=str(tmp_path / "outputs"),
    )
    row = LeaderboardRow(
        agent_id="run_a",
        checkpoint_path=str(checkpoint),
        games_played=5,
        win_rate_vs_sniper=0.8,
        gates_passed=True,
    )
    result = TournamentResult(
        tournament_id="t1",
        output_dir=tmp_path / "tournament",
        outcomes=(),
        leaderboard=(row,),
    )

    attempt = promote_from_tournament(cfg, context, row=row, tournament=result)

    assert attempt.promoted is False
    assert attempt.reason == "incumbent_win_rate_vs_sniper_unchanged"
