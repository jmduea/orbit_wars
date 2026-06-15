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
from src.artifacts.tournament.runner import challenger_won_2p, run_match
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


def test_challenger_won_2p_rejects_mutual_positive_rewards() -> None:
    tie = MatchOutcome(
        match_id="tie",
        format_name="2p_vs_baseline",
        seed=0,
        agent_ids=("cand_a", "baseline:noop"),
        rewards={"cand_a": 1.0, "baseline:noop": 1.0},
        results={"cand_a": "win", "baseline:noop": "win"},
    )

    assert challenger_won_2p(tie, "cand_a") is False
    assert challenger_won_2p(tie, "baseline:noop") is False

    rows = build_leaderboard(
        (_candidate("cand_a"),),
        (tie,),
        incumbent_id=None,
        baseline_name="noop",
    )
    assert rows[0].win_rate_vs_sniper == 0.0


def test_aggregate_pairwise_win_rates_ignores_ties() -> None:
    outcomes = (
        MatchOutcome(
            match_id="tie",
            format_name="2p_head_to_head",
            seed=0,
            agent_ids=("a", "b"),
            rewards={"a": 1.0, "b": 1.0},
            results={"a": "win", "b": "win"},
        ),
    )

    matrix = aggregate_pairwise_win_rates(outcomes)

    assert matrix["a"]["b"] == 0.0
    assert matrix["b"]["a"] == 0.0


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

    context = run_context_for_agent(
        agent, campaign="demo", output_root=str(tmp_path / "outputs")
    )

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
            {
                "schema_version": 5,
                "planet_feature_dim": 13,
                "edge_feature_dim": 12,
                "global_feature_dim": 46,
                "edge_k": 3,
                "encoder_backbone": "planet_graph",
            },
            {
                "schema_version": 5,
                "planet_feature_dim": 99,
                "edge_feature_dim": 12,
                "global_feature_dim": 46,
                "edge_k": 3,
                "encoder_backbone": "planet_graph",
            },
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


def test_eval_submit_dry_run_with_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "submission.tar.gz"
    package.write_bytes(b"fake")

    with patch(
        "src.cli.eval.submit_competition_package",
        return_value=MagicMock(returncode=0),
    ) as submit_mock:
        exit_code = eval_cli.main(
            [
                "submit",
                "--package",
                str(package),
                "--dry-run",
                "-m",
                "smoke",
            ]
        )

    assert exit_code == 0
    submit_mock.assert_called_once()
    call_kwargs = submit_mock.call_args.kwargs
    assert call_kwargs["dry_run"] is True
    assert call_kwargs["competition"] == "orbit-wars"


def test_eval_worker_requires_run_or_queue(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Provide --run or --queue-dir"):
        eval_cli.main(["worker"])


def test_eval_worker_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    queue_dir = run_dir / "queue" / "optional_jobs"
    queue_dir.mkdir(parents=True)

    with patch("src.cli.eval.run_optional_job_worker", return_value=0) as worker_mock:
        exit_code = eval_cli.main(["worker", "--run", str(run_dir)])

    assert exit_code == 0
    worker_mock.assert_called_once()
    assert worker_mock.call_args.kwargs["once"] is True
    assert worker_mock.call_args.args[0] == queue_dir


def test_eval_package_skips_docker_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "jax_ckpt.pkl"
    checkpoint.write_bytes(b"unused")
    output_dir = tmp_path / "out"

    with patch(
        "src.cli.eval.package_checkpoint_submission",
        return_value=output_dir / "submission.tar.gz",
    ) as package_mock:
        exit_code = eval_cli.main(
            [
                "package",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert exit_code == 0
    package_mock.assert_called_once()
    assert package_mock.call_args.kwargs["validate_docker"] is False
    assert "docker_validation=skipped" in capsys.readouterr().err


def test_eval_package_help_mentions_validate_docker_for_submit_valid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = eval_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["package", "--help"])
    help_text = capsys.readouterr().out.replace("\n", " ").lower()
    assert "validate-docker" in help_text
    assert "submit" in help_text and "valid proof" in help_text


def test_eval_tournament_help_classifies_write_replays_as_inspect_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = eval_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tournament", "--help"])
    help_text = capsys.readouterr().out
    assert "--write-replays" in help_text
    assert "inspect" in help_text.lower() or "not submit-valid" in help_text.lower()


def test_eval_submit_without_validate_docker_prints_packaging_only_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "jax_ckpt.pkl"
    checkpoint.write_bytes(b"unused")
    output_dir = tmp_path / "out"
    package_path = output_dir / "submission.tar.gz"

    with (
        patch(
            "src.cli.eval.package_checkpoint_submission",
            return_value=package_path,
        ),
        patch("src.cli.eval.submit_competition_package"),
    ):
        exit_code = eval_cli.main(
            [
                "submit",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

    assert exit_code == 0
    err = capsys.readouterr().err
    assert "docker_validation=skipped" in err
    assert "packaging only" in err
