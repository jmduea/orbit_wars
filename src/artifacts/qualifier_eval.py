"""Composite qualifier eval: Docker validation then unified qualifier ladder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.artifacts.run_paths import atomic_write_json
from src.artifacts.submit_valid_funnel import (
    docker_gate_passed,
    run_submit_valid_docker_gate,
)
from src.artifacts.tournament.bracket.qualifier import evaluate_qualifier_scores
from src.artifacts.tournament.bracket.scheduler import queue_round_robin_matches
from src.artifacts.tournament.bracket.state import (
    BracketEntry,
    bracket_state_path,
    load_bracket_state,
    save_bracket_state,
    upsert_entry,
)
from src.artifacts.tournament.resolve import load_train_config_from_checkpoint
from src.artifacts.tournament.unified.ladder import run_unified_ladder
from src.artifacts.tournament.unified.reporting import UnifiedLadderVerdict
from src.artifacts.tournament.unified.spec import load_unified_tournament_spec
from src.jax.preflight_calibration import default_calibration_json_path
from src.jax.train.bracket_training import apply_qualifier_verdict_to_state

REPO_ROOT = Path(__file__).resolve().parents[2]


def _opponent_rows_from_verdict(verdict: UnifiedLadderVerdict) -> tuple[Any, ...]:
    rows: list[Any] = []
    for stage in verdict.stages:
        rows.extend(stage.opponents)
    return tuple(rows)


def run_qualifier_eval_job(
    job: dict[str, object],
    *,
    result_dir: Path,
    queue_dir: Path | None = None,
) -> dict[str, Any]:
    """Run docker validation, qualifier-mode unified ladder, and bracket state update."""

    checkpoint_path = Path(str(job["checkpoint_path"]))
    update = int(job["update"])
    campaign = str(job.get("campaign", "default"))
    output_root = Path(str(job.get("output_root", REPO_ROOT / "outputs")))

    docker_output_dir = result_dir / "docker_validation"
    docker_manifest = run_submit_valid_docker_gate(
        checkpoint_path=checkpoint_path,
        output_dir=docker_output_dir,
        repo_root=REPO_ROOT,
        docker_image=str(
            job.get("docker_image", "gcr.io/kaggle-images/python-simulations")
        ),
        seed=int(job.get("seed", 42)),
        player_count=str(job.get("player_count", "both")),
        per_step_seconds=float(job.get("per_step_seconds", 1.0)),
        overage_budget_seconds=float(job.get("overage_budget_seconds", 60.0)),
        episode_steps=int(job.get("episode_steps", 500)),
    )
    atomic_write_json(result_dir / "docker_manifest.json", docker_manifest)
    validation_ok = docker_gate_passed(docker_manifest)
    if not validation_ok:
        return {
            "validation_ok": False,
            "qualifier_cleared": False,
            "docker_manifest_path": str(result_dir / "docker_manifest.json"),
            "ladder_skipped": True,
            "ladder_skipped_reason": "docker_validation_failed",
        }

    cfg = load_train_config_from_checkpoint(checkpoint_path)
    spec = load_unified_tournament_spec(
        default_calibration_json_path(REPO_ROOT),
        hydra=cfg.artifacts.unified_tournament,
    )
    ladder_dir = result_dir / "qualifier_ladder"
    verdict = run_unified_ladder(
        checkpoint_path,
        spec,
        ladder_dir,
        campaign=campaign,
        output_root=output_root,
        qualifier_mode=True,
    )
    atomic_write_json(result_dir / "unified_verdict.json", verdict.to_dict())

    state_path = bracket_state_path(campaign=campaign, output_root=output_root)
    state = load_bracket_state(state_path)
    agent_id = str(job.get("agent_id", f"u{update}"))
    upsert_entry(
        state,
        BracketEntry(agent_id=agent_id, checkpoint_path=str(checkpoint_path)),
    )
    qualifier_verdict = evaluate_qualifier_scores(
        _opponent_rows_from_verdict(verdict),
        incumbent_crowned=state.incumbent_crowned,
    )
    apply_qualifier_verdict_to_state(
        state,
        agent_id=agent_id,
        verdict=qualifier_verdict,
    )
    save_bracket_state(state_path, state)

    round_robin_jobs = 0
    if queue_dir is not None and qualifier_verdict.cleared and state.phase == "main":
        jobs = queue_round_robin_matches(
            queue_dir,
            state=state,
            update=update,
            result_root=Path(str(job["result_root"])) if job.get("result_root") else None,
            campaign=campaign,
            output_root=output_root,
        )
        round_robin_jobs = len(jobs)
    save_bracket_state(state_path, state)

    return {
        "validation_ok": True,
        "qualifier_cleared": qualifier_verdict.cleared,
        "qualifier_fail_reason": qualifier_verdict.fail_reason,
        "crown_incumbent": qualifier_verdict.crown_incumbent,
        "ladder_passed": verdict.passed,
        "ladder_reason": verdict.reason,
        "bracket_state_path": str(state_path),
        "bracket_phase": state.phase,
        "round_robin_jobs_queued": round_robin_jobs,
    }
