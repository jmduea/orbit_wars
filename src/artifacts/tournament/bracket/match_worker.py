"""Process bracket_match optional jobs: 2p head-to-head and μ/σ update."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.artifacts.tournament.bracket.scheduler import (
    apply_head_to_head_outcome,
    outcome_from_rewards,
)
from src.artifacts.tournament.bracket.state import (
    bracket_state_path,
    load_bracket_state,
    save_bracket_state,
)
from src.artifacts.tournament.resolve import agent_from_checkpoint, load_train_config_from_checkpoint
from src.artifacts.tournament.runner import run_match

REPO_ROOT = Path(__file__).resolve().parents[4]


def run_bracket_match_job(
    job: dict[str, object],
    *,
    result_dir: Path,
) -> dict[str, Any]:
    """Run one 2p bracket pairing and update campaign bracket ratings."""

    agent_a = str(job["agent_a"])
    agent_b = str(job["agent_b"])
    checkpoint_a = Path(str(job.get("checkpoint_path_a", job["checkpoint_path"])))
    checkpoint_b = Path(str(job["checkpoint_path_b"]))
    campaign = str(job.get("campaign", "default"))
    output_root = Path(str(job.get("output_root", REPO_ROOT / "outputs")))
    update = int(job.get("update", 0))

    cfg = load_train_config_from_checkpoint(checkpoint_a)
    tournament_cfg = cfg.artifacts.tournament
    seed = int(job.get("seed", int(getattr(cfg, "seed", 42)) + update))

    entry_a = agent_from_checkpoint(checkpoint_a, agent_id=agent_a)
    entry_b = agent_from_checkpoint(checkpoint_b, agent_id=agent_b)

    outcome, _, timing = run_match(
        match_id=f"bracket_{agent_a}_vs_{agent_b}_u{update:06d}",
        format_name="2p_head_to_head",
        seed=seed,
        agent_ids=(agent_a, agent_b),
        agents=(entry_a.act_fn, entry_b.act_fn),
        max_steps=int(job.get("episode_steps", getattr(cfg.artifacts.replay, "max_steps", 500))),
        per_step_seconds=float(
            job.get("per_step_seconds", tournament_cfg.per_step_seconds)
        ),
        overage_budget_seconds=float(
            job.get("overage_budget_seconds", tournament_cfg.overage_budget_seconds)
        ),
    )

    label = outcome_from_rewards(agent_a, agent_b, outcome.rewards)
    state_path = bracket_state_path(campaign=campaign, output_root=output_root)
    state = load_bracket_state(state_path)
    apply_head_to_head_outcome(state, agent_a=agent_a, agent_b=agent_b, outcome=label)
    save_bracket_state(state_path, state)

    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "outcome": label,
        "rewards": dict(outcome.rewards),
        "bracket_state_path": str(state_path),
        "timing": timing,
        "result_dir": str(result_dir),
    }
