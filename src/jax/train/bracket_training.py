"""Training-time bracket qualifier hooks and weak_config classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts.pipeline import write_optional_job
from src.artifacts.tournament.bracket.lineage import qualifier_skip_for_checkpoint
from src.artifacts.tournament.bracket.state import (
    BracketEntry,
    BracketState,
    bracket_state_path,
    load_bracket_state,
    mark_qualifier_cleared,
    mark_weak_config,
    save_bracket_state,
    upsert_entry,
)
from src.config import TrainConfig


@dataclass(frozen=True, slots=True)
class BracketTrainingTick:
    phase: str
    weak_config: bool
    qualifier_eval_queued: bool
    events: tuple[dict[str, object], ...]


def bracket_training_enabled(cfg: TrainConfig) -> bool:
    return bool(getattr(cfg.artifacts, "bracket_training", None) and cfg.artifacts.bracket_training.enabled)


def bracket_training_tick(
    cfg: TrainConfig,
    *,
    update: int,
    total_env_steps: int,
    checkpoint_path: Path | None,
    queue_dir: Path,
    output_root: Path,
    result_root: Path | None = None,
) -> BracketTrainingTick:
    """Evaluate bracket phase, weak_config, and periodic qualifier eval queue."""

    bt = cfg.artifacts.bracket_training
    state_path = bracket_state_path(campaign=cfg.output.campaign, output_root=output_root)
    state = load_bracket_state(state_path)
    events: list[dict[str, object]] = []
    weak_config = False
    queued = False

    if checkpoint_path is not None and checkpoint_path.is_file():
        agent_id = f"u{update}"
        skip = qualifier_skip_for_checkpoint(
            checkpoint_path,
            campaign=cfg.output.campaign,
            output_root=output_root,
            bracket_state=state,
        )
        entry = BracketEntry(
            agent_id=agent_id,
            checkpoint_path=str(checkpoint_path),
            lineage_skip=skip,
            qualifier_cleared=skip,
        )
        upsert_entry(state, entry)
        if skip:
            state.phase = "main"
            events.append({"event": "bracket_lineage_skip", "agent_id": agent_id})

    if (
        not state.incumbent_crowned
        and not any(entry.qualifier_cleared for entry in state.entries.values())
        and total_env_steps >= bt.qualifier_max_env_steps
    ):
        mark_weak_config(state)
        weak_config = True
        events.append(
            {
                "event": "bracket_weak_config",
                "total_env_steps": total_env_steps,
                "budget": bt.qualifier_max_env_steps,
            }
        )

    if (
        checkpoint_path is not None
        and bt.qualifier_eval_interval_updates > 0
        and update % bt.qualifier_eval_interval_updates == 0
        and state.phase == "qualifier"
    ):
        write_optional_job(
            queue_dir,
            kind="qualifier_eval",
            update=update,
            checkpoint_path=checkpoint_path,
            payload={
                "campaign": cfg.output.campaign,
                "output_root": str(output_root),
                "agent_id": f"u{update}",
                "qualifier_mode": True,
                "qualifier_floors": "1.0",
            },
            result_root=result_root,
        )
        queued = True
        events.append({"event": "qualifier_eval_queued", "update": update})

    save_bracket_state(state_path, state)
    return BracketTrainingTick(
        phase=state.phase,
        weak_config=weak_config,
        qualifier_eval_queued=queued,
        events=tuple(events),
    )


def apply_qualifier_verdict_to_state(
    state: BracketState,
    *,
    agent_id: str,
    verdict: Any,
) -> BracketState:
    if verdict.cleared:
        mark_qualifier_cleared(
            state,
            agent_id=agent_id,
            crown_incumbent=bool(verdict.crown_incumbent),
        )
    return state
