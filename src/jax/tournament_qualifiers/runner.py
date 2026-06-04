"""Training-loop hooks for SSOT JAX tournament qualifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from src.jax.tournament_qualifiers.promotion import (
    STAGE_NAMES,
    LegWinSummary,
    evaluate_stage_promotion,
    opponent_family_probs_for_stage,
)


@dataclass(frozen=True, slots=True)
class SsotQualifierTick:
    phase: str
    qualifier_stage: int
    weak_config: bool
    promotion_event: dict[str, object] | None
    events: tuple[dict[str, object], ...]
    leg_summaries: tuple[LegWinSummary, ...]


def ssot_pipeline_enabled(cfg: TrainConfig) -> bool:
    ssot = getattr(cfg.artifacts, "ssot_pipeline", None)
    return bool(ssot and ssot.enabled)


def _ssot_config(cfg: TrainConfig) -> Any:
    return cfg.artifacts.ssot_pipeline


def evaluate_qualifier_legs(
    *,
    leg_wins: dict[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    """Aggregate wins per opponent leg.

    When ``qualifier_games_per_seed`` is 0 (default), callers supply ``leg_wins``
    from JAX eval; until wired, ticks pass empty counts and promotion does not advance.
    """

    return dict(leg_wins or {})


def ssot_qualifier_tick(
    cfg: TrainConfig,
    *,
    update: int,
    total_env_steps: int,
    checkpoint_path: Path | None,
    output_root: Path,
) -> SsotQualifierTick:
    """Checkpoint-tick qualifier promotion, weak_config budget, and bracket state."""

    ssot = _ssot_config(cfg)
    state_path = bracket_state_path(campaign=cfg.output.campaign, output_root=output_root)
    state = load_bracket_state(state_path)
    stage = max(1, int(getattr(state, "ssot_qualifier_stage", 1) or 1))
    events: list[dict[str, object]] = []
    weak_config = False
    promotion_event: dict[str, object] | None = None
    leg_summaries: tuple[LegWinSummary, ...] = ()

    if checkpoint_path is not None and checkpoint_path.is_file():
        agent_id = f"u{update}"
        upsert_entry(
            state,
            BracketEntry(
                agent_id=agent_id,
                checkpoint_path=str(checkpoint_path),
            ),
        )

    if (
        stage < 4
        and not any(entry.qualifier_cleared for entry in state.entries.values())
        and total_env_steps >= int(ssot.qualifier_max_env_steps)
    ):
        mark_weak_config(state)
        weak_config = True
        events.append(
            {
                "event": "ssot_weak_config",
                "update": update,
                "total_env_steps": total_env_steps,
            }
        )

    interval = max(1, int(ssot.qualifier_eval_interval_updates))
    should_eval = (
        checkpoint_path is not None
        and checkpoint_path.is_file()
        and update > 0
        and update % interval == 0
        and not weak_config
        and stage < 4
    )
    if should_eval:
        leg_wins = evaluate_qualifier_legs()
        verdict = evaluate_stage_promotion(stage=stage, leg_wins=leg_wins)
        leg_summaries = verdict.leg_summaries
        events.append(
            {
                "event": "ssot_qualifier_eval",
                "update": update,
                "stage": stage,
                "promoted": verdict.promoted,
                "fail_reason": verdict.fail_reason,
            }
        )
        if verdict.promoted:
            state.ssot_qualifier_stage = verdict.next_stage
            promotion_event = {
                "event": "ssot_qualifier_stage_promotion",
                "update": update,
                "from_stage": stage,
                "to_stage": verdict.next_stage,
                "stage_name": STAGE_NAMES.get(verdict.next_stage, "unknown"),
            }
            events.append(promotion_event)
            if verdict.enter_main_bracket:
                for entry in state.entries.values():
                    if entry.checkpoint_path == str(checkpoint_path):
                        mark_qualifier_cleared(state, agent_id=entry.agent_id)
                state.phase = "main"
                events.append(
                    {
                        "event": "ssot_main_bracket_entry",
                        "update": update,
                    }
                )
            stage = verdict.next_stage

    save_bracket_state(state_path, state)
    phase = state.phase if weak_config else STAGE_NAMES.get(stage, "qualifier")
    return SsotQualifierTick(
        phase=str(phase),
        qualifier_stage=stage,
        weak_config=weak_config,
        promotion_event=promotion_event,
        events=tuple(events),
        leg_summaries=leg_summaries,
    )


def ssot_qualifier_telemetry(tick: SsotQualifierTick) -> dict[str, object]:
    record: dict[str, object] = {
        "ssot_qualifier_stage": tick.qualifier_stage,
        "ssot_qualifier_phase": tick.phase,
        "weak_config": tick.weak_config,
    }
    probs = opponent_family_probs_for_stage(tick.qualifier_stage)
    from src.opponents.constants import OPPONENT_FAMILY_NAMES

    for name, prob in zip(OPPONENT_FAMILY_NAMES, probs, strict=True):
        record[f"ssot_rollout_family_prob_{name}"] = prob
    for summary in tick.leg_summaries:
        if summary.win_rate is not None:
            record[f"ssot_qualifier_win_rate_{summary.opponent}"] = summary.win_rate
    return record
