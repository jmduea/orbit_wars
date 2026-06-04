"""Deterministic SSOT preflight W&B sweep shortlist (Gates 2–3 guardrails)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.jax.planet_flow_shortlist import (
    ShortlistRunInput,
    enrich_summary_from_run_dir,
    fetch_finished_sweep_runs,
    hydra_overrides_from_config,
    write_shortlist_report,
)
from src.jax.train.sweep_score import (
    SSOT_PREFLIGHT_SWEEP_SCORE_INELIGIBLE,
    ssot_preflight_learning_signal_thresholds,
    ssot_preflight_sweep_score,
)


def _summary_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def ssot_preflight_guardrail_reasons(
    *,
    win_rate_delta: float | None,
    approx_kl: float | None,
    entropy: float | None,
    min_win_rate_delta: float,
    max_approx_kl: float,
    min_entropy: float,
) -> list[str]:
    reasons: list[str] = []
    if win_rate_delta is None:
        reasons.append("missing win_rate_delta_10")
    elif win_rate_delta < min_win_rate_delta:
        reasons.append(
            f"win_rate_delta_10 {win_rate_delta:.4f} < {min_win_rate_delta:.4f}"
        )
    if approx_kl is None:
        reasons.append("missing approx_kl_window_mean")
    elif approx_kl > max_approx_kl:
        reasons.append(f"approx_kl_window_mean {approx_kl:.4f} > {max_approx_kl:.4f}")
    if entropy is None:
        reasons.append("missing entropy_window_mean")
    elif entropy < min_entropy:
        reasons.append(f"entropy_window_mean {entropy:.6f} < {min_entropy:.6f}")
    return reasons


def evaluate_ssot_shortlist_run(
    run: ShortlistRunInput,
    *,
    min_win_rate_delta: float | None = None,
    max_approx_kl: float | None = None,
    min_entropy: float | None = None,
) -> dict[str, object]:
    summary = enrich_summary_from_run_dir(run.summary, run.run_dir)
    train_overrides = hydra_overrides_from_config(run.config)
    win_rate_delta = _summary_float(summary, "win_rate_delta_10")
    approx_kl = _summary_float(summary, "approx_kl_window_mean")
    entropy = _summary_float(summary, "entropy_window_mean")
    min_delta, max_kl, min_ent = ssot_preflight_learning_signal_thresholds()
    if min_win_rate_delta is not None:
        min_delta = float(min_win_rate_delta)
    if max_approx_kl is not None:
        max_kl = float(max_approx_kl)
    if min_entropy is not None:
        min_ent = float(min_entropy)
    score = ssot_preflight_sweep_score(
        win_rate_delta=win_rate_delta,
        approx_kl=approx_kl,
        entropy=entropy,
        min_win_rate_delta=min_delta,
        max_approx_kl=max_kl,
        min_entropy=min_ent,
    )
    guardrail_reasons = ssot_preflight_guardrail_reasons(
        win_rate_delta=win_rate_delta,
        approx_kl=approx_kl,
        entropy=entropy,
        min_win_rate_delta=min_delta,
        max_approx_kl=max_kl,
        min_entropy=min_ent,
    )
    eligible = score != SSOT_PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    checkpoint_artifact = f"checkpoint-u{int(summary.get('update', 50) or 50)}"
    return {
        "run_id": run.run_id,
        "name": run.name,
        "train_overrides": list(train_overrides),
        "win_rate_delta_10": win_rate_delta,
        "approx_kl_window_mean": approx_kl,
        "entropy_window_mean": entropy,
        "ssot_preflight_sweep_score": score,
        "guardrail_reasons": guardrail_reasons,
        "eligible": eligible,
        "checkpoint_artifact": checkpoint_artifact,
    }


def rank_ssot_eligible_entries(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    eligible = [entry for entry in entries if entry.get("eligible")]

    def sort_key(entry: dict[str, object]) -> tuple[float, float, float]:
        score = float(entry.get("ssot_preflight_sweep_score") or float("-inf"))
        delta = float(entry.get("win_rate_delta_10") or float("-inf"))
        kl = float(entry.get("approx_kl_window_mean") or float("inf"))
        return (-score, -delta, kl)

    return sorted(eligible, key=sort_key)


def build_ssot_shortlist_report(
    runs: list[ShortlistRunInput],
    *,
    sweep_id: str,
    limit: int | None = None,
) -> dict[str, object]:
    min_delta, max_kl, min_ent = ssot_preflight_learning_signal_thresholds()
    evaluated = [evaluate_ssot_shortlist_run(run) for run in runs]
    eligible_ranked = rank_ssot_eligible_entries(evaluated)
    if limit is not None:
        eligible_ranked = eligible_ranked[: max(int(limit), 0)]
    audit = [entry for entry in evaluated if not entry.get("eligible")]
    winner = eligible_ranked[0] if eligible_ranked else None
    return {
        "sweep_id": sweep_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "guardrails": {
            "min_win_rate_delta_10": min_delta,
            "max_approx_kl_window_mean": max_kl,
            "min_entropy_window_mean": min_ent,
        },
        "eligible": eligible_ranked,
        "audit": audit,
        "winner": winner,
    }


__all__ = [
    "ShortlistRunInput",
    "build_ssot_shortlist_report",
    "evaluate_ssot_shortlist_run",
    "fetch_finished_sweep_runs",
    "write_shortlist_report",
]
