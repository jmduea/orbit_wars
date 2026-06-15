"""Deterministic preflight W&B sweep shortlist (Gates 2–3 guardrails)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.jax.planet_flow_shortlist import (
    ShortlistRunInput,
    enrich_summary_from_run_dir,
    fetch_finished_sweep_runs,
    hydra_overrides_from_config,
    write_shortlist_report,
)
from src.jax.train.sweep_score import (
    PREFLIGHT_SWEEP_SCORE_INELIGIBLE,
    preflight_learning_signal_thresholds,
    preflight_sweep_score,
)


def _summary_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def preflight_guardrail_reasons(
    *,
    win_rate_delta: float | None,
    win_rate_recovery_delta: float | None,
    approx_kl: float | None,
    entropy: float | None,
    entropy_retention_ratio: float | None,
    min_win_rate_delta: float,
    max_approx_kl: float,
    min_entropy: float,
    min_entropy_retention_ratio: float = 0.25,
) -> list[str]:
    reasons: list[str] = []
    if win_rate_delta is None and win_rate_recovery_delta is None:
        reasons.append("missing win_rate_delta_10")
    else:
        candidate_delta = max(
            float(win_rate_delta) if win_rate_delta is not None else float("-inf"),
            float(win_rate_recovery_delta)
            if win_rate_recovery_delta is not None
            else float("-inf"),
        )
        if candidate_delta < min_win_rate_delta:
            reasons.append(
                f"win_rate_trend {candidate_delta:.4f} < {min_win_rate_delta:.4f}"
            )
    if approx_kl is None:
        reasons.append("missing approx_kl_window_mean")
    elif approx_kl > max_approx_kl:
        reasons.append(f"approx_kl_window_mean {approx_kl:.4f} > {max_approx_kl:.4f}")
    if entropy is None:
        reasons.append("missing entropy_window_mean")
    elif entropy < min_entropy:
        reasons.append(f"entropy_window_mean {entropy:.6f} < {min_entropy:.6f}")
    if (
        entropy_retention_ratio is not None
        and entropy_retention_ratio < min_entropy_retention_ratio
    ):
        reasons.append(
            "entropy_retention_ratio_10 "
            f"{entropy_retention_ratio:.4f} < {min_entropy_retention_ratio:.4f}"
        )
    return reasons


def evaluate_preflight_shortlist_run(
    run: ShortlistRunInput,
    *,
    min_win_rate_delta: float | None = None,
    max_approx_kl: float | None = None,
    min_entropy: float | None = None,
) -> dict[str, object]:
    summary = enrich_summary_from_run_dir(run.summary, run.run_dir)
    train_overrides = hydra_overrides_from_config(run.config)
    win_rate_delta = _summary_float(summary, "win_rate_delta_10")
    win_rate_recovery_delta = _summary_float(summary, "win_rate_recovery_delta_10")
    approx_kl = _summary_float(summary, "approx_kl_window_mean")
    entropy = _summary_float(summary, "entropy_window_mean")
    entropy_retention_ratio = _summary_float(summary, "entropy_retention_ratio_10")
    min_delta, max_kl, min_ent = preflight_learning_signal_thresholds()
    if min_win_rate_delta is not None:
        min_delta = float(min_win_rate_delta)
    if max_approx_kl is not None:
        max_kl = float(max_approx_kl)
    if min_entropy is not None:
        min_ent = float(min_entropy)
    score = preflight_sweep_score(
        win_rate_delta=win_rate_delta,
        win_rate_recovery_delta=win_rate_recovery_delta,
        approx_kl=approx_kl,
        entropy=entropy,
        entropy_retention_ratio=entropy_retention_ratio,
        min_win_rate_delta=min_delta,
        max_approx_kl=max_kl,
        min_entropy=min_ent,
    )
    guardrail_reasons = preflight_guardrail_reasons(
        win_rate_delta=win_rate_delta,
        win_rate_recovery_delta=win_rate_recovery_delta,
        approx_kl=approx_kl,
        entropy=entropy,
        entropy_retention_ratio=entropy_retention_ratio,
        min_win_rate_delta=min_delta,
        max_approx_kl=max_kl,
        min_entropy=min_ent,
    )
    eligible = score != PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    checkpoint_artifact = f"checkpoint-u{int(summary.get('update', 50) or 50)}"
    return {
        "run_id": run.run_id,
        "name": run.name,
        "train_overrides": list(train_overrides),
        "win_rate_delta_10": win_rate_delta,
        "win_rate_recovery_delta_10": win_rate_recovery_delta,
        "approx_kl_window_mean": approx_kl,
        "entropy_window_mean": entropy,
        "entropy_retention_ratio_10": entropy_retention_ratio,
        "preflight_sweep_score": score,
        "guardrail_reasons": guardrail_reasons,
        "eligible": eligible,
        "checkpoint_artifact": checkpoint_artifact,
    }


def rank_preflight_eligible_entries(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    eligible = [entry for entry in entries if entry.get("eligible")]

    def sort_key(entry: dict[str, object]) -> tuple[float, float, float]:
        score = float(entry.get("preflight_sweep_score") or float("-inf"))
        delta = float(entry.get("win_rate_delta_10") or float("-inf"))
        kl = float(entry.get("approx_kl_window_mean") or float("inf"))
        return (-score, -delta, kl)

    return sorted(eligible, key=sort_key)


def build_preflight_shortlist_report(
    runs: list[ShortlistRunInput],
    *,
    sweep_id: str,
    limit: int | None = None,
) -> dict[str, object]:
    min_delta, max_kl, min_ent = preflight_learning_signal_thresholds()
    evaluated = [evaluate_preflight_shortlist_run(run) for run in runs]
    eligible_ranked = rank_preflight_eligible_entries(evaluated)
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
            "min_entropy_retention_ratio_10": 0.25,
        },
        "eligible": eligible_ranked,
        "audit": audit,
        "winner": winner,
    }


__all__ = [
    "ShortlistRunInput",
    "build_preflight_shortlist_report",
    "evaluate_preflight_shortlist_run",
    "fetch_finished_sweep_runs",
    "write_shortlist_report",
]
