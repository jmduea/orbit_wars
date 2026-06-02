"""Deterministic Planet Flow W&B sweep shortlist (window-mean guardrails)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.jax.preflight_calibration import (
    WINDOW_UPDATES,
    read_jsonl_records,
    window_mean_from_metric_rows,
)
from src.jax.train.sweep_score import (
    PLANET_FLOW_MAX_APPROX_KL,
    PLANET_FLOW_MIN_ENTROPY,
    PLANET_FLOW_SWEEP_SCORE_INELIGIBLE,
    WinRateTrendTracker,
    planet_flow_sweep_eval,
)

PPO_OVERRIDE_KEYS: tuple[str, ...] = (
    "training.lr",
    "training.clip_coef",
    "training.ent_coef",
    "training.epochs",
    "training.vf_coef",
    "training.max_grad_norm",
    "training.update_chunk_rows",
)

SUMMARY_METRIC_KEYS: tuple[str, ...] = (
    "win_rate_delta_10",
    "approx_kl_window_mean",
    "entropy_window_mean",
    "planet_flow_sweep_score",
    "mean_active_launches_per_turn",
    "planet_flow_demanded_mass_sum",
    "planet_flow_emitted_launch_count",
    "planet_flow_unreachable_demand_rate",
    "env_steps_per_sec",
)


def _config_value(config: dict[str, Any], key: str) -> object | None:
    if key in config:
        return config[key]
    cur: object = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def hydra_overrides_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    overrides: list[str] = []
    for key in PPO_OVERRIDE_KEYS:
        value = _config_value(config, key)
        if value is None:
            continue
        overrides.append(f"{key}={value}")
    return tuple(overrides)


def _summary_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _metric_rows_from_jsonl(log_path: Path) -> list[dict[str, object]]:
    return [
        record
        for record in read_jsonl_records(log_path)
        if "overall_win_rate" in record and record.get("update") is not None
    ]


def _window_metrics_from_jsonl(
    log_path: Path,
    *,
    window: int = WINDOW_UPDATES,
) -> dict[str, float]:
    metric_rows = _metric_rows_from_jsonl(log_path)
    if len(metric_rows) < window:
        return {}
    out: dict[str, float] = {}
    approx_kl = window_mean_from_metric_rows(metric_rows, "approx_kl", last_n=window)
    entropy = window_mean_from_metric_rows(metric_rows, "entropy", last_n=window)
    if approx_kl is not None:
        out["approx_kl_window_mean"] = approx_kl
    if entropy is not None:
        out["entropy_window_mean"] = entropy
    if len(metric_rows) >= window * 2:
        trend = WinRateTrendTracker(window=window)
        for row in metric_rows:
            trend.observe(float(row["overall_win_rate"]))
        win_rate_delta = trend.win_rate_delta()
        if win_rate_delta is not None:
            out["win_rate_delta_10"] = win_rate_delta
    return out


def enrich_summary_from_run_dir(
    summary: dict[str, Any],
    run_dir: Path | None,
) -> dict[str, Any]:
    if run_dir is None:
        return summary
    logs = sorted((run_dir / "logs").glob("*_jax.jsonl"))
    if not logs:
        return summary
    merged = dict(summary)
    for key, value in _window_metrics_from_jsonl(logs[-1]).items():
        merged.setdefault(key, value)
    return merged


@dataclass(frozen=True, slots=True)
class ShortlistRunInput:
    run_id: str
    name: str
    summary: dict[str, Any]
    config: dict[str, Any]
    run_dir: Path | None = None


def evaluate_shortlist_run(
    run: ShortlistRunInput,
    *,
    max_kl: float = PLANET_FLOW_MAX_APPROX_KL,
    min_entropy: float = PLANET_FLOW_MIN_ENTROPY,
) -> dict[str, object]:
    summary = enrich_summary_from_run_dir(run.summary, run.run_dir)
    train_overrides = hydra_overrides_from_config(run.config)
    win_rate_delta = _summary_float(summary, "win_rate_delta_10")
    approx_kl = _summary_float(summary, "approx_kl_window_mean")
    entropy = _summary_float(summary, "entropy_window_mean")
    mean_launches = _summary_float(summary, "mean_active_launches_per_turn")
    demand_mass = _summary_float(summary, "planet_flow_demanded_mass_sum")
    emitted_launches = _summary_float(summary, "planet_flow_emitted_launch_count")
    unreachable_rate = _summary_float(summary, "planet_flow_unreachable_demand_rate")
    score, guardrail_reasons = planet_flow_sweep_eval(
        win_rate_delta=win_rate_delta,
        mean_active_launches_per_turn=mean_launches,
        planet_flow_demanded_mass_sum=demand_mass,
        planet_flow_emitted_launch_count=emitted_launches,
        entropy=entropy,
        approx_kl=approx_kl,
        planet_flow_unreachable_demand_rate=unreachable_rate,
        max_approx_kl=max_kl,
        min_entropy=min_entropy,
    )
    eligible = score != PLANET_FLOW_SWEEP_SCORE_INELIGIBLE
    entry: dict[str, object] = {
        "run_id": run.run_id,
        "name": run.name,
        "train_overrides": list(train_overrides),
        "win_rate_delta_10": win_rate_delta,
        "approx_kl_window_mean": approx_kl,
        "entropy_window_mean": entropy,
        "planet_flow_sweep_score": score,
        "mean_active_launches_per_turn": mean_launches,
        "planet_flow_emitted_launch_count": emitted_launches,
        "env_steps_per_sec": _summary_float(summary, "env_steps_per_sec"),
        "guardrail_reasons": guardrail_reasons,
        "eligible": eligible,
    }
    return entry


def rank_eligible_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    eligible = [entry for entry in entries if entry.get("eligible")]

    def sort_key(entry: dict[str, object]) -> tuple[float, float, float, float]:
        delta = float(entry.get("win_rate_delta_10") or float("-inf"))
        kl = float(entry.get("approx_kl_window_mean") or float("inf"))
        ent = float(entry.get("entropy_window_mean") or float("-inf"))
        launches = float(entry.get("mean_active_launches_per_turn") or 0.0)
        return (-delta, kl, -ent, -launches)

    return sorted(eligible, key=sort_key)


def build_shortlist_report(
    runs: list[ShortlistRunInput],
    *,
    sweep_id: str,
    max_kl: float = PLANET_FLOW_MAX_APPROX_KL,
    min_entropy: float = PLANET_FLOW_MIN_ENTROPY,
    limit: int | None = None,
) -> dict[str, object]:
    evaluated = [
        evaluate_shortlist_run(run, max_kl=max_kl, min_entropy=min_entropy)
        for run in runs
    ]
    eligible_ranked = rank_eligible_entries(evaluated)
    if limit is not None:
        eligible_ranked = eligible_ranked[: max(int(limit), 0)]
    audit = [entry for entry in evaluated if not entry.get("eligible")]
    winner = eligible_ranked[0] if eligible_ranked else None
    return {
        "sweep_id": sweep_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "guardrails": {
            "max_approx_kl_window_mean": max_kl,
            "min_entropy_window_mean": min_entropy,
        },
        "eligible": eligible_ranked,
        "audit": audit,
        "winner": winner,
    }


def fetch_finished_sweep_runs(
    *,
    entity: str,
    project: str,
    sweep_id: str,
) -> list[ShortlistRunInput]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}"
    runs = list(api.runs(path, filters={"sweep": sweep_id, "state": "finished"}))
    out: list[ShortlistRunInput] = []
    for run in runs:
        out.append(
            ShortlistRunInput(
                run_id=str(run.id),
                name=str(run.name),
                summary=dict(getattr(run, "summary", {}) or {}),
                config=dict(getattr(run, "config", {}) or {}),
            )
        )
    return out


def write_shortlist_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
