"""Extract rollout phase timing breakdown from training JSONL (opt-in telemetry)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from src.benchmark.jsonl_window import (
    ThroughputWindow,
    default_throughput_window,
    record_float,
    record_update,
    resolve_log_path_from_input,
)
from src.jax.preflight import read_jsonl_records

PHASE_NAMES: tuple[str, ...] = (
    "policy",
    "opponent",
    "env_step",
    "reset",
    "post_step",
)

OPPONENT_DETAIL_NAMES: tuple[str, ...] = (
    "opponent_sample",
    "opponent_encode",
)

PHASE_SECOND_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_seconds" for name in PHASE_NAMES
)
PHASE_FRACTION_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_fraction" for name in PHASE_NAMES
)
OPPONENT_DETAIL_SECOND_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_seconds" for name in OPPONENT_DETAIL_NAMES
)
OPPONENT_DETAIL_FRACTION_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_fraction" for name in OPPONENT_DETAIL_NAMES
)
MEASURED_TOTAL_KEY = "rollout_phase_measured_total_seconds"


def _has_phase_timing(record: Mapping[str, object]) -> bool:
    return record_float(record, PHASE_SECOND_KEYS[0]) is not None


def extract_rollout_phase_breakdown_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    window: ThroughputWindow | None = None,
) -> dict[str, object]:
    """Aggregate per-phase rollout seconds and fractions from JSONL rows."""

    resolved = window or default_throughput_window()
    selected: list[Mapping[str, object]] = []
    for record in records:
        update = record_update(record)
        if update is None or not resolved.includes(update):
            continue
        if not _has_phase_timing(record):
            continue
        selected.append(record)

    if not selected:
        raise ValueError(
            "no rollout phase timing rows in window "
            f"updates {resolved.first_update}–{resolved.max_measured_update}. "
            "Run `ow benchmark rollout-phase-profile` (integration worktree), "
            "then `ow benchmark rollout-phase-breakdown` on the output JSONL."
        )

    phase_seconds: dict[str, list[float]] = {name: [] for name in PHASE_NAMES}
    phase_fractions: dict[str, list[float]] = {name: [] for name in PHASE_NAMES}
    opponent_detail_seconds: dict[str, list[float]] = {
        name: [] for name in OPPONENT_DETAIL_NAMES
    }
    opponent_detail_fractions: dict[str, list[float]] = {
        name: [] for name in OPPONENT_DETAIL_NAMES
    }
    measured_totals: list[float] = []
    rollout_seconds: list[float] = []

    for record in selected:
        for name, sec_key, frac_key in zip(
            PHASE_NAMES, PHASE_SECOND_KEYS, PHASE_FRACTION_KEYS, strict=True
        ):
            sec = record_float(record, sec_key)
            frac = record_float(record, frac_key)
            if sec is not None:
                phase_seconds[name].append(sec)
            if frac is not None:
                phase_fractions[name].append(frac)
        for name, sec_key, frac_key in zip(
            OPPONENT_DETAIL_NAMES,
            OPPONENT_DETAIL_SECOND_KEYS,
            OPPONENT_DETAIL_FRACTION_KEYS,
            strict=True,
        ):
            sec = record_float(record, sec_key)
            frac = record_float(record, frac_key)
            if sec is not None:
                opponent_detail_seconds[name].append(sec)
            if frac is not None:
                opponent_detail_fractions[name].append(frac)
        measured = record_float(record, MEASURED_TOTAL_KEY)
        if measured is not None:
            measured_totals.append(measured)
        rollout_s = record_float(record, "rollout_seconds")
        if rollout_s is not None:
            rollout_seconds.append(rollout_s)

    def _mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    phases_payload: dict[str, object] = {}
    for name in PHASE_NAMES:
        phases_payload[name] = {
            "seconds_mean": _mean(phase_seconds[name]),
            "fraction_mean": _mean(phase_fractions[name]),
        }
    opponent_details: dict[str, object] = {}
    for name in OPPONENT_DETAIL_NAMES:
        if opponent_detail_seconds[name]:
            opponent_details[name] = {
                "seconds_mean": _mean(opponent_detail_seconds[name]),
                "fraction_mean": _mean(opponent_detail_fractions[name]),
            }

    measured_mean = _mean(measured_totals)
    rollout_mean = _mean(rollout_seconds)
    payload: dict[str, object] = {
        "warmup": resolved.warmup,
        "max_measured_update": resolved.max_measured_update,
        "measured_updates": len(selected),
        "updates_in_window": sorted(
            u for u in (record_update(record) for record in selected) if u is not None
        ),
        "phases": phases_payload,
        "opponent_details": opponent_details,
        "rollout_phase_measured_total_seconds_mean": measured_mean,
        "rollout_seconds_mean": rollout_mean,
        "rollout_seconds_gap_mean": rollout_mean - measured_mean,
    }
    return payload


def extract_rollout_phase_breakdown_from_log(
    log_path: Path,
    *,
    window: ThroughputWindow | None = None,
) -> dict[str, object]:
    records = read_jsonl_records(log_path)
    payload = extract_rollout_phase_breakdown_from_records(records, window=window)
    payload["log_path"] = str(log_path)
    return payload


def _profile_records_from_payload(
    payload: Mapping[str, object],
) -> list[dict[str, object]] | None:
    raw = payload.get("per_update_records")
    if not isinstance(raw, list) or not raw:
        return None
    records: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            records.append(dict(item))
    return records or None


def extract_rollout_phase_breakdown_from_input(
    path: Path,
    *,
    window: ThroughputWindow | None = None,
) -> dict[str, object]:
    """Load phase rows from profile JSON, gate JSON (via log_path), or JSONL."""

    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    if path.name.endswith("_jax.jsonl") or path.suffix == ".jsonl":
        return extract_rollout_phase_breakdown_from_log(path, window=window)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")

    profile_records = _profile_records_from_payload(payload)
    if profile_records is not None:
        resolved_window = window
        if resolved_window is None:
            warmup = payload.get("warmup")
            max_update = payload.get("max_measured_update")
            resolved_window = ThroughputWindow(
                warmup=int(warmup) if isinstance(warmup, int | float) else 2,
                max_measured_update=(
                    int(max_update) if isinstance(max_update, int | float) else 20
                ),
            )
        breakdown = extract_rollout_phase_breakdown_from_records(
            profile_records,
            window=resolved_window,
        )
        breakdown["source_path"] = str(path)
        breakdown["preset"] = payload.get("preset")
        return breakdown

    log_path, gate_result_path = resolve_log_path_from_input(path)
    breakdown = extract_rollout_phase_breakdown_from_log(log_path, window=window)
    if gate_result_path is not None:
        breakdown["gate_result_path"] = str(gate_result_path)
    return breakdown


def format_rollout_phase_breakdown(payload: Mapping[str, object]) -> str:
    """Human-readable table for terminal output."""

    lines = [
        (
            f"Rollout phase breakdown "
            f"(updates {payload['warmup'] + 1}–{payload['max_measured_update']}, "
            f"n={payload['measured_updates']})"
        ),
        f"  source: {payload.get('log_path') or payload.get('source_path', '(unknown)')}",
        "",
        f"  {'Phase':<12} {'Mean s':>8}  {'Share':>7}",
        f"  {'-' * 12} {'-' * 8}  {'-' * 7}",
    ]
    phases = payload.get("phases")
    if isinstance(phases, dict):
        opponent_details = payload.get("opponent_details")
        for name in PHASE_NAMES:
            entry = phases.get(name)
            if not isinstance(entry, dict):
                continue
            sec = float(entry.get("seconds_mean", 0.0))
            frac = float(entry.get("fraction_mean", 0.0))
            lines.append(f"  {name:<12} {sec:8.3f}  {frac * 100:6.1f}%")
            if name == "opponent" and isinstance(opponent_details, dict):
                for detail_name in OPPONENT_DETAIL_NAMES:
                    detail = opponent_details.get(detail_name)
                    if not isinstance(detail, dict):
                        continue
                    detail_sec = float(detail.get("seconds_mean", 0.0))
                    detail_frac = float(detail.get("fraction_mean", 0.0))
                    label = detail_name.removeprefix("opponent_")
                    lines.append(
                        f"    {label:<10} {detail_sec:8.3f}  {detail_frac * 100:6.1f}%"
                    )
    measured = float(payload.get("rollout_phase_measured_total_seconds_mean", 0.0))
    rollout = float(payload.get("rollout_seconds_mean", 0.0))
    gap = float(payload.get("rollout_seconds_gap_mean", 0.0))
    lines.extend(
        [
            "",
            f"  measured total (instrumented): {measured:.3f}s",
            f"  rollout_seconds (wall):        {rollout:.3f}s",
            f"  gap (host/sync overhead):      {gap:.3f}s",
        ]
    )
    return "\n".join(lines)
