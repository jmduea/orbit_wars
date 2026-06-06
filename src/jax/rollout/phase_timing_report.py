"""Extract rollout phase timing breakdown from training JSONL (opt-in telemetry)."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.jax.preflight import read_jsonl_records

PHASE_NAMES: tuple[str, ...] = (
    "policy",
    "opponent",
    "env_step",
    "reset",
    "post_step",
)

PHASE_SECOND_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_seconds" for name in PHASE_NAMES
)
PHASE_FRACTION_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_fraction" for name in PHASE_NAMES
)
MEASURED_TOTAL_KEY = "rollout_phase_measured_total_seconds"


@dataclass(frozen=True, slots=True)
class PhaseTimingWindow:
    warmup: int
    max_measured_update: int

    @property
    def first_update(self) -> int:
        return self.warmup + 1

    def includes(self, update: int) -> bool:
        return self.first_update <= update <= self.max_measured_update


def _record_update(record: Mapping[str, object]) -> int | None:
    update = record.get("update")
    if isinstance(update, int):
        return update
    if isinstance(update, float) and update.is_integer():
        return int(update)
    return None


def _record_float(record: Mapping[str, object], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _profile_records_from_payload(payload: Mapping[str, object]) -> list[dict[str, object]] | None:
    raw = payload.get("per_update_records")
    if not isinstance(raw, list) or not raw:
        return None
    records: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            records.append(dict(item))
    return records or None


def resolve_log_path_from_input(path: Path) -> tuple[Path, Path | None]:
    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    if path.name.endswith("_jax.jsonl") or path.suffix == ".jsonl":
        return path, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    if _profile_records_from_payload(payload) is not None:
        raise ValueError(
            f"{path} is a rollout-phase-profile summary; "
            "use extract_rollout_phase_breakdown_from_input()"
        )
    stage = payload.get("stage")
    if isinstance(stage, dict):
        log_path = stage.get("log_path")
        if isinstance(log_path, str) and log_path:
            return Path(log_path), path
    log_path = payload.get("log_path")
    if isinstance(log_path, str) and log_path:
        return Path(log_path), path
    raise ValueError(
        f"no log_path in gate result {path}; pass a *_jax.jsonl path directly"
    )


def extract_rollout_phase_breakdown_from_input(
    path: Path,
    *,
    window: PhaseTimingWindow | None = None,
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
            resolved_window = PhaseTimingWindow(
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


def _has_phase_timing(record: Mapping[str, object]) -> bool:
    return _record_float(record, PHASE_SECOND_KEYS[0]) is not None


def extract_rollout_phase_breakdown_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    window: PhaseTimingWindow | None = None,
) -> dict[str, object]:
    resolved = window or PhaseTimingWindow(warmup=2, max_measured_update=20)
    selected: list[Mapping[str, object]] = []
    for record in records:
        update = _record_update(record)
        if update is None or not resolved.includes(update):
            continue
        if not _has_phase_timing(record):
            continue
        selected.append(record)

    if not selected:
        raise ValueError(
            "no rollout phase timing rows in window "
            f"updates {resolved.first_update}–{resolved.max_measured_update}. "
            "Run `ow benchmark rollout-phase-profile`, then "
            "`ow benchmark rollout-phase-breakdown` on --out JSON."
        )

    phase_seconds: dict[str, list[float]] = {name: [] for name in PHASE_NAMES}
    phase_fractions: dict[str, list[float]] = {name: [] for name in PHASE_NAMES}
    measured_totals: list[float] = []
    rollout_seconds: list[float] = []

    for record in selected:
        for name, sec_key, frac_key in zip(
            PHASE_NAMES, PHASE_SECOND_KEYS, PHASE_FRACTION_KEYS, strict=True
        ):
            sec = _record_float(record, sec_key)
            frac = _record_float(record, frac_key)
            if sec is not None:
                phase_seconds[name].append(sec)
            if frac is not None:
                phase_fractions[name].append(frac)
        measured = _record_float(record, MEASURED_TOTAL_KEY)
        if measured is not None:
            measured_totals.append(measured)
        rollout_s = _record_float(record, "rollout_seconds")
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

    measured_mean = _mean(measured_totals)
    rollout_mean = _mean(rollout_seconds)
    return {
        "warmup": resolved.warmup,
        "max_measured_update": resolved.max_measured_update,
        "measured_updates": len(selected),
        "updates_in_window": sorted(
            u for u in (_record_update(record) for record in selected) if u is not None
        ),
        "phases": phases_payload,
        "rollout_phase_measured_total_seconds_mean": measured_mean,
        "rollout_seconds_mean": rollout_mean,
        "rollout_seconds_gap_mean": rollout_mean - measured_mean,
    }


def extract_rollout_phase_breakdown_from_log(
    log_path: Path,
    *,
    window: PhaseTimingWindow | None = None,
) -> dict[str, object]:
    records = read_jsonl_records(log_path)
    payload = extract_rollout_phase_breakdown_from_records(records, window=window)
    payload["log_path"] = str(log_path)
    return payload


def format_rollout_phase_breakdown(payload: Mapping[str, object]) -> str:
    lines = [
        (
            f"Rollout phase breakdown "
            f"(updates {int(payload['warmup']) + 1}–{payload['max_measured_update']}, "
            f"n={payload['measured_updates']})"
        ),
        f"  source: {payload.get('log_path') or payload.get('source_path', '(unknown)')}",
        "",
        f"  {'Phase':<12} {'Mean s':>8}  {'Share':>7}",
        f"  {'-' * 12} {'-' * 8}  {'-' * 7}",
    ]
    phases = payload.get("phases")
    if isinstance(phases, dict):
        for name in PHASE_NAMES:
            entry = phases.get(name)
            if not isinstance(entry, dict):
                continue
            sec = float(entry.get("seconds_mean", 0.0))
            frac = float(entry.get("fraction_mean", 0.0))
            lines.append(f"  {name:<12} {sec:8.3f}  {frac * 100:6.1f}%")
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
