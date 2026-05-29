"""Aggregate validation seed-sweep JSON artifacts into mean ± std summary."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "docs" / "benchmarks"

METRICS = (
    "seconds_total",
    "env_steps_per_sec",
    "compile_seconds_to_update_3",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "mean_active_launches_per_turn",
    "overall_win_rate",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _seed_from_payload(payload: dict) -> int | None:
    for override in payload.get("overrides", []):
        if override.startswith("seed="):
            return int(override.split("=", 1)[1])
    return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "—"
    if std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    self_play_paths = sorted(BENCH_DIR.glob("validation-seed-*-500u.json"))
    legacy = BENCH_DIR / "validation-500u.json"
    if legacy.exists() and legacy not in self_play_paths:
        self_play_paths = [legacy] + self_play_paths

    rows: list[dict] = []
    for path in self_play_paths:
        payload = _load(path)
        seed = _seed_from_payload(payload)
        row = {"path": path.name, "seed": seed, "updates": payload.get("updates")}
        for key in METRICS:
            val = payload.get(key)
            row[key] = float(val) if val is not None else None
        rows.append(row)

    summary: dict[str, tuple[float, float]] = {}
    for key in METRICS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = _mean_std(vals)

    print(json.dumps({"rows": rows, "summary": {k: {"mean": m, "std": s} for k, (m, s) in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
