from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class RetentionDecision:
    deleted: list[Path]
    kept: list[Path]
    reclaimed_bytes: int
    dry_run: bool


MetricMode = Literal["min", "max"]


def _parse_update(path: Path) -> int | None:
    stem = path.stem
    for prefix in ("ckpt_", "jax_ckpt_"):
        if stem.startswith(prefix):
            suffix = stem[len(prefix):]
            if suffix == "last":
                return None
            if suffix.isdigit():
                return int(suffix)
    return None


def _collect_metric_by_update(log_path: Path, metric_name: str) -> dict[int, float]:
    if not log_path.exists() or not metric_name:
        return {}
    metrics: dict[int, float] = {}
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            update = row.get("update")
            metric = row.get(metric_name)
            if isinstance(update, int) and isinstance(metric, int | float):
                metrics[update] = float(metric)
    return metrics


def prune_checkpoints(
    run_dir: Path,
    *,
    log_path: Path,
    keep_last_n: int,
    keep_every_n_updates: int,
    keep_best_k_by_metric: int,
    best_metric_name: str,
    best_metric_mode: MetricMode,
    min_update_for_pruning: int,
    dry_run_pruning: bool,
) -> RetentionDecision:
    """Prune unprotected checkpoints within ``run_dir`` based on retention policy."""

    files = sorted(
        [p for p in run_dir.iterdir() if p.is_file() and _parse_update(p) is not None],
        key=lambda p: _parse_update(p) or -1,
    )
    updates = {p: _parse_update(p) for p in files}
    protected: set[Path] = set()

    sorted_updates = sorted([u for u in updates.values() if u is not None])
    if keep_last_n > 0:
        keep_latest = set(sorted_updates[-keep_last_n:])
        protected.update(p for p, u in updates.items() if u in keep_latest)

    if keep_every_n_updates > 0:
        protected.update(
            p for p, u in updates.items() if u is not None and u % keep_every_n_updates == 0
        )

    if keep_best_k_by_metric > 0 and best_metric_name:
        by_update = _collect_metric_by_update(log_path, best_metric_name)
        scored = [
            (u, by_update[u])
            for u in sorted_updates
            if u is not None and u in by_update
        ]
        reverse = best_metric_mode == "max"
        scored.sort(key=lambda item: item[1], reverse=reverse)
        best_updates = {u for u, _ in scored[:keep_best_k_by_metric]}
        protected.update(p for p, u in updates.items() if u in best_updates)

    deleted: list[Path] = []
    kept: list[Path] = []
    reclaimed = 0
    run_dir_resolved = run_dir.resolve()

    for path in files:
        update = updates[path]
        if update is None or update < min_update_for_pruning or path in protected:
            kept.append(path)
            continue

        resolved = path.resolve()
        try:
            resolved.relative_to(run_dir_resolved)
        except ValueError:
            kept.append(path)
            continue
        if not resolved.is_file() or resolved.is_symlink():
            kept.append(path)
            continue

        size = resolved.stat().st_size
        deleted.append(path)
        reclaimed += size
        if not dry_run_pruning:
            resolved.unlink(missing_ok=True)

    return RetentionDecision(deleted=deleted, kept=kept, reclaimed_bytes=reclaimed, dry_run=dry_run_pruning)
