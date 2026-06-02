"""Calibration sweep for preflight learning-signal thresholds."""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

OpponentProfile = Literal["noop_only", "random_only"]

DEFAULT_SEEDS: tuple[int, ...] = (42, 43)
DEFAULT_UPDATE_COUNTS: tuple[int, ...] = (200, 500)
DEFAULT_MODEL = "transformer_factorized_small"
WINDOW_UPDATES = 10

PREFLIGHT_TRAIN_BASE: tuple[str, ...] = (
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "telemetry.metric_groups.action_decision=true",
    "task=shield_cheap",
    "seed=42",
    "training.log_every=1",
)


def read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _launches_key(records: list[dict[str, object]]) -> str | None:
    for key in ("mean_active_launches_per_turn", "stop_utilization_ratio"):
        if any(key in record for record in records):
            return key
    return None


def window_mean_from_metric_rows(
    records: list[dict[str, object]], key: str, *, last_n: int
) -> float | None:
    """Mean of ``key`` over the last ``last_n`` metric rows (shared with preflight gates)."""

    return _window_mean(records, key, last_n=last_n)


def _window_mean(
    records: list[dict[str, object]], key: str, *, last_n: int
) -> float | None:
    tail = records[-last_n:] if last_n > 0 else records
    values = [
        float(record[key])
        for record in tail
        if key in record and record[key] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _window_mean_first(
    records: list[dict[str, object]], key: str, *, first_n: int
) -> float | None:
    head = records[:first_n] if first_n > 0 else records
    values = [
        float(record[key])
        for record in head
        if key in record and record[key] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def run_ow_train(
    overrides: list[str],
    *,
    repo_root: Path,
    dry_run: bool = False,
    label: str | None = None,
) -> None:
    cmd = ["uv", "run", "ow", "train", *overrides]
    if dry_run:
        print(" ".join(cmd), flush=True)
        return
    banner = label or "ow train"
    print(f"\n=== {banner} ===", flush=True)
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ow train failed with exit code {return_code}")


def latest_run_dir(*, campaign: str, output_root: Path) -> Path:
    runs_root = output_root / "campaigns" / campaign / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"No runs directory for campaign {campaign!r}: {runs_root}"
        )
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs under campaign {campaign!r}: {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    paths = sorted(ckpt_dir.glob("jax_ckpt_*.pkl"))
    return paths[-1] if paths else None


@dataclass(frozen=True, slots=True)
class TrainingSignalSnapshot:
    """Per-run JAX telemetry summary for calibration."""

    opponent: OpponentProfile
    seed: int
    total_updates: int
    model: str
    run_dir: str | None
    log_path: str | None
    checkpoint_path: str | None
    record_count: int
    win_rate_first_window: float | None
    win_rate_last_window: float | None
    win_rate_delta: float | None
    best_rolling_win_rate: float | None
    win_rate_mean: float | None
    launches_first_window: float | None
    launches_last_window: float | None
    launches_ratio: float | None


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Aggregate stats used to derive gate thresholds."""

    run_count: int
    models: tuple[str, ...]
    win_rate_delta_p25: float | None
    win_rate_delta_median: float | None
    best_rolling_win_rate_max: float | None
    best_rolling_win_rate_p75: float | None
    launches_ratio_p25: float | None
    observed_absolute_win_rate_max: float | None


CALIBRATION_CAMPAIGN_RE = re.compile(
    r"^preflight_calibrate_(noop|random)_s(\d+)_u(\d+)$"
)


def calibration_campaign(
    opponent: OpponentProfile, *, seed: int, total_updates: int
) -> str:
    slug = opponent.removesuffix("_only")
    return f"preflight_calibrate_{slug}_s{seed}_u{total_updates}"


def snapshot_from_run_dir(
    run_dir: Path,
    *,
    opponent: OpponentProfile,
    seed: int,
    total_updates: int,
    model: str = DEFAULT_MODEL,
) -> TrainingSignalSnapshot | None:
    """Build a calibration snapshot from a completed run directory."""

    log_files = sorted((run_dir / "logs").glob("*_jax.jsonl"))
    if not log_files:
        return None
    records = read_jsonl_records(log_files[0])
    if not records:
        return None
    return extract_training_signals(
        records,
        opponent=opponent,
        seed=seed,
        total_updates=total_updates,
        model=model,
        run_dir=run_dir,
        log_path=log_files[0],
        checkpoint=find_latest_checkpoint(run_dir),
    )


def discover_calibration_snapshots(
    output_root: Path,
    *,
    campaign_glob: str = "preflight_calibrate_*",
    model: str = DEFAULT_MODEL,
) -> list[TrainingSignalSnapshot]:
    """Analyze all completed calibration campaigns under ``output_root/campaigns/``."""

    campaigns_root = output_root / "campaigns"
    if not campaigns_root.is_dir():
        return []

    snapshots: list[TrainingSignalSnapshot] = []
    for campaign_dir in sorted(campaigns_root.glob(campaign_glob)):
        if not campaign_dir.is_dir():
            continue
        match = CALIBRATION_CAMPAIGN_RE.match(campaign_dir.name)
        if match is None:
            continue
        slug, seed_text, updates_text = match.groups()
        opponent: OpponentProfile = "noop_only" if slug == "noop" else "random_only"
        runs_root = campaign_dir / "runs"
        if not runs_root.is_dir():
            continue
        run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
        if not run_dirs:
            continue
        for run_dir in sorted(
            run_dirs, key=lambda path: path.stat().st_mtime, reverse=True
        ):
            snapshot = snapshot_from_run_dir(
                run_dir,
                opponent=opponent,
                seed=int(seed_text),
                total_updates=int(updates_text),
                model=model,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
                break
    return snapshots


def calibration_train_overrides(
    opponent: OpponentProfile,
    *,
    seed: int,
    total_updates: int,
    model: str = DEFAULT_MODEL,
) -> tuple[str, ...]:
    training_profile = (
        "planet_flow" if model == "planet_flow_target_heatmap" else "2p_16"
    )
    return (
        f"model={model}",
        f"training={training_profile}",
        f"training.total_updates={total_updates}",
        f"opponents={opponent}",
        "curriculum=off",
        *PREFLIGHT_TRAIN_BASE,
        *(
            (
                "artifacts=planet_flow_proof",
                "artifacts.artifact_pipeline.enabled=true",
            )
            if model == "planet_flow_target_heatmap"
            else ()
        ),
        f"seed={seed}",
    )


def extract_training_signals(
    records: list[dict[str, object]],
    *,
    opponent: OpponentProfile,
    seed: int,
    total_updates: int,
    model: str = DEFAULT_MODEL,
    run_dir: Path | None = None,
    log_path: Path | None = None,
    checkpoint: Path | None = None,
) -> TrainingSignalSnapshot:
    metric_rows = [
        record
        for record in records
        if "overall_win_rate" in record and record.get("update") is not None
    ]
    launches_key = _launches_key(metric_rows)
    window = min(WINDOW_UPDATES, len(metric_rows)) if metric_rows else 0
    win_first = (
        _window_mean_first(metric_rows, "overall_win_rate", first_n=window)
        if window
        else None
    )
    win_last = (
        _window_mean(metric_rows, "overall_win_rate", last_n=window) if window else None
    )
    win_delta = (
        (win_last - win_first)
        if win_first is not None and win_last is not None
        else None
    )
    rolling_best: float | None = None
    if metric_rows and window > 0:
        rolling = [
            _window_mean(metric_rows[: index + 1], "overall_win_rate", last_n=window)
            for index in range(window - 1, len(metric_rows))
        ]
        rolling_values = [value for value in rolling if value is not None]
        rolling_best = max(rolling_values) if rolling_values else None
    win_mean_values = [
        float(record["overall_win_rate"])
        for record in metric_rows
        if record.get("overall_win_rate") is not None
    ]
    launches_first = (
        _window_mean_first(metric_rows, launches_key, first_n=window)
        if window and launches_key is not None
        else None
    )
    launches_last = (
        _window_mean(metric_rows, launches_key, last_n=window)
        if window and launches_key is not None
        else None
    )
    launches_ratio = (
        (launches_last / launches_first)
        if launches_first is not None
        and launches_last is not None
        and launches_first > 0.0
        else None
    )
    return TrainingSignalSnapshot(
        opponent=opponent,
        seed=seed,
        total_updates=total_updates,
        model=model,
        run_dir=str(run_dir) if run_dir is not None else None,
        log_path=str(log_path) if log_path is not None else None,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        record_count=len(metric_rows),
        win_rate_first_window=win_first,
        win_rate_last_window=win_last,
        win_rate_delta=win_delta,
        best_rolling_win_rate=rolling_best,
        win_rate_mean=(statistics.mean(win_mean_values) if win_mean_values else None),
        launches_first_window=launches_first,
        launches_last_window=launches_last,
        launches_ratio=launches_ratio,
    )


def snapshot_to_dict(snapshot: TrainingSignalSnapshot) -> dict[str, object]:
    return {
        "opponent": snapshot.opponent,
        "seed": snapshot.seed,
        "total_updates": snapshot.total_updates,
        "model": snapshot.model,
        "run_dir": snapshot.run_dir,
        "log_path": snapshot.log_path,
        "checkpoint_path": snapshot.checkpoint_path,
        "record_count": snapshot.record_count,
        "win_rate_first_window": snapshot.win_rate_first_window,
        "win_rate_last_window": snapshot.win_rate_last_window,
        "win_rate_delta": snapshot.win_rate_delta,
        "best_rolling_win_rate": snapshot.best_rolling_win_rate,
        "win_rate_mean": snapshot.win_rate_mean,
        "launches_first_window": snapshot.launches_first_window,
        "launches_last_window": snapshot.launches_last_window,
        "launches_ratio": snapshot.launches_ratio,
    }


def summarize_calibration(
    snapshots: list[TrainingSignalSnapshot],
) -> CalibrationSummary:
    deltas = [
        value
        for value in (item.win_rate_delta for item in snapshots)
        if value is not None
    ]
    rolling = [
        value
        for value in (item.best_rolling_win_rate for item in snapshots)
        if value is not None
    ]
    launch_ratios = [
        value
        for value in (item.launches_ratio for item in snapshots)
        if value is not None
    ]
    absolute = [
        value
        for value in (item.win_rate_last_window for item in snapshots)
        if value is not None
    ]
    return CalibrationSummary(
        run_count=len(snapshots),
        models=tuple(sorted({item.model for item in snapshots})),
        win_rate_delta_p25=_percentile(deltas, 0.25),
        win_rate_delta_median=_percentile(deltas, 0.50),
        best_rolling_win_rate_max=max(rolling) if rolling else None,
        best_rolling_win_rate_p75=_percentile(rolling, 0.75),
        launches_ratio_p25=_percentile(launch_ratios, 0.25),
        observed_absolute_win_rate_max=max(absolute) if absolute else None,
    )


def derive_thresholds(summary: CalibrationSummary) -> dict[str, object]:
    """Derive gate thresholds from calibration aggregates."""

    if summary.run_count == 0:
        return default_thresholds(reason="no_calibration_runs")

    delta_floor = summary.win_rate_delta_p25
    if delta_floor is None:
        delta_floor = 0.08
    else:
        delta_floor = max(0.05, round(delta_floor * 0.75, 3))

    absolute_max = summary.observed_absolute_win_rate_max or 0.0
    rolling_max = summary.best_rolling_win_rate_max or 0.0
    use_tournament_win_proof = absolute_max < 0.75 or rolling_max < 0.80

    noop_tournament = min(0.70, round(max(rolling_max, absolute_max) * 0.85, 2))
    random_tournament = min(0.60, round(max(rolling_max, absolute_max) * 0.70, 2))
    noop_tournament = max(noop_tournament, 0.45)
    random_tournament = max(random_tournament, 0.35)

    learning_signal = {
        "window_updates": WINDOW_UPDATES,
        "min_win_rate_delta": delta_floor,
        "max_approx_kl": 0.15,
        "min_entropy": 1.0e-4,
    }
    thresholds = {
        "mode": "trend_plus_tournament" if use_tournament_win_proof else "absolute_jax",
        "learning_signal": learning_signal,
        "win_proof_tournament": {
            "noop_min_win_rate": noop_tournament,
            "random_min_win_rate": random_tournament,
            "games_per_pair": 4,
            "seeds": "0,1,2,3,4",
            "formats": "2p_vs_baseline",
        },
        "notes": [
            "Gates 2-3 use JAX learning-signal (trend), not absolute win rate.",
            "Gate 5 tournament carries absolute win proof on kaggle_environments.",
            f"Calibration observed max last-window win rate {absolute_max:.3f}, "
            f"max rolling-10 {rolling_max:.3f}.",
        ],
    }
    if "planet_flow_target_heatmap" in summary.models:
        planet_flow_learning = dict(learning_signal)
        planet_flow_learning["max_post_mask_unreachable_demand_rate"] = 0.05
        thresholds["planet_flow_learning_signal"] = planet_flow_learning
    return thresholds


def default_thresholds(*, reason: str) -> dict[str, object]:
    return {
        "mode": "trend_plus_tournament",
        "reason": reason,
        "learning_signal": {
            "window_updates": WINDOW_UPDATES,
            "min_win_rate_delta": 0.08,
            "max_approx_kl": 0.15,
            "min_entropy": 1.0e-4,
        },
        "win_proof_tournament": {
            "noop_min_win_rate": 0.55,
            "random_min_win_rate": 0.45,
            "games_per_pair": 4,
            "seeds": "0,1,2,3,4",
            "formats": "2p_vs_baseline",
        },
        "notes": ["Fallback thresholds; run ow benchmark calibrate to refresh."],
    }


def load_thresholds(path: Path) -> dict[str, object]:
    if not path.is_file():
        return default_thresholds(reason=f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "learning_signal" in payload and "win_proof_tournament" in payload:
        return payload
    if "thresholds" in payload and isinstance(payload["thresholds"], dict):
        return payload["thresholds"]
    return default_thresholds(reason="invalid calibration payload")


def analyze_jsonl_path(
    path: Path,
    *,
    opponent: OpponentProfile,
    seed: int,
    total_updates: int,
    model: str = DEFAULT_MODEL,
) -> TrainingSignalSnapshot:
    records = read_jsonl_records(path)
    return extract_training_signals(
        records,
        opponent=opponent,
        seed=seed,
        total_updates=total_updates,
        model=model,
        log_path=path,
    )


def run_calibration_train(
    opponent: OpponentProfile,
    *,
    seed: int,
    total_updates: int,
    model: str = DEFAULT_MODEL,
    output_root: Path = Path("outputs"),
    repo_root: Path,
    dry_run: bool = False,
) -> TrainingSignalSnapshot:
    campaign = calibration_campaign(opponent, seed=seed, total_updates=total_updates)
    overrides = [
        f"output.campaign={campaign}",
        f"output.root={output_root.as_posix()}",
        *calibration_train_overrides(
            opponent, seed=seed, total_updates=total_updates, model=model
        ),
    ]
    run_ow_train(
        overrides,
        repo_root=repo_root,
        dry_run=dry_run,
        label=f"preflight calibration {opponent} seed={seed} updates={total_updates}",
    )
    if dry_run:
        return extract_training_signals(
            [],
            opponent=opponent,
            seed=seed,
            total_updates=total_updates,
            model=model,
        )

    run_dir = latest_run_dir(campaign=campaign, output_root=output_root)
    snapshot = snapshot_from_run_dir(
        run_dir,
        opponent=opponent,
        seed=seed,
        total_updates=total_updates,
        model=model,
    )
    if snapshot is None:
        raise FileNotFoundError(
            f"No *_jax.jsonl under {run_dir / 'logs'} for campaign {campaign!r}"
        )
    return snapshot


def run_calibration_sweep(
    *,
    opponents: tuple[OpponentProfile, ...],
    seeds: tuple[int, ...],
    update_counts: tuple[int, ...],
    model: str = DEFAULT_MODEL,
    output_root: Path = Path("outputs"),
    repo_root: Path,
    dry_run: bool = False,
) -> list[TrainingSignalSnapshot]:
    snapshots: list[TrainingSignalSnapshot] = []
    for opponent in opponents:
        for seed in seeds:
            for total_updates in update_counts:
                snapshots.append(
                    run_calibration_train(
                        opponent,
                        seed=seed,
                        total_updates=total_updates,
                        model=model,
                        output_root=output_root,
                        repo_root=repo_root,
                        dry_run=dry_run,
                    )
                )
    return snapshots


def build_calibration_report(
    snapshots: list[TrainingSignalSnapshot],
    *,
    commit_sha: str | None,
    thresholds: dict[str, object],
    seconds_total: float,
    analyze_only: bool,
) -> dict[str, object]:
    summary = summarize_calibration(snapshots)
    return {
        "gate": "preflight_calibration",
        "commit_sha": commit_sha,
        "seconds_total": seconds_total,
        "analyze_only": analyze_only,
        "window_updates": WINDOW_UPDATES,
        "run_count": summary.run_count,
        "summary": {
            "win_rate_delta_p25": summary.win_rate_delta_p25,
            "win_rate_delta_median": summary.win_rate_delta_median,
            "best_rolling_win_rate_max": summary.best_rolling_win_rate_max,
            "best_rolling_win_rate_p75": summary.best_rolling_win_rate_p75,
            "launches_ratio_p25": summary.launches_ratio_p25,
            "observed_absolute_win_rate_max": summary.observed_absolute_win_rate_max,
        },
        "thresholds": thresholds,
        "runs": [snapshot_to_dict(item) for item in snapshots],
    }


def write_calibration_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = quantile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def git_head_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def default_calibration_json_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "benchmarks" / "preflight-calibration.json"
