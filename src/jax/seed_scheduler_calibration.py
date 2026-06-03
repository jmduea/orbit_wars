"""Calibration sweep for periodic seed-scheduler reseed intervals."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.jax.preflight_calibration import (
    find_latest_checkpoint,
    read_jsonl_records,
    run_ow_train,
)
from src.training.seed_scheduler import resolve_reseed_every_updates

OpponentProfile = Literal["noop_only", "random_only", "self_play_only"]

DEFAULT_OPPONENTS: tuple[OpponentProfile, ...] = (
    "noop_only",
    "random_only",
    "self_play_only",
)
DEFAULT_RESEED_INTERVALS: tuple[int, ...] = (0, 25, 50, 100)
DEFAULT_TRAIN_SEED = 42
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 43, 44, 45, 46)
DEFAULT_TOTAL_UPDATES = 500
DEFAULT_BASELINE = "noop"

SEED_SCHED_TRAIN_BASE: tuple[str, ...] = (
    "training=workstation",
    "task=shield_off",
    "curriculum=off",
    f"seed={DEFAULT_TRAIN_SEED}",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "artifacts.replay.enabled=false",
    "training.log_every=1",
)

SEED_SCHED_CAMPAIGN_RE = re.compile(
    r"^seed_sched_cal_(noop_only|random_only|self_play_only)_reseed(-?\d+)_u(\d+)$"
)


def seed_sched_campaign(
    opponent: OpponentProfile, *, reseed_interval: int, total_updates: int
) -> str:
    return f"seed_sched_cal_{opponent}_reseed{reseed_interval}_u{total_updates}"


def expand_reseed_intervals(
    configured: tuple[int, ...], *, total_updates: int, include_total_fifth: bool
) -> tuple[int, ...]:
    intervals = list(configured)
    if include_total_fifth:
        fifth = max(1, int(total_updates) // 5)
        if fifth not in intervals:
            intervals.append(fifth)
    return tuple(sorted(set(intervals)))


def count_distinct_reseed_seeds(records: list[dict[str, object]]) -> int:
    seeds: set[int] = set()
    for record in records:
        events = record.get("reseed_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict) and "new_seed" in event:
                seeds.add(int(event["new_seed"]))
    return len(seeds)


def training_proxy_metrics(
    records: list[dict[str, object]], *, window: int = 10
) -> dict[str, float | None]:
    win_rates = [
        float(record["overall_win_rate"])
        for record in records
        if record.get("overall_win_rate") is not None
    ]
    if not win_rates:
        return {
            "overall_win_rate_mean": None,
            "overall_win_rate_last_window": None,
        }
    tail = win_rates[-window:] if window > 0 else win_rates
    return {
        "overall_win_rate_mean": sum(win_rates) / len(win_rates),
        "overall_win_rate_last_window": sum(tail) / len(tail),
    }


def training_stability_metrics(
    records: list[dict[str, object]],
) -> dict[str, float | bool | None]:
    kl_values = [
        abs(float(record["approx_kl"]))
        for record in records
        if record.get("approx_kl") is not None
    ]
    policy_loss_values = [
        float(record["policy_loss"])
        for record in records
        if record.get("policy_loss") is not None
    ]
    value_loss_values = [
        float(record["value_loss"])
        for record in records
        if record.get("value_loss") is not None
    ]
    return {
        "approx_kl_mean": (sum(kl_values) / len(kl_values) if kl_values else None),
        "policy_loss_mean": (
            sum(policy_loss_values) / len(policy_loss_values)
            if policy_loss_values
            else None
        ),
        "value_loss_mean": (
            sum(value_loss_values) / len(value_loss_values)
            if value_loss_values
            else None
        ),
        "finite_scalars": all(
            value == value and abs(value) != float("inf")
            for value in (
                *(kl_values or [0.0]),
                *(policy_loss_values or [0.0]),
                *(value_loss_values or [0.0]),
            )
        ),
    }


@dataclass(frozen=True, slots=True)
class SeedSchedRunSnapshot:
    opponent: OpponentProfile
    reseed_interval: int
    effective_reseed_interval: int
    total_updates: int
    train_seed: int
    run_dir: str | None
    log_path: str | None
    checkpoint_path: str | None
    record_count: int
    distinct_reseed_seeds: int
    stability: dict[str, float | bool | None]
    training_proxy: dict[str, float | None]
    eval_win_rates_by_seed: dict[str, float]
    eval_win_rate_mean: float | None
    eval_win_rate_std: float | None
    eval_win_rate_min: float | None


def _eval_seeds_excluding_train(
    eval_seeds: tuple[int, ...], *, train_seed: int
) -> tuple[int, ...]:
    return tuple(seed for seed in eval_seeds if int(seed) != int(train_seed))


def run_tournament_win_rate(
    checkpoint: Path,
    *,
    eval_seed: int,
    repo_root: Path,
    output_root: Path,
    campaign: str,
    baseline: str,
    games_per_pair: int,
    dry_run: bool,
) -> float | None:
    output_dir = (
        output_root
        / "campaigns"
        / campaign
        / "evaluations"
        / f"seed_sched_eval_s{eval_seed}"
    )
    cmd = [
        "uv",
        "run",
        "ow",
        "eval",
        "tournament",
        "--checkpoint",
        str(checkpoint),
        "--campaign",
        campaign,
        "--output-root",
        str(output_root),
        "--output-dir",
        str(output_dir),
        "--seeds",
        str(eval_seed),
        "--games-per-pair",
        str(games_per_pair),
        "--formats",
        "2p_vs_baseline",
        "--baselines",
        baseline,
    ]
    if dry_run:
        print(" ".join(cmd), flush=True)
        return None
    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tournament eval failed for seed {eval_seed} (exit {proc.returncode})"
        )
    leaderboard_path = output_dir / "leaderboard.json"
    if not leaderboard_path.is_file():
        return None
    rows = json.loads(leaderboard_path.read_text(encoding="utf-8")).get("rows", [])
    if not rows:
        return None
    observed = rows[0].get("win_rate_vs_baseline")
    if observed is None:
        observed = rows[0].get("win_rate_vs_sniper")
    return float(observed) if observed is not None else None


def analyze_training_logs(
    *,
    opponent: OpponentProfile,
    reseed_interval: int,
    total_updates: int,
    train_seed: int,
    run_dir: Path,
) -> SeedSchedRunSnapshot:
    log_paths = sorted(run_dir.glob("logs/*_jax.jsonl"))
    log_path = log_paths[-1] if log_paths else None
    records = read_jsonl_records(log_path) if log_path is not None else []
    checkpoint = find_latest_checkpoint(run_dir)
    effective = (
        resolve_reseed_every_updates(
            configured=reseed_interval, total_updates=total_updates
        )
        if reseed_interval == -1
        else int(reseed_interval)
    )
    return SeedSchedRunSnapshot(
        opponent=opponent,
        reseed_interval=int(reseed_interval),
        effective_reseed_interval=int(effective),
        total_updates=int(total_updates),
        train_seed=int(train_seed),
        run_dir=str(run_dir),
        log_path=str(log_path) if log_path is not None else None,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        record_count=len(records),
        distinct_reseed_seeds=count_distinct_reseed_seeds(records),
        stability=training_stability_metrics(records),
        training_proxy=training_proxy_metrics(records),
        eval_win_rates_by_seed={},
        eval_win_rate_mean=None,
        eval_win_rate_std=None,
        eval_win_rate_min=None,
    )


def analyze_seed_sched_run(
    *,
    opponent: OpponentProfile,
    reseed_interval: int,
    total_updates: int,
    train_seed: int,
    run_dir: Path,
    eval_seeds: tuple[int, ...],
    repo_root: Path,
    output_root: Path,
    baseline: str,
    games_per_pair: int,
    dry_run: bool,
    run_eval: bool = True,
) -> SeedSchedRunSnapshot:
    snapshot = analyze_training_logs(
        opponent=opponent,
        reseed_interval=reseed_interval,
        total_updates=total_updates,
        train_seed=train_seed,
        run_dir=run_dir,
    )
    if not run_eval or snapshot.checkpoint_path is None:
        return snapshot
    eval_rates: dict[str, float] = {}
    campaign = seed_sched_campaign(
        opponent, reseed_interval=reseed_interval, total_updates=total_updates
    )
    for eval_seed in _eval_seeds_excluding_train(eval_seeds, train_seed=train_seed):
        rate = run_tournament_win_rate(
            Path(snapshot.checkpoint_path),
            eval_seed=eval_seed,
            repo_root=repo_root,
            output_root=output_root,
            campaign=campaign,
            baseline=baseline,
            games_per_pair=games_per_pair,
            dry_run=dry_run,
        )
        if rate is not None:
            eval_rates[str(eval_seed)] = rate
    rate_values = list(eval_rates.values())
    return SeedSchedRunSnapshot(
        opponent=snapshot.opponent,
        reseed_interval=snapshot.reseed_interval,
        effective_reseed_interval=snapshot.effective_reseed_interval,
        total_updates=snapshot.total_updates,
        train_seed=snapshot.train_seed,
        run_dir=snapshot.run_dir,
        log_path=snapshot.log_path,
        checkpoint_path=snapshot.checkpoint_path,
        record_count=snapshot.record_count,
        distinct_reseed_seeds=snapshot.distinct_reseed_seeds,
        stability=snapshot.stability,
        training_proxy=snapshot.training_proxy,
        eval_win_rates_by_seed=eval_rates,
        eval_win_rate_mean=(statistics.mean(rate_values) if rate_values else None),
        eval_win_rate_std=(
            statistics.pstdev(rate_values) if len(rate_values) > 1 else 0.0
        )
        if rate_values
        else None,
        eval_win_rate_min=min(rate_values) if rate_values else None,
    )


def latest_completed_run_dir(*, campaign: str, output_root: Path) -> Path | None:
    """Return the newest run directory that has a non-empty JAX training JSONL."""

    runs_root = output_root / "campaigns" / campaign / "runs"
    if not runs_root.is_dir():
        return None
    completed: list[Path] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        logs = sorted(run_dir.glob("logs/*_jax.jsonl"))
        if logs and logs[-1].stat().st_size > 0:
            completed.append(run_dir)
    if not completed:
        return None
    return max(completed, key=lambda path: path.stat().st_mtime)


def discover_seed_sched_runs(
    output_root: Path, *, total_updates: int, train_seed: int
) -> list[tuple[OpponentProfile, int, Path]]:
    campaigns_root = output_root / "campaigns"
    if not campaigns_root.is_dir():
        return []
    discovered: list[tuple[OpponentProfile, int, Path]] = []
    for campaign_dir in sorted(campaigns_root.iterdir()):
        if not campaign_dir.is_dir():
            continue
        match = SEED_SCHED_CAMPAIGN_RE.match(campaign_dir.name)
        if not match:
            continue
        opponent, reseed_text, updates_text = match.groups()
        if int(updates_text) != int(total_updates):
            continue
        try:
            run_dir = latest_completed_run_dir(
                campaign=campaign_dir.name, output_root=output_root
            )
        except FileNotFoundError:
            run_dir = None
        if run_dir is None:
            continue
        discovered.append((opponent, int(reseed_text), run_dir))  # type: ignore[arg-type]
    return discovered


def run_seed_scheduler_sweep(
    *,
    opponents: tuple[OpponentProfile, ...],
    reseed_intervals: tuple[int, ...],
    total_updates: int,
    output_root: Path,
    repo_root: Path,
    dry_run: bool,
) -> None:
    arms = [
        (opponent, reseed_interval)
        for opponent in opponents
        for reseed_interval in reseed_intervals
    ]
    print(
        f"Seed-scheduler calibration: {len(arms)} training arm(s), "
        f"{total_updates} updates each, output_root={output_root}",
        flush=True,
    )
    for arm_index, (opponent, reseed_interval) in enumerate(arms, start=1):
        campaign = seed_sched_campaign(
            opponent,
            reseed_interval=reseed_interval,
            total_updates=total_updates,
        )
        overrides = [
            *SEED_SCHED_TRAIN_BASE,
            f"opponents={opponent}",
            f"training.total_updates={total_updates}",
            f"training.reseed_every_updates={reseed_interval}",
            f"output.campaign={campaign}",
            f"output.root={output_root}",
        ]
        run_ow_train(
            overrides,
            repo_root=repo_root,
            dry_run=dry_run,
            label=(
                f"seed-scheduler calibration arm {arm_index}/{len(arms)} "
                f"opponent={opponent} reseed={reseed_interval} "
                f"updates={total_updates} campaign={campaign}"
            ),
        )


def _stability_passes(snapshot: SeedSchedRunSnapshot) -> bool:
    kl = snapshot.stability.get("approx_kl_mean")
    if kl is None or not isinstance(kl, (int, float)):
        return False
    if float(kl) > 0.005:
        return False
    finite = snapshot.stability.get("finite_scalars")
    return bool(finite)


def pick_reseed_interval(
    snapshots: list[SeedSchedRunSnapshot],
    *,
    required_opponents: tuple[OpponentProfile, ...] = DEFAULT_OPPONENTS,
) -> dict[str, object]:
    """Choose the best reseed interval by min eval win rate across opponents."""

    by_interval: dict[int, list[SeedSchedRunSnapshot]] = {}
    for snapshot in snapshots:
        if snapshot.eval_win_rate_min is None:
            continue
        by_interval.setdefault(snapshot.reseed_interval, []).append(snapshot)

    required = set(required_opponents)
    candidates: list[tuple[int, float, float, int]] = []
    for interval, group in sorted(by_interval.items()):
        opponents_present = {item.opponent for item in group}
        if opponents_present != required:
            continue
        if not all(_stability_passes(item) for item in group):
            continue
        min_rates = [float(item.eval_win_rate_min) for item in group]
        std_rates = [float(item.eval_win_rate_std or 0.0) for item in group]
        candidates.append(
            (
                interval,
                min(min_rates),
                statistics.mean(std_rates),
                len(group),
            )
        )

    if not candidates:
        return {
            "chosen_interval": None,
            "reason": "no interval passed stability on all opponents with eval data",
        }

    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    chosen = candidates[0][0]
    baseline_min = by_interval.get(0)
    baseline_min_rate = (
        min(float(item.eval_win_rate_min) for item in baseline_min)
        if baseline_min
        else None
    )
    return {
        "chosen_interval": chosen,
        "chosen_effective_interval": resolve_reseed_every_updates(
            configured=chosen, total_updates=snapshots[0].total_updates
        ),
        "min_eval_win_rate": candidates[0][1],
        "mean_eval_win_rate_std": candidates[0][2],
        "baseline_min_eval_win_rate": baseline_min_rate,
        "candidate_count": len(candidates),
    }


def snapshot_to_dict(snapshot: SeedSchedRunSnapshot) -> dict[str, object]:
    return {
        "opponent": snapshot.opponent,
        "reseed_interval": snapshot.reseed_interval,
        "effective_reseed_interval": snapshot.effective_reseed_interval,
        "total_updates": snapshot.total_updates,
        "train_seed": snapshot.train_seed,
        "run_dir": snapshot.run_dir,
        "log_path": snapshot.log_path,
        "checkpoint_path": snapshot.checkpoint_path,
        "record_count": snapshot.record_count,
        "distinct_reseed_seeds": snapshot.distinct_reseed_seeds,
        "stability": snapshot.stability,
        "training_proxy": snapshot.training_proxy,
        "eval_win_rates_by_seed": snapshot.eval_win_rates_by_seed,
        "eval_win_rate_mean": snapshot.eval_win_rate_mean,
        "eval_win_rate_std": snapshot.eval_win_rate_std,
        "eval_win_rate_min": snapshot.eval_win_rate_min,
    }


def build_seed_scheduler_calibration_report(
    snapshots: list[SeedSchedRunSnapshot],
    *,
    commit_sha: str | None,
    seconds_total: float,
    analyze_only: bool,
    eval_seeds: tuple[int, ...],
    train_seed: int,
    required_opponents: tuple[OpponentProfile, ...] = DEFAULT_OPPONENTS,
) -> dict[str, object]:
    decision = pick_reseed_interval(snapshots, required_opponents=required_opponents)
    return {
        "gate": "seed_scheduler_calibration",
        "commit_sha": commit_sha,
        "seconds_total": seconds_total,
        "analyze_only": analyze_only,
        "train_seed": train_seed,
        "eval_seeds": list(
            _eval_seeds_excluding_train(eval_seeds, train_seed=train_seed)
        ),
        "decision": decision,
        "runs": [snapshot_to_dict(item) for item in snapshots],
    }


def write_seed_scheduler_calibration_report(
    path: Path, report: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
