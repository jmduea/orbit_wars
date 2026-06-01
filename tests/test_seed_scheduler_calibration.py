"""Unit tests for seed scheduler calibration helpers."""

from __future__ import annotations

from pathlib import Path

from src.jax.seed_scheduler_calibration import (
    SEED_SCHED_TRAIN_BASE,
    SeedSchedRunSnapshot,
    count_distinct_reseed_seeds,
    expand_reseed_intervals,
    latest_completed_run_dir,
    pick_reseed_interval,
)


def test_seed_sched_train_base_logs_every_update() -> None:
    assert "training.log_every=1" in SEED_SCHED_TRAIN_BASE


def test_run_seed_scheduler_sweep_dry_run_prints_arm_plan(capsys) -> None:
    from src.jax.seed_scheduler_calibration import run_seed_scheduler_sweep

    run_seed_scheduler_sweep(
        opponents=("noop_only",),
        reseed_intervals=(0, 25),
        total_updates=500,
        output_root=Path("outputs"),
        repo_root=Path("."),
        dry_run=True,
    )
    captured = capsys.readouterr().out
    assert "2 training arm(s)" in captured
    assert "training.log_every=1" in captured or "ow train" in captured


def test_expand_reseed_intervals_includes_total_fifth() -> None:
    intervals = expand_reseed_intervals(
        (0, 50), total_updates=500, include_total_fifth=True
    )
    assert intervals == (0, 50, 100)


def test_count_distinct_reseed_seeds() -> None:
    records = [
        {"reseed_events": [{"new_seed": 7}, {"new_seed": 8}]},
        {"reseed_events": [{"new_seed": 8}]},
        {"reseed_events": []},
    ]
    assert count_distinct_reseed_seeds(records) == 2


def test_latest_completed_run_dir_skips_empty_jsonl(tmp_path: Path) -> None:
    campaign = "seed_sched_cal_self_play_only_reseed25_u500"
    runs_root = tmp_path / "campaigns" / campaign / "runs"
    empty_run = runs_root / "empty-run"
    good_run = runs_root / "good-run"
    empty_run.mkdir(parents=True)
    good_run.mkdir(parents=True)
    (empty_run / "logs").mkdir()
    (good_run / "logs").mkdir()
    (empty_run / "logs" / "empty_jax.jsonl").write_text("", encoding="utf-8")
    (good_run / "logs" / "good_jax.jsonl").write_text(
        '{"update": 1}\n', encoding="utf-8"
    )

    assert latest_completed_run_dir(campaign=campaign, output_root=tmp_path) == good_run


def _snapshot(
    *,
    opponent: str,
    interval: int,
    min_rate: float,
    std_rate: float = 0.05,
    kl: float = 0.002,
) -> SeedSchedRunSnapshot:
    return SeedSchedRunSnapshot(
        opponent=opponent,  # type: ignore[arg-type]
        reseed_interval=interval,
        effective_reseed_interval=interval,
        total_updates=500,
        train_seed=42,
        run_dir=None,
        log_path=None,
        checkpoint_path=None,
        record_count=500,
        distinct_reseed_seeds=10,
        stability={"approx_kl_mean": kl, "finite_scalars": True},
        training_proxy={
            "overall_win_rate_mean": min_rate,
            "overall_win_rate_last_window": min_rate,
        },
        eval_win_rates_by_seed={"43": min_rate},
        eval_win_rate_mean=min_rate,
        eval_win_rate_std=std_rate,
        eval_win_rate_min=min_rate,
    )


def test_pick_reseed_interval_prefers_higher_min_win_rate() -> None:
    snapshots = [
        _snapshot(opponent="noop_only", interval=0, min_rate=0.4),
        _snapshot(opponent="random_only", interval=0, min_rate=0.35),
        _snapshot(opponent="self_play_only", interval=0, min_rate=0.38),
        _snapshot(opponent="noop_only", interval=50, min_rate=0.55),
        _snapshot(opponent="random_only", interval=50, min_rate=0.52),
        _snapshot(opponent="self_play_only", interval=50, min_rate=0.51),
    ]
    decision = pick_reseed_interval(snapshots)
    assert decision["chosen_interval"] == 50
    assert decision["min_eval_win_rate"] == 0.51
