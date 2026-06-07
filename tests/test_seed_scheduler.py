"""Unit tests for adaptive rollout seed scheduling."""

from __future__ import annotations

import pytest

from src.training.seed_scheduler import (
    SeedScheduleConfig,
    SeedScheduler,
    resolve_reseed_every_updates,
)


def test_should_reseed_periodic_at_multiples() -> None:
    scheduler = SeedScheduler(
        base_seed=42,
        cfg=SeedScheduleConfig(reseed_every_updates=3),
    )
    assert scheduler.should_reseed(1) == (False, "")
    assert scheduler.should_reseed(2) == (False, "")
    assert scheduler.should_reseed(3) == (True, "periodic")
    assert scheduler.should_reseed(6) == (True, "periodic")


def test_should_reseed_disabled_when_interval_zero() -> None:
    scheduler = SeedScheduler(
        base_seed=42,
        cfg=SeedScheduleConfig(reseed_every_updates=0),
    )
    for update in (1, 10, 100):
        assert scheduler.should_reseed(update) == (False, "")


def test_should_reseed_on_plateau_after_window_fills() -> None:
    scheduler = SeedScheduler(
        base_seed=7,
        cfg=SeedScheduleConfig(
            reseed_on_plateau=True,
            plateau_window=3,
            plateau_delta=0.01,
        ),
    )
    for value in (1.0, 1.005, 1.002):
        scheduler.update_metric(value)
    assert scheduler.should_reseed(5) == (True, "plateau")


def test_reseed_random_jump_changes_seed() -> None:
    scheduler = SeedScheduler(
        base_seed=100,
        cfg=SeedScheduleConfig(reseed_every_updates=2),
    )
    event = scheduler.reseed(update=2, reason="periodic")
    assert event.old_seed == 100
    assert event.new_seed != event.old_seed
    assert event.policy == "random_jump"
    assert event.reason == "periodic"


def test_reseed_shuffled_pool_cycles_training_set() -> None:
    pool = [11, 22, 33]
    scheduler = SeedScheduler(
        base_seed=5,
        cfg=SeedScheduleConfig(training_seed_set=pool),
    )
    first = scheduler.reseed(update=1, reason="forced", policy="shuffled_pool")
    second = scheduler.reseed(update=2, reason="forced", policy="shuffled_pool")
    third = scheduler.reseed(update=3, reason="forced", policy="shuffled_pool")
    fourth = scheduler.reseed(update=4, reason="forced", policy="shuffled_pool")

    assert {first.new_seed, second.new_seed, third.new_seed} == set(pool)
    assert fourth.new_seed == first.new_seed


def test_next_seed_policy_prefers_pool_when_configured() -> None:
    scheduler = SeedScheduler(
        base_seed=1,
        cfg=SeedScheduleConfig(training_seed_set=[9, 8]),
    )
    assert scheduler.next_seed_policy(update=1) == "shuffled_pool"


def test_parse_seed_set_range_and_list() -> None:
    assert SeedScheduler.parse_seed_set("1..3") == [1, 2, 3]
    assert SeedScheduler.parse_seed_set("10,20,30") == [10, 20, 30]
    assert SeedScheduler.parse_seed_set([4, 5]) == [4, 5]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1-3", [1, 2, 3]),
        ("3-1", [3, 2, 1]),
    ],
)
def test_parse_seed_set_dash_range(raw: str, expected: list[int]) -> None:
    assert SeedScheduler.parse_seed_set(raw) == expected


def test_random_jump_avoids_eval_seed_set() -> None:
    scheduler = SeedScheduler(
        base_seed=100,
        cfg=SeedScheduleConfig(
            reseed_every_updates=2,
            eval_seed_set=[101, 102],
        ),
    )
    for _ in range(20):
        event = scheduler.reseed(update=2, reason="periodic")
        assert event.new_seed not in {101, 102}


def test_incremental_reseed_skips_eval_seed_set() -> None:
    scheduler = SeedScheduler(
        base_seed=42,
        cfg=SeedScheduleConfig(eval_seed_set=[43, 44, 45, 46]),
    )
    event = scheduler.reseed(update=1, reason="forced", policy="incremental")
    assert event.new_seed == 47
    assert event.new_seed not in {43, 44, 45, 46}


def test_advance_skips_eval_seed_set() -> None:
    scheduler = SeedScheduler(
        base_seed=42,
        cfg=SeedScheduleConfig(eval_seed_set=[43, 44, 45, 46]),
    )
    assert scheduler.advance(1) == 47


def test_resolve_reseed_every_updates_auto_scale() -> None:
    assert resolve_reseed_every_updates(configured=-1, total_updates=500) == 50
    assert resolve_reseed_every_updates(configured=-1, total_updates=100) == 25
    assert resolve_reseed_every_updates(configured=-1, total_updates=2000) == 200
    assert resolve_reseed_every_updates(configured=0, total_updates=500) == 0
    assert resolve_reseed_every_updates(configured=50, total_updates=500) == 50
