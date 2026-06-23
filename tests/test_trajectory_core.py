"""Tests for shared trajectory-shield core primitives."""

from __future__ import annotations

from src.shield.trajectory_core import (
    SHIELD_PARITY_PAIRS,
    acceptable_planet_hit,
    bounds_exit_time,
    fleet_speed,
    line_circle_intersection_time,
    moving_circle_hit_time,
    ship_count_for_bucket,
)


def test_shield_parity_registry_covers_eight_symbol_pairs() -> None:
    assert len(SHIELD_PARITY_PAIRS) == 8


def test_ship_count_for_bucket_edges() -> None:
    assert ship_count_for_bucket(10, 0, 8) == 0
    assert ship_count_for_bucket(10, 4, 8) == 6


def test_fleet_speed_monotonic_and_capped() -> None:
    assert fleet_speed(1.0) <= fleet_speed(100.0)
    assert fleet_speed(10_000.0) <= 6.0


def test_moving_circle_hit_at_start_when_overlapping() -> None:
    assert (
        moving_circle_hit_time(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5) == 0.0
    )


def test_line_circle_intersection_hits_sun_center() -> None:
    hit = line_circle_intersection_time(0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 1.0)
    assert hit is not None
    assert 0.0 <= hit <= 1.0


def test_bounds_exit_time_detects_right_edge() -> None:
    hit = bounds_exit_time(0.5, 0.5, 25.0, 0.5)
    assert hit is not None
    assert 0.0 < hit <= 1.0


def test_acceptable_planet_hit_modes() -> None:
    assert acceptable_planet_hit(
        planet_id=3,
        planet_owner=1,
        player=0,
        target_id=3,
        hit_mode="selected_target",
    )
    assert not acceptable_planet_hit(
        planet_id=3,
        planet_owner=0,
        player=0,
        target_id=3,
        hit_mode="non_friendly",
    )
