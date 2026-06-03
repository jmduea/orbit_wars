"""Tests for unified ladder qualifier mode."""

from __future__ import annotations

from src.artifacts.tournament.unified.spec import (
    parse_unified_tournament_section,
    qualifier_sniper_stage,
    with_qualifier_floors,
)


def test_with_qualifier_floors_sets_one_point_zero() -> None:
    spec = parse_unified_tournament_section(
        {"noop_min_combined": 0.76, "random_min_combined": 0.76}
    )
    qualified = with_qualifier_floors(spec)
    assert qualified.stage1.floors["noop"] == 1.0
    assert qualified.stage1.floors["random"] == 1.0


def test_qualifier_sniper_stage_targets_nearest_sniper() -> None:
    spec = parse_unified_tournament_section({})
    sniper = qualifier_sniper_stage(spec)
    assert sniper.opponents == ("nearest_sniper",)
    assert sniper.floors["nearest_sniper"] == 1.0
