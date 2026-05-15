import numpy as np

from src.config import EnvConfig
from src.features import build_candidate_features, build_candidates, real_candidate_slots
from src.game_types import GameState, PlanetState
from src.train import candidate_diagnostics


def planet(pid: int, owner: int, x: float, y: float) -> PlanetState:
    return PlanetState(id=pid, owner=owner, x=x, y=y, radius=1.0, ships=10, production=1)


def test_real_candidate_slots_reserves_index_zero_for_no_op() -> None:
    assert real_candidate_slots(0) == 0
    assert real_candidate_slots(1) == 0
    assert real_candidate_slots(8) == 7


def test_build_candidates_uses_real_slots_and_does_not_cap_enemies_at_one_third() -> None:
    src = planet(0, 0, 10.0, 10.0)
    enemies = [planet(pid, 1, 12.0 + pid, 10.0) for pid in range(1, 6)]
    neutral = planet(6, -1, 20.0, 10.0)
    friendly = planet(7, 0, 21.0, 10.0)
    state = GameState(step=0, player=0, planets=[src, *enemies, neutral, friendly], fleets=[])
    cfg = EnvConfig(candidate_count=8)

    candidates = build_candidates(src, state, cfg)

    assert len(candidates) == 7
    assert sum(candidate.owner == 1 for candidate in candidates) == 5
    assert sum(candidate.owner == -1 for candidate in candidates) == 1
    assert sum(candidate.owner == 0 for candidate in candidates) == 1


def test_candidate_diagnostics_excludes_no_op_and_reports_owner_shares() -> None:
    src = planet(0, 0, 10.0, 10.0)
    candidates = [planet(1, 1, 12.0, 10.0), planet(2, -1, 14.0, 10.0), planet(3, 0, 16.0, 10.0)]
    state = GameState(step=0, player=0, planets=[src, *candidates], fleets=[])
    cfg = EnvConfig(candidate_count=4)
    features, mask, *_ = build_candidate_features(src, candidates, state, cfg)
    batch = type(
        "Batch",
        (),
        {
            "candidate_features": features.reshape(1, cfg.candidate_count, -1),
            "candidate_mask": mask.reshape(1, cfg.candidate_count),
        },
    )()

    stats = candidate_diagnostics(batch)  # type: ignore[arg-type]

    assert stats["candidate_valid_total"] == 3.0
    assert stats["candidate_source_rows"] == 1.0
    np.testing.assert_allclose(stats["candidate_enemy_total"], 1.0)
    np.testing.assert_allclose(stats["candidate_neutral_total"], 1.0)
    np.testing.assert_allclose(stats["candidate_friendly_total"], 1.0)
