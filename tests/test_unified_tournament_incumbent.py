from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.artifacts.tournament.types import AgentEntry
from src.artifacts.tournament.unified.incumbent import resolve_incumbent
from src.artifacts.tournament.unified.reporting import (
    UnifiedLadderVerdict,
    UnifiedStageResult,
)
from src.artifacts.tournament.unified.spec import parse_unified_tournament_section
from src.config import TrainConfig


def _spec(**overrides: object):
    base = {
        "enforcement": False,
        "four_p_baseline_fillers": ["noop", "random", "random"],
        "incumbent_bootstrap_opponent": "nearest_sniper",
    }
    base.update(overrides)
    return parse_unified_tournament_section(base)


def test_resolve_incumbent_from_scripted_bootstrap(tmp_path: Path) -> None:
    spec = _spec()
    incumbent = resolve_incumbent(spec, campaign="missing", output_root=tmp_path)
    assert incumbent is not None
    assert incumbent.agent_id == "incumbent"
    assert str(incumbent.checkpoint_path).startswith("scripted:")


def test_bootstrap_incumbent_differs_from_challenger_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "challenger.pkl"
    ckpt.write_bytes(b"stub")
    spec = _spec()
    with patch(
        "src.artifacts.tournament.unified.ladder.agent_from_checkpoint"
    ) as mock_agent:
        mock_agent.return_value = AgentEntry(
            agent_id="cand",
            checkpoint_path=ckpt,
            cfg=TrainConfig(),
            act_fn=lambda _obs: [],
        )
        challenger = mock_agent.return_value
        incumbent = resolve_incumbent(spec, campaign="missing", output_root=tmp_path)
    assert incumbent is not None
    assert incumbent.checkpoint_path != challenger.checkpoint_path
    assert incumbent.agent_id != challenger.agent_id


def test_resolve_incumbent_prefers_promoted_manifest(tmp_path: Path) -> None:
    spec = _spec()
    promoted = AgentEntry(
        agent_id="incumbent",
        checkpoint_path=tmp_path / "promoted.pkl",
        cfg=TrainConfig(),
        act_fn=lambda _obs: [],
    )
    with patch(
        "src.artifacts.tournament.unified.incumbent.resolve_promoted_agent",
        return_value=promoted,
    ):
        incumbent = resolve_incumbent(
            spec, campaign="test_campaign", output_root=tmp_path
        )
    assert incumbent is promoted


def test_swap_denied_when_seed_below_perfect(tmp_path: Path) -> None:
    from src.artifacts.tournament.unified.incumbent import (
        swap_incumbent_on_unified_pass,
    )

    verdict = UnifiedLadderVerdict(
        passed=False,
        reason="incumbent_not_defeated",
        stages=(UnifiedStageResult(name="stage2_incumbent", passed=False),),
        challenger_checkpoint=str(tmp_path / "c.pkl"),
        incumbent_swap=False,
    )
    challenger = AgentEntry(
        agent_id="cand",
        checkpoint_path=tmp_path / "c.pkl",
        cfg=TrainConfig(),
        act_fn=lambda _obs: [],
    )
    swapped = swap_incumbent_on_unified_pass(
        TrainConfig(),
        challenger=challenger,
        verdict=verdict,
        campaign="test",
        output_root=tmp_path,
        tournament_output_dir=tmp_path / "tournament",
    )
    assert swapped is False
