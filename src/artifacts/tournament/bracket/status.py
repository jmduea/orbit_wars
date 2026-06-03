"""Bracket state summaries for CLI and status polling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.artifacts.tournament.bracket.state import (
    BracketState,
    bracket_state_path,
    load_bracket_state,
)


def summarize_bracket(
    *,
    campaign: str,
    output_root: Path,
) -> dict[str, Any]:
    """Return a compact JSON-serializable bracket summary for agents."""

    path = bracket_state_path(campaign=campaign, output_root=output_root)
    state = load_bracket_state(path)
    main_entries = state.main_phase_entries()
    return {
        "campaign": campaign,
        "state_path": str(path.resolve()),
        "exists": path.is_file(),
        "phase": state.phase,
        "incumbent_crowned": state.incumbent_crowned,
        "incumbent_agent_id": state.incumbent_agent_id,
        "entry_count": len(state.entries),
        "main_entry_count": len(main_entries),
        "weak_config": state.phase == "weak_config",
        "entries": [
            {
                "agent_id": entry.agent_id,
                "qualifier_cleared": entry.qualifier_cleared,
                "lineage_skip": entry.lineage_skip,
                "mu": entry.mu,
                "sigma": entry.sigma,
            }
            for entry in sorted(state.entries.values(), key=lambda e: e.agent_id)
        ],
    }


def bracket_show_payload(
    *,
    campaign: str,
    output_root: Path,
) -> dict[str, Any]:
    """Return full bracket state dict for ``ow eval bracket show``."""

    path = bracket_state_path(campaign=campaign, output_root=output_root)
    state = load_bracket_state(path)
    return {
        "campaign": campaign,
        "state_path": str(path.resolve()),
        "state": state.to_dict(),
    }
