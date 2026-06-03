"""Sample self-play opponents from the main tournament bracket."""

from __future__ import annotations

import random

from src.artifacts.tournament.bracket.state import BracketState


def sample_bracket_checkpoints(
    state: BracketState,
    *,
    count: int,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    """Return checkpoint paths from main-phase bracket entries."""

    if state.phase != "main":
        return ()
    entries = state.main_phase_entries()
    if not entries:
        return ()
    rng = rng or random.Random()
    paths = [entry.checkpoint_path for entry in entries]
    if count >= len(paths):
        return tuple(paths)
    return tuple(rng.sample(paths, count))
