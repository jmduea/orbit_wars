"""Kaggle-style tournament bracket: ranking, qualifier, and training integration."""

from __future__ import annotations

from src.artifacts.tournament.bracket.lineage import (
    parent_checkpoint_path,
    qualifier_skip_for_checkpoint,
    resolve_promoted_incumbent_checkpoint,
)
from src.artifacts.tournament.bracket.qualifier import (
    QUALIFIER_FLOOR,
    QUALIFIER_OPPONENTS,
    QualifierVerdict,
    evaluate_qualifier_scores,
    qualifier_floors,
)
from src.artifacts.tournament.bracket.scheduler import (
    apply_head_to_head_outcome,
    iter_round_robin_pairs,
    queue_round_robin_matches,
)
from src.artifacts.tournament.bracket.self_play import sample_bracket_checkpoints
from src.artifacts.tournament.bracket.status import bracket_show_payload, summarize_bracket
from src.artifacts.tournament.bracket.state import (
    BracketEntry,
    BracketState,
    bracket_state_path,
    load_bracket_state,
    mark_qualifier_cleared,
    mark_weak_config,
    save_bracket_state,
    upsert_entry,
)
from src.artifacts.tournament.bracket.trueskill import (
    DEFAULT_BETA,
    DEFAULT_MU,
    DEFAULT_SIGMA,
    Rating,
    update_draw,
    update_win,
)

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_MU",
    "DEFAULT_SIGMA",
    "QUALIFIER_FLOOR",
    "QUALIFIER_OPPONENTS",
    "BracketEntry",
    "BracketState",
    "QualifierVerdict",
    "apply_head_to_head_outcome",
    "bracket_show_payload",
    "iter_round_robin_pairs",
    "queue_round_robin_matches",
    "summarize_bracket",
    "Rating",
    "bracket_state_path",
    "evaluate_qualifier_scores",
    "load_bracket_state",
    "mark_qualifier_cleared",
    "mark_weak_config",
    "parent_checkpoint_path",
    "qualifier_floors",
    "qualifier_skip_for_checkpoint",
    "resolve_promoted_incumbent_checkpoint",
    "sample_bracket_checkpoints",
    "save_bracket_state",
    "update_draw",
    "update_win",
    "upsert_entry",
]
