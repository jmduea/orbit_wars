"""Incumbent lineage detection for qualifier skip."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.checkpoint_compat import load_checkpoint_payload
from src.artifacts.promotion_manifest import promoted_manifest_path
from src.artifacts.tournament.bracket.state import BracketState


def _normalize_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def parent_checkpoint_path(checkpoint_path: Path) -> str | None:
    payload = load_checkpoint_payload(checkpoint_path)
    if not isinstance(payload, dict):
        return None
    parent = payload.get("parent_checkpoint_path")
    if parent is None:
        return None
    return _normalize_path(str(parent))


def resolve_promoted_incumbent_checkpoint(
    campaign: str,
    output_root: Path,
) -> str | None:
    manifest_path = promoted_manifest_path(output_root / "campaigns" / campaign)
    if not manifest_path.is_file():
        return None
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    checkpoint = payload.get("checkpoint_path")
    if checkpoint is None:
        return None
    return _normalize_path(str(checkpoint))


def qualifier_skip_for_checkpoint(
    checkpoint_path: Path,
    *,
    campaign: str,
    output_root: Path,
    bracket_state: BracketState | None = None,
) -> bool:
    """Return True when checkpoint was trained from the current incumbent."""

    parent = parent_checkpoint_path(checkpoint_path)
    if parent is None:
        return False
    incumbent_path = resolve_promoted_incumbent_checkpoint(campaign, output_root)
    if incumbent_path is not None and parent == incumbent_path:
        return True
    if bracket_state is not None and bracket_state.incumbent_agent_id is not None:
        incumbent_entry = bracket_state.entries.get(bracket_state.incumbent_agent_id)
        if incumbent_entry is not None:
            return parent == _normalize_path(incumbent_entry.checkpoint_path)
    return False
