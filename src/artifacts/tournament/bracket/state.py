"""Persistent bracket state for qualifier and main ranking phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.artifacts.run_paths import atomic_write_json
from src.artifacts.tournament.bracket.trueskill import DEFAULT_MU, DEFAULT_SIGMA, Rating

BracketPhase = Literal["qualifier", "main", "weak_config"]


@dataclass(slots=True)
class BracketEntry:
    agent_id: str
    checkpoint_path: str
    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA
    qualifier_cleared: bool = False
    lineage_skip: bool = False

    def rating(self) -> Rating:
        return Rating(mu=self.mu, sigma=self.sigma)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "checkpoint_path": self.checkpoint_path,
            "mu": self.mu,
            "sigma": self.sigma,
            "qualifier_cleared": self.qualifier_cleared,
            "lineage_skip": self.lineage_skip,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BracketEntry:
        return cls(
            agent_id=str(payload["agent_id"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            mu=float(payload.get("mu", DEFAULT_MU)),
            sigma=float(payload.get("sigma", DEFAULT_SIGMA)),
            qualifier_cleared=bool(payload.get("qualifier_cleared", False)),
            lineage_skip=bool(payload.get("lineage_skip", False)),
        )


@dataclass(slots=True)
class BracketState:
    phase: BracketPhase = "qualifier"
    incumbent_crowned: bool = False
    incumbent_agent_id: str | None = None
    entries: dict[str, BracketEntry] = field(default_factory=dict)

    def main_phase_entries(self) -> tuple[BracketEntry, ...]:
        return tuple(
            entry
            for entry in self.entries.values()
            if entry.qualifier_cleared or entry.lineage_skip
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "incumbent_crowned": self.incumbent_crowned,
            "incumbent_agent_id": self.incumbent_agent_id,
            "entries": {
                agent_id: entry.to_dict() for agent_id, entry in self.entries.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BracketState:
        raw_entries = payload.get("entries", {})
        entries: dict[str, BracketEntry] = {}
        if isinstance(raw_entries, dict):
            for agent_id, raw in raw_entries.items():
                if isinstance(raw, dict):
                    entries[str(agent_id)] = BracketEntry.from_dict(raw)
        phase = str(payload.get("phase", "qualifier"))
        if phase not in {"qualifier", "main", "weak_config"}:
            phase = "qualifier"
        return cls(
            phase=phase,  # type: ignore[arg-type]
            incumbent_crowned=bool(payload.get("incumbent_crowned", False)),
            incumbent_agent_id=(
                str(payload["incumbent_agent_id"])
                if payload.get("incumbent_agent_id") is not None
                else None
            ),
            entries=entries,
        )


def bracket_state_path(*, campaign: str, output_root: Path) -> Path:
    return output_root / "campaigns" / campaign / "bracket" / "state.json"


def load_bracket_state(path: Path) -> BracketState:
    if not path.is_file():
        return BracketState()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return BracketState()
    return BracketState.from_dict(payload)


def save_bracket_state(path: Path, state: BracketState) -> None:
    atomic_write_json(path, state.to_dict())


def upsert_entry(state: BracketState, entry: BracketEntry) -> BracketState:
    state.entries[entry.agent_id] = entry
    return state


def mark_qualifier_cleared(
    state: BracketState,
    *,
    agent_id: str,
    crown_incumbent: bool = False,
) -> BracketState:
    entry = state.entries.get(agent_id)
    if entry is None:
        return state
    entry.qualifier_cleared = True
    state.phase = "main"
    if crown_incumbent:
        state.incumbent_crowned = True
        state.incumbent_agent_id = agent_id
    return state


def mark_weak_config(state: BracketState) -> BracketState:
    state.phase = "weak_config"
    return state
