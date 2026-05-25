from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _RunningStats:
    mean: Any
    var: Any
    count: float


class ObservationNormalizer:
    """Legacy running-stat normalizer kept for checkpoint/state compatibility."""

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.clip = float(clip)
        self.eps = float(eps)

    def update(self, batch: object) -> None:
        del batch

    def normalize_batch(self, batch: object) -> object:
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {"clip": self.clip, "eps": self.eps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
