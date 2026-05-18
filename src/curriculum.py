from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


METRIC_KEYS = (
    "win_rate_2p",
    "first_place_rate_4p",
    "survival_time",
    "score_share",
    "kl_stability",
)


@dataclass(slots=True)
class CurriculumPhase:
    id: str
    promote_if: dict[str, float] = field(default_factory=dict)
    demote_if: dict[str, float] = field(default_factory=dict)
    min_dwell_updates: int = 1
    cooldown_updates: int = 0
    opponent_mix_weights: dict[str, float] = field(default_factory=dict)
    format_weights: dict[str, float] = field(default_factory=dict)
    reward_shaping: dict[str, float] = field(default_factory=dict)


class CurriculumController:
    """Metric-driven curriculum controller with hysteresis and cooldown."""

    def __init__(self, phases: list[dict[str, Any]]) -> None:
        self.phases = [self._parse_phase(raw, i) for i, raw in enumerate(phases)]
        if not self.phases:
            self.phases = [CurriculumPhase(id="default")]
        self.phase_index = 0
        self.phase_start_update = 1
        self.cooldown_until = 0

    @property
    def phase(self) -> CurriculumPhase:
        return self.phases[self.phase_index]

    def current_phase_id(self) -> str:
        return self.phase.id

    def apply(self, cfg: Any) -> None:
        phase = self.phase
        if phase.opponent_mix_weights:
            cfg.opponent_mix.weights = dict(phase.opponent_mix_weights)
        if phase.reward_shaping:
            for key, value in phase.reward_shaping.items():
                if hasattr(cfg.env, key):
                    setattr(cfg.env, key, float(value))

    def current_format_weights(self) -> dict[int, float]:
        weights = self.phase.format_weights or {"2": 1.0, "4": 1.0}
        out: dict[int, float] = {}
        for key, value in weights.items():
            out[int(key)] = float(value)
        return out

    def update(self, update_idx: int, metrics: dict[str, float]) -> dict[str, Any] | None:
        if update_idx < self.cooldown_until:
            return None
        phase = self.phase
        dwell = update_idx - self.phase_start_update + 1
        if dwell < max(int(phase.min_dwell_updates), 1):
            return None
        next_index = self.phase_index
        reason = ""
        if self._all_ge(metrics, phase.promote_if) and self.phase_index < len(self.phases) - 1:
            next_index = self.phase_index + 1
            reason = "promote"
        elif self._any_lt(metrics, phase.demote_if) and self.phase_index > 0:
            next_index = self.phase_index - 1
            reason = "demote"
        if next_index == self.phase_index:
            return None
        prev = self.phase
        self.phase_index = next_index
        self.phase_start_update = update_idx
        self.cooldown_until = update_idx + max(int(prev.cooldown_updates), 0)
        return {
            "update": update_idx,
            "from_phase": prev.id,
            "to_phase": self.phase.id,
            "reason": reason,
        }

    def _all_ge(self, metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
        return all(float(metrics.get(k, float("-inf"))) >= float(v) for k, v in thresholds.items())

    def _any_lt(self, metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
        return any(float(metrics.get(k, float("inf"))) < float(v) for k, v in thresholds.items())

    def _parse_phase(self, raw: dict[str, Any], index: int) -> CurriculumPhase:
        return CurriculumPhase(
            id=str(raw.get("id", f"phase_{index}")),
            promote_if={str(k): float(v) for k, v in dict(raw.get("promote_if", {})).items()},
            demote_if={str(k): float(v) for k, v in dict(raw.get("demote_if", {})).items()},
            min_dwell_updates=int(raw.get("min_dwell_updates", 1)),
            cooldown_updates=int(raw.get("cooldown_updates", 0)),
            opponent_mix_weights={
                str(k): float(v) for k, v in dict(raw.get("opponent_mix_weights", {})).items()
            },
            format_weights={str(k): float(v) for k, v in dict(raw.get("format_weights", {})).items()},
            reward_shaping={str(k): float(v) for k, v in dict(raw.get("reward_shaping", {})).items()},
        )
