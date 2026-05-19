from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


OPPONENT_LATEST = 0
OPPONENT_HISTORICAL = 1
OPPONENT_SCRIPTED_SNIPER = 2
OPPONENT_RANDOM = 3


@dataclass(slots=True)
class OpponentMixturePhase:
    start_update: int = 0
    end_update: int = -1
    weights: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0


@dataclass(slots=True)
class OpponentRegistryConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "latest": 1.0,
            "historical": 0.0,
            "scripted_sniper": 0.0,
            "random": 0.0,
        }
    )
    temperature: float = 1.0
    curriculum: list[dict[str, Any]] = field(default_factory=list)


class OpponentRegistry:
    """Backend-agnostic registry for opponent families and curriculum mixtures."""

    def __init__(self, cfg: OpponentRegistryConfig) -> None:
        self.cfg = cfg

    def phase_for_update(self, update: int) -> OpponentMixturePhase | None:
        for raw in self.cfg.curriculum:
            start = int(raw.get("start_update", 0))
            end = int(raw.get("end_update", -1))
            if update < start:
                continue
            if end >= 0 and update > end:
                continue
            return OpponentMixturePhase(
                start_update=start,
                end_update=end,
                weights={
                    str(k): float(v) for k, v in dict(raw.get("weights", {})).items()
                },
                temperature=float(raw.get("temperature", self.cfg.temperature)),
            )
        return None

    def _weights_and_temperature(
        self, update: int | None
    ) -> tuple[dict[str, float], float]:
        if update is not None:
            phase = self.phase_for_update(update)
            if phase is not None:
                merged = dict(self.cfg.weights)
                merged.update(phase.weights)
                return merged, phase.temperature
        return dict(self.cfg.weights), float(self.cfg.temperature)

    def ids_and_probs(self, update: int | None = None) -> tuple[list[int], list[float]]:
        weights, temperature = self._weights_and_temperature(update)
        items = [
            (OPPONENT_LATEST, weights.get("latest", 0.0)),
            (OPPONENT_HISTORICAL, weights.get("historical", 0.0)),
            (OPPONENT_SCRIPTED_SNIPER, weights.get("scripted_sniper", 0.0)),
            (OPPONENT_RANDOM, weights.get("random", 0.0)),
        ]
        logits = []
        ids = []
        for opponent_id, weight in items:
            if weight <= 0.0:
                continue
            ids.append(opponent_id)
            logits.append(float(np.log(float(weight) + 1e-12)))
        if not ids:
            return [OPPONENT_LATEST], [1.0]
        temp = max(float(temperature), 1e-6)
        raw = np.asarray(logits, dtype=np.float32) / temp
        raw = raw - np.max(raw)
        probs = np.exp(raw)
        probs = probs / np.sum(probs)
        return ids, [float(x) for x in probs]


def sample_opponent_type_ids_jax(
    key: jax.Array,
    env_count: int,
    player_count: int,
    *,
    ids: jax.Array,
    probs: jax.Array,
) -> jax.Array:
    """Sample opponent IDs per [env, player] slot from a categorical mixture."""

    logits = jnp.log(jnp.maximum(probs, 1e-12))
    sampled = jax.random.categorical(
        key,
        logits[None, None, :],
        axis=-1,
        shape=(env_count, player_count),
    )
    return ids[sampled]
