from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import TurnBatch


@dataclass(slots=True)
class _RunningStats:
    mean: np.ndarray
    var: np.ndarray
    count: float


class ObservationNormalizer:
    """Running-stat normalizer for model input features."""

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.clip = float(clip)
        self.eps = float(eps)
        self._self: _RunningStats | None = None
        self._candidate: _RunningStats | None = None
        self._global: _RunningStats | None = None

    def update(self, batch: TurnBatch) -> None:
        self._self = self._update_stats(self._self, batch.self_features)
        self._candidate = self._update_stats(self._candidate, batch.candidate_features)
        self._global = self._update_stats(self._global, batch.global_features)

    def normalize_batch(self, batch: TurnBatch) -> TurnBatch:
        return TurnBatch(
            self_features=self._normalize(batch.self_features, self._self),
            candidate_features=self._normalize(batch.candidate_features, self._candidate),
            global_features=self._normalize(batch.global_features, self._global),
            candidate_mask=batch.candidate_mask,
            contexts=batch.contexts,
            state=batch.state,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "eps": self.eps,
            "self": self._stats_to_dict(self._self),
            "candidate": self._stats_to_dict(self._candidate),
            "global": self._stats_to_dict(self._global),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        self._self = self._stats_from_dict(state.get("self"))
        self._candidate = self._stats_from_dict(state.get("candidate"))
        self._global = self._stats_from_dict(state.get("global"))

    def _normalize(self, values: np.ndarray, stats: _RunningStats | None) -> np.ndarray:
        if stats is None or values.size == 0:
            return values
        norm = (values - stats.mean) / np.sqrt(stats.var + self.eps)
        if self.clip > 0:
            norm = np.clip(norm, -self.clip, self.clip)
        return norm.astype(np.float32, copy=False)

    def _update_stats(
        self, current: _RunningStats | None, values: np.ndarray
    ) -> _RunningStats | None:
        if values.size == 0:
            return current
        sample_count = int(values.shape[0])
        reduce_axes = (0,) if values.ndim == 2 else (0, 1)
        batch_mean = values.mean(axis=reduce_axes)
        batch_var = values.var(axis=reduce_axes)
        if current is None:
            return _RunningStats(
                mean=batch_mean.astype(np.float32),
                var=np.maximum(batch_var, self.eps).astype(np.float32),
                count=float(sample_count),
            )

        delta = batch_mean - current.mean
        total_count = current.count + sample_count
        new_mean = current.mean + delta * (sample_count / total_count)
        m_a = current.var * current.count
        m_b = batch_var * sample_count
        m2 = m_a + m_b + (delta**2) * current.count * sample_count / total_count
        new_var = np.maximum(m2 / total_count, self.eps)
        return _RunningStats(
            mean=new_mean.astype(np.float32),
            var=new_var.astype(np.float32),
            count=float(total_count),
        )

    @staticmethod
    def _stats_to_dict(stats: _RunningStats | None) -> dict[str, Any] | None:
        if stats is None:
            return None
        return {
            "mean": stats.mean.astype(np.float32),
            "var": stats.var.astype(np.float32),
            "count": float(stats.count),
        }

    @staticmethod
    def _stats_from_dict(payload: dict[str, Any] | None) -> _RunningStats | None:
        if not payload:
            return None
        return _RunningStats(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            var=np.asarray(payload["var"], dtype=np.float32),
            count=float(payload["count"]),
        )
