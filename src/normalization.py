from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .features import TurnBatch, candidate_feature_dim, global_feature_dim, self_feature_dim


@dataclass(slots=True)
class RunningMeanStd:
    """Track running feature moments and normalize tensors with them."""

    shape: tuple[int, ...]
    epsilon: float = 1e-4
    mean: torch.Tensor = field(init=False)
    var: torch.Tensor = field(init=False)
    count: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.mean = torch.zeros(self.shape, dtype=torch.float64)
        self.var = torch.ones(self.shape, dtype=torch.float64)
        self.count = torch.tensor(float(self.epsilon), dtype=torch.float64)

    def update(self, values: np.ndarray | torch.Tensor) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64, device=self.mean.device)
        if tensor.numel() == 0:
            return
        tensor = tensor.reshape(-1, *self.shape)
        if tensor.shape[0] == 0:
            return
        batch_mean = tensor.mean(dim=0)
        batch_var = tensor.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(tensor.shape[0]), dtype=torch.float64, device=self.mean.device)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: torch.Tensor) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray | torch.Tensor, clip: float | None = None) -> torch.Tensor:
        tensor = torch.as_tensor(values)
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = torch.sqrt(self.var.to(device=tensor.device, dtype=tensor.dtype) + 1e-8)
        normalized = (tensor - mean) / std
        if clip is not None and clip > 0.0:
            normalized = normalized.clamp(-clip, clip)
        return normalized

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.mean = torch.as_tensor(state_dict["mean"], dtype=torch.float64).clone()
        self.var = torch.as_tensor(state_dict["var"], dtype=torch.float64).clone()
        self.count = torch.as_tensor(state_dict["count"], dtype=torch.float64).clone()


class ObservationNormalizer:
    """Normalize PlanetPolicy observation tensors with candidate-mask-aware stats."""

    def __init__(self, clip: float = 10.0) -> None:
        self.clip = float(clip)
        self.self_features = RunningMeanStd((self_feature_dim(),))
        self.candidate_features = RunningMeanStd((candidate_feature_dim(),))
        self.global_features = RunningMeanStd((global_feature_dim(),))

    def update(self, batch: TurnBatch) -> None:
        if batch.self_features.shape[0] == 0:
            return
        self.self_features.update(batch.self_features)
        self.global_features.update(batch.global_features)
        valid_candidates = batch.candidate_features[batch.candidate_mask]
        if valid_candidates.shape[0] > 0:
            self.candidate_features.update(valid_candidates)

    def normalize_tensors(
        self,
        self_features: np.ndarray | torch.Tensor,
        candidate_features: np.ndarray | torch.Tensor,
        global_features: np.ndarray | torch.Tensor,
        candidate_mask: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_self = self.self_features.normalize(self_features, self.clip).float()
        normalized_candidates = self.candidate_features.normalize(candidate_features, self.clip).float()
        mask = torch.as_tensor(candidate_mask, device=normalized_candidates.device).bool()
        normalized_candidates = torch.where(mask.unsqueeze(-1), normalized_candidates, torch.zeros_like(normalized_candidates))
        normalized_global = self.global_features.normalize(global_features, self.clip).float()
        return normalized_self, normalized_candidates, normalized_global

    def normalize_batch(self, batch: TurnBatch) -> TurnBatch:
        normalized_self, normalized_candidates, normalized_global = self.normalize_tensors(
            batch.self_features,
            batch.candidate_features,
            batch.global_features,
            batch.candidate_mask,
        )
        return TurnBatch(
            self_features=normalized_self.cpu().numpy().astype(np.float32, copy=False),
            candidate_features=normalized_candidates.cpu().numpy().astype(np.float32, copy=False),
            global_features=normalized_global.cpu().numpy().astype(np.float32, copy=False),
            candidate_mask=batch.candidate_mask,
            contexts=batch.contexts,
            state=batch.state,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "self_features": self.self_features.state_dict(),
            "candidate_features": self.candidate_features.state_dict(),
            "global_features": self.global_features.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.clip = float(state_dict.get("clip", self.clip))
        self.self_features.load_state_dict(state_dict["self_features"])
        self.candidate_features.load_state_dict(state_dict["candidate_features"])
        self.global_features.load_state_dict(state_dict["global_features"])
