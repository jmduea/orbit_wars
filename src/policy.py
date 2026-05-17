from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass(slots=True)
class PolicyOutput:
    target_logits: torch.Tensor
    ship_logits: torch.Tensor
    value: torch.Tensor


class PlanetPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        ship_bucket_count: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.ship_bucket_count = ship_bucket_count
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ship_bucket_count),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PolicyOutput:
        self_hidden = self.self_encoder(self_features)
        global_hidden = self.global_encoder(global_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(
            -1, self.candidate_count, -1
        )
        joint = torch.cat([expanded_self, expanded_global, candidate_hidden], dim=-1)
        target_logits = self.target_head(joint).squeeze(-1)
        target_logits = target_logits.masked_fill(
            ~candidate_mask, torch.finfo(target_logits.dtype).min
        )
        ship_logits = self.ship_head(joint)
        pooled_candidates = candidate_hidden.mean(dim=1)
        value = self.value_head(
            torch.cat([self_hidden, global_hidden, pooled_candidates], dim=-1)
        ).squeeze(-1)
        return PolicyOutput(
            target_logits=target_logits, ship_logits=ship_logits, value=value
        )


class AttentionPlanetPolicy(nn.Module):
    """Experimental attention policy with the same action/value interface as PlanetPolicy.

    The policy keeps the current target-index and ship-bucket action semantics: it
    returns one target logit per candidate, one ship-bucket distribution per
    candidate target, and one scalar value per source planet decision row.
    """

    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        ship_bucket_count: int,
        hidden_size: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size % attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by attention_heads ({attention_heads})."
            )
        self.candidate_count = candidate_count
        self.ship_bucket_count = ship_bucket_count
        self.attention_heads = attention_heads
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.context_query = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.candidate_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.target_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ship_bucket_count),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PolicyOutput:
        candidate_mask = candidate_mask.bool()
        self_hidden = self.self_encoder(self_features)
        global_hidden = self.global_encoder(global_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        key_padding_mask = _safe_key_padding_mask(candidate_mask)

        attended_candidates, _ = self.candidate_attention(
            query=candidate_hidden,
            key=candidate_hidden,
            value=candidate_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_candidates = self.target_norm(candidate_hidden + attended_candidates)

        context_query = self.context_query(
            torch.cat([self_hidden, global_hidden], dim=-1)
        ).unsqueeze(1)
        attended_context, _ = self.context_attention(
            query=context_query,
            key=attended_candidates,
            value=attended_candidates,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_context = self.context_norm(context_query + attended_context).squeeze(
            1
        )

        expanded_self = self_hidden.unsqueeze(1).expand(-1, self.candidate_count, -1)
        expanded_global = global_hidden.unsqueeze(1).expand(
            -1, self.candidate_count, -1
        )
        expanded_context = attended_context.unsqueeze(1).expand(
            -1, self.candidate_count, -1
        )
        target_input = torch.cat(
            [expanded_self, expanded_global, expanded_context, attended_candidates],
            dim=-1,
        )
        target_logits = self.target_head(target_input).squeeze(-1)
        target_logits = target_logits.masked_fill(
            ~candidate_mask, torch.finfo(target_logits.dtype).min
        )

        pooled_candidates = _masked_mean(attended_candidates, candidate_mask)
        ship_logits = self.ship_head(target_input)
        value = self.value_head(
            torch.cat(
                [self_hidden, global_hidden, attended_context, pooled_candidates],
                dim=-1,
            )
        ).squeeze(-1)
        return PolicyOutput(
            target_logits=target_logits, ship_logits=ship_logits, value=value
        )


def build_policy(
    *,
    architecture: Literal["mlp", "attention"] | str,
    self_dim: int,
    candidate_dim: int,
    global_dim: int,
    candidate_count: int,
    ship_bucket_count: int,
    hidden_size: int = 128,
    attention_heads: int = 4,
) -> nn.Module:
    normalized_architecture = architecture.strip().lower()
    if normalized_architecture == "mlp":
        return PlanetPolicy(
            self_dim=self_dim,
            candidate_dim=candidate_dim,
            global_dim=global_dim,
            candidate_count=candidate_count,
            ship_bucket_count=ship_bucket_count,
            hidden_size=hidden_size,
        )
    if normalized_architecture in {"attention", "transformer"}:
        return AttentionPlanetPolicy(
            self_dim=self_dim,
            candidate_dim=candidate_dim,
            global_dim=global_dim,
            candidate_count=candidate_count,
            ship_bucket_count=ship_bucket_count,
            hidden_size=hidden_size,
            attention_heads=attention_heads,
        )
    raise ValueError(
        f"Unsupported model architecture '{architecture}'. Expected 'mlp', 'attention', or 'transformer'."
    )


def _safe_key_padding_mask(candidate_mask: torch.Tensor) -> torch.Tensor:
    """Return a MultiheadAttention mask while avoiding rows with every key masked."""

    key_padding_mask = ~candidate_mask
    all_masked = key_padding_mask.all(dim=-1)
    if all_masked.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked, 0] = False
    return key_padding_mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    summed = (values * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts
