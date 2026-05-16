from __future__ import annotations

import math
import random
from collections import deque, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .config import TrainConfig
from .features import encode_turn, ship_count_for_bucket
from .normalization import ObservationNormalizer
from .policy import build_policy
from .ppo import sample_actions

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]:
        ...


class SniperOpponent:
    def act(self, observation: Any) -> list[list[float | int]]:
        moves: list[list[float | int]] = []
        player = obs_get(observation, "player", 0)
        raw_planets = obs_get(observation, "planets", [])
        planets = [Planet(*planet) for planet in raw_planets]
        my_planets = [planet for planet in planets if planet.owner == player]
        targets = [planet for planet in planets if planet.owner != player]
        if not targets:
            return moves
        for source in my_planets:
            nearest = min(targets, key=lambda target: math.hypot(source.x - target.x, source.y - target.y))
            ships_needed = max(nearest.ships + 1, 20)
            if source.ships < ships_needed:
                continue
            angle = math.atan2(nearest.y - source.y, nearest.x - source.x)
            moves.append([source.id, angle, ships_needed])
        return moves


class KaggleRandomOpponent:
    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self._agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        payload = {
            "player": obs_get(observation, "player", 0),
            "planets": list(obs_get(observation, "planets", [])),
        }
        return list(self._agent(payload))


class SelfPlayOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device, deterministic: bool = True) -> None:
        from .features import (
            candidate_feature_dim,
            global_feature_dim,
            self_feature_dim,
        )

        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policy = build_policy(
            architecture=cfg.model.architecture,
            self_dim=self_feature_dim(),
            candidate_dim=candidate_feature_dim(),
            global_dim=global_feature_dim(),
            candidate_count=cfg.env.candidate_count,
            ship_bucket_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
            attention_heads=cfg.model.attention_heads,
        ).to(device)
        self.policy.eval()
        self.normalizer: ObservationNormalizer | None = None

    def sync_from(self, source_policy: torch.nn.Module, normalizer: ObservationNormalizer | None = None) -> None:
        self.policy.load_state_dict(source_policy.state_dict())
        self.policy.eval()
        self.normalizer = clone_normalizer(normalizer)

    def act(self, observation: Any) -> list[list[float | int]]:
        batch = encode_turn(observation, self.cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        policy_batch = self.normalizer.normalize_batch(batch) if self.normalizer is not None else batch
        with torch.inference_mode():
            outputs = self.policy(
                torch.from_numpy(policy_batch.self_features).to(self.device),
                torch.from_numpy(policy_batch.candidate_features).to(self.device),
                torch.from_numpy(policy_batch.global_features).to(self.device),
                torch.from_numpy(policy_batch.candidate_mask).to(self.device).bool(),
            )
            sampled = sample_actions(outputs, deterministic=self.deterministic)
        target_indices = sampled.target_index.detach().cpu().numpy()
        ship_buckets = sampled.ship_bucket.detach().cpu().numpy()
        moves: list[list[float | int]] = []
        for row_idx, context in enumerate(batch.contexts):
            target_idx = int(target_indices[row_idx])
            bucket_idx = int(ship_buckets[row_idx])
            if target_idx == 0 or bucket_idx == 0:
                continue
            if target_idx >= len(context.candidate_ids):
                continue
            if not context.candidate_mask[target_idx]:
                continue
            ships = ship_count_for_bucket(context.source_ships, bucket_idx, self.cfg.env.ship_bucket_count)
            if ships <= 0:
                continue
            moves.append([context.source_id, float(context.target_angles[target_idx]), ships])
        return moves


@dataclass(slots=True)
class SnapshotMetadata:
    snapshot_id: int
    update: int
    source: str


@dataclass(slots=True)
class HistoricalSnapshot:
    metadata: SnapshotMetadata
    opponent: SelfPlayOpponent


class SelfPlayOpponentPool:
    """Episode-level opponent sampler with bot baselines plus policy snapshots."""

    def __init__(self, cfg: TrainConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.random_bot = KaggleRandomOpponent()
        self.sniper_bot = SniperOpponent()
        self.latest = SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
        self.history: deque[HistoricalSnapshot] = deque(maxlen=max(0, cfg.self_play_pool_size))
        self.latest_metadata = SnapshotMetadata(snapshot_id=0, update=0, source="latest")
        self._next_snapshot_id = 1

    def sync_from(
        self,
        source_policy: torch.nn.Module,
        normalizer: ObservationNormalizer | None = None,
        update: int = 0,
    ) -> None:
        self.latest.sync_from(source_policy, normalizer)
        self.latest_metadata = SnapshotMetadata(snapshot_id=0, update=update, source="latest")

    def add_snapshot(
        self,
        source_policy: torch.nn.Module,
        normalizer: ObservationNormalizer | None = None,
        update: int = 0,
    ) -> None:
        if self.history.maxlen == 0:
            return
        snapshot = SelfPlayOpponent(self.cfg, device=self.device, deterministic=self.cfg.self_play_deterministic)
        snapshot.sync_from(source_policy, normalizer)
        metadata = SnapshotMetadata(snapshot_id=self._next_snapshot_id, update=update, source="historical")
        self._next_snapshot_id += 1
        self.history.append(HistoricalSnapshot(metadata=metadata, opponent=snapshot))

    def sample_opponent(self) -> OpponentPolicy:
        latest_probability = min(max(self.cfg.self_play_latest_probability, 0.0), 1.0)
        if random.random() < latest_probability:
            return self.latest
        choices: list[OpponentPolicy] = [self.random_bot, self.sniper_bot]
        choices.extend(snapshot.opponent for snapshot in self.history)
        return random.choice(choices)

    def act(self, observation: Any) -> list[list[float | int]]:
        return self.sample_opponent().act(observation)

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "latest": dataclass_to_dict(self.latest_metadata),
            "pool_size": len(self.history),
            "max_pool_size": self.history.maxlen or 0,
            "snapshots": [dataclass_to_dict(snapshot.metadata) for snapshot in self.history],
            "next_snapshot_id": self._next_snapshot_id,
        }


def clone_normalizer(normalizer: ObservationNormalizer | None) -> ObservationNormalizer | None:
    if normalizer is None:
        return None
    cloned = ObservationNormalizer(clip=normalizer.clip)
    cloned.load_state_dict(deepcopy(normalizer.state_dict()))
    return cloned


def dataclass_to_dict(metadata: SnapshotMetadata) -> dict[str, int | str]:
    return {"snapshot_id": metadata.snapshot_id, "update": metadata.update, "source": metadata.source}


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> OpponentPolicy:
    if name == "sniper":
        return SniperOpponent()
    if name == "random":
        return KaggleRandomOpponent()
    if name == "self":
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for self opponent")
        if cfg.self_play_enabled:
            return SelfPlayOpponentPool(cfg, device=device)
        return SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
