from __future__ import annotations

import math
import random
from collections import deque, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from .config import TrainConfig
from .features import encode_turn, ship_count_for_bucket
from .jax_policy import build_jax_policy
from .jax_policy import sample_actions as sample_jax_actions
from .normalization import ObservationNormalizer

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]: ...


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
            nearest = min(
                targets,
                key=lambda target: math.hypot(source.x - target.x, source.y - target.y),
            )
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
    def __init__(
        self, cfg: TrainConfig, device: str = "auto", deterministic: bool = True
    ) -> None:

        self.cfg = cfg
        self.device = device
        self.rng = jax.random.PRNGKey(0)
        self.deterministic = deterministic
        self.policy = build_jax_policy(cfg=self.cfg)
        self.params: dict[str, Any] | None = None
        self.normalizer: ObservationNormalizer | None = None

    def sync_from(
        self,
        source_params: dict[str, Any],
        normalizer: ObservationNormalizer | None = None,
    ) -> None:
        self.params = jax.tree.map(lambda x: jnp.asarray(x), source_params)
        self.normalizer = clone_normalizer(normalizer)

    def act(self, observation: Any) -> list[list[float | int]]:
        batch = encode_turn(observation, self.cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        policy_batch = (
            self.normalizer.normalize_batch(batch)
            if self.normalizer is not None
            else batch
        )
        if self.params is None:
            raise ValueError("SelfPlayOpponent params are not initialized; call sync_from first.")
        outputs = self.policy.apply(
            {"params": self.params},
            jnp.asarray(policy_batch.self_features),
            jnp.asarray(policy_batch.candidate_features),
            jnp.asarray(policy_batch.global_features),
            jnp.asarray(policy_batch.candidate_mask).astype(bool),
        )
        self.rng, step_key = jax.random.split(self.rng)
        target_indices, ship_buckets, _logp, _entropy = sample_jax_actions(
            step_key, outputs, deterministic=self.deterministic
        )
        target_indices = jax.device_get(target_indices)
        ship_buckets = jax.device_get(ship_buckets)
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
            ships = ship_count_for_bucket(
                context.source_ships, bucket_idx, self.cfg.env.ship_bucket_count
            )
            if ships <= 0:
                continue
            moves.append(
                [context.source_id, float(context.target_angles[target_idx]), ships]
            )
        return moves


@dataclass(slots=True)
class SnapshotMetadata:
    snapshot_id: int
    update: int
    source: str


@dataclass(slots=True)
class OpponentSelection:
    policy: OpponentPolicy
    metadata: dict[str, Any]


@dataclass(slots=True)
class HistoricalSnapshot:
    metadata: SnapshotMetadata
    opponent: SelfPlayOpponent


class SelfPlayOpponentPool:
    """Episode-level opponent sampler with bot baselines plus policy snapshots."""

    def __init__(self, cfg: TrainConfig, device: str = "auto") -> None:
        self.cfg = cfg
        self.device = device
        self.rng = jax.random.PRNGKey(0)
        self.random_bot = KaggleRandomOpponent()
        self.sniper_bot = SniperOpponent()
        self.latest = SelfPlayOpponent(
            cfg, device=device, deterministic=cfg.self_play_deterministic
        )
        self.history: deque[HistoricalSnapshot] = deque(
            maxlen=max(0, cfg.self_play_pool_size)
        )
        self.latest_metadata = SnapshotMetadata(
            snapshot_id=0, update=0, source="latest"
        )
        self._next_snapshot_id = 1

    def sync_from(
        self,
        source_policy: dict[str, Any],
        normalizer: ObservationNormalizer | None = None,
        update: int = 0,
    ) -> None:
        self.latest.sync_from(source_policy, normalizer)
        self.latest_metadata = SnapshotMetadata(
            snapshot_id=0, update=update, source="latest"
        )

    def add_snapshot(
        self,
        source_policy: dict[str, Any],
        normalizer: ObservationNormalizer | None = None,
        update: int = 0,
    ) -> None:
        if self.history.maxlen == 0:
            return
        snapshot = SelfPlayOpponent(
            self.cfg, device=self.device, deterministic=self.cfg.self_play_deterministic
        )
        snapshot.sync_from(source_policy, normalizer)
        metadata = SnapshotMetadata(
            snapshot_id=self._next_snapshot_id, update=update, source="historical"
        )
        self._next_snapshot_id += 1
        self.history.append(HistoricalSnapshot(metadata=metadata, opponent=snapshot))

    def sample_selection(self) -> OpponentSelection:
        mode = self.cfg.multi_opponent_mode.strip().lower()
        if mode == "shared_current":
            return OpponentSelection(
                policy=self.latest, metadata=dataclass_to_dict(self.latest_metadata)
            )
        if mode == "sampled_pool":
            choices = list(self.history)
            if not choices:
                return OpponentSelection(
                    policy=self.latest, metadata=dataclass_to_dict(self.latest_metadata)
                )
            snapshot = random.choice(choices)
            return OpponentSelection(
                policy=snapshot.opponent, metadata=dataclass_to_dict(snapshot.metadata)
            )
        if mode != "mixed":
            raise ValueError(
                "multi_opponent_mode must be one of shared_current, sampled_pool, or mixed; "
                f"got {self.cfg.multi_opponent_mode!r}."
            )
        latest_probability = min(max(self.cfg.self_play_latest_probability, 0.0), 1.0)
        if random.random() < latest_probability:
            return OpponentSelection(
                policy=self.latest, metadata=dataclass_to_dict(self.latest_metadata)
            )
        choices: list[OpponentSelection] = [
            OpponentSelection(
                self.random_bot, {"snapshot_id": -1, "update": 0, "source": "random"}
            ),
            OpponentSelection(
                self.sniper_bot, {"snapshot_id": -2, "update": 0, "source": "sniper"}
            ),
        ]
        choices.extend(
            OpponentSelection(snapshot.opponent, dataclass_to_dict(snapshot.metadata))
            for snapshot in self.history
        )
        return random.choice(choices)

    def sample_opponent(self) -> OpponentPolicy:
        return self.sample_selection().policy

    def sample_opponents(self, count: int) -> list[OpponentSelection]:
        if count <= 0:
            return []
        if self.cfg.multi_opponent_mode.strip().lower() == "shared_current":
            shared = OpponentSelection(
                policy=self.latest, metadata=dataclass_to_dict(self.latest_metadata)
            )
            return [shared for _ in range(count)]
        return [self.sample_selection() for _ in range(count)]

    def act(self, observation: Any) -> list[list[float | int]]:
        return self.sample_opponent().act(observation)

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "latest": dataclass_to_dict(self.latest_metadata),
            "pool_size": len(self.history),
            "max_pool_size": self.history.maxlen or 0,
            "snapshots": [
                dataclass_to_dict(snapshot.metadata) for snapshot in self.history
            ],
            "next_snapshot_id": self._next_snapshot_id,
        }


def clone_normalizer(
    normalizer: ObservationNormalizer | None,
) -> ObservationNormalizer | None:
    if normalizer is None:
        return None
    cloned = ObservationNormalizer(clip=normalizer.clip)
    cloned.load_state_dict(deepcopy(normalizer.state_dict()))
    return cloned


def dataclass_to_dict(metadata: SnapshotMetadata) -> dict[str, int | str]:
    return {
        "snapshot_id": metadata.snapshot_id,
        "update": metadata.update,
        "source": metadata.source,
    }


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: str | None = None,
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
        return SelfPlayOpponent(
            cfg, device=device, deterministic=cfg.self_play_deterministic
        )
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
