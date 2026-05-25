from __future__ import annotations

import math
import random
from collections import deque, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.game.types import parse_observation
from src.features import FeatureExtractor
from src.jax.submission_runtime import (
    batch_game,
    batch_turn,
    jax_game_from_observation,
    moves_from_jax_action,
    select_runtime_shielded_policy_actions,
)
from src.jax.policy import build_jax_policy
from src.features.normalization import ObservationNormalizer
from src.game.trajectory_shield import (
    filter_moves_with_trajectory_shield,
    is_trajectory_safe_for_launch,
)

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
DEFAULT_RUNTIME_ENV = TrainConfig().task


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]: ...


class SniperOpponent:
    def act(self, observation: Any) -> list[list[float | int]]:
        moves: list[list[float | int]] = []
        state = parse_observation(observation)
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
            if not is_trajectory_safe_for_launch(
                state,
                int(source.id),
                int(nearest.id),
                angle,
                ships_needed,
                DEFAULT_RUNTIME_ENV,
            ):
                continue
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
        state = parse_observation(observation)
        return filter_moves_with_trajectory_shield(
            list(self._agent(payload)), state, DEFAULT_RUNTIME_ENV
        )


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
        self._feature_extractor = FeatureExtractor(cfg.task)

    def sync_from(
        self,
        source_params: dict[str, Any],
        normalizer: ObservationNormalizer | None = None,
    ) -> None:
        self.params = jax.tree.map(lambda x: jnp.asarray(x), source_params)
        self.normalizer = clone_normalizer(normalizer)

    def act(self, observation: Any) -> list[list[float | int]]:
        if self.params is None:
            raise ValueError("SelfPlayOpponent params are not initialized; call sync_from first.")
        extracted = self._feature_extractor.extract(
            observation,
            max_fleet_slots=int(self.cfg.task.max_fleets),
        )
        game = extracted.game
        batch = extracted.batch
        self.rng, step_key = jax.random.split(self.rng)
        action = select_runtime_shielded_policy_actions(
            step_key,
            self.policy,
            {"params": self.params},
            batch_game(game),
            batch_turn(batch),
            self.cfg,
            deterministic=self.deterministic,
        )
        return moves_from_jax_action(action)


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
            cfg, device=device, deterministic=cfg.opponents.self_play.deterministic
        )
        self.history: deque[HistoricalSnapshot] = deque(
            maxlen=max(0, cfg.opponents.snapshot.pool_size)
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
            self.cfg, device=self.device, deterministic=self.cfg.opponents.self_play.deterministic
        )
        snapshot.sync_from(source_policy, normalizer)
        metadata = SnapshotMetadata(
            snapshot_id=self._next_snapshot_id, update=update, source="historical"
        )
        self._next_snapshot_id += 1
        self.history.append(HistoricalSnapshot(metadata=metadata, opponent=snapshot))

    def sample_selection(self) -> OpponentSelection:
        mode = self.cfg.opponents.mode.multi_opponent_mode.strip().lower()
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
                f"got {self.cfg.opponents.mode.multi_opponent_mode!r}."
            )
        latest_probability = min(
            max(self.cfg.opponents.mix.weights.get("latest", 0.0), 0.0), 1.0
        )
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
        if self.cfg.opponents.mode.multi_opponent_mode.strip().lower() == "shared_current":
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
        if cfg.opponents.self_play.enabled:
            return SelfPlayOpponentPool(cfg, device=device)
        return SelfPlayOpponent(
            cfg, device=device, deterministic=cfg.opponents.self_play.deterministic
        )
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
