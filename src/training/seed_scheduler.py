from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class SeedScheduleConfig:
    reseed_every_updates: int = 0
    reseed_on_plateau: bool = False
    plateau_metric: str = "episode_reward_mean"
    plateau_window: int = 10
    plateau_delta: float = 0.0
    heldout_eval_seed_set: list[int] = field(default_factory=list)


@dataclass(slots=True)
class SeedEvent:
    update: int
    old_seed: int
    new_seed: int
    reason: str
    policy: str


def resolve_reseed_every_updates(*, configured: int, total_updates: int) -> int:
    """Map ``training.reseed_every_updates`` to an effective periodic interval.

    ``0`` disables periodic reseed. ``-1`` auto-scales to ``max(25, total_updates // 10)``.
    """

    if int(configured) == -1:
        return max(25, int(total_updates) // 10)
    return int(configured)


class SeedScheduler:
    """Adaptive seed scheduling utility for training resets and RNG key updates."""

    def __init__(self, base_seed: int, cfg: SeedScheduleConfig) -> None:
        self._next_seed = int(base_seed)
        self._cfg = cfg
        self._rng = random.Random(base_seed)
        self._recent_metrics: deque[float] = deque(maxlen=max(cfg.plateau_window, 1))
        self._pool = list(cfg.heldout_eval_seed_set)
        self._pool_index = 0

    @property
    def next_seed(self) -> int:
        return self._next_seed

    def update_metric(self, metric_value: float) -> None:
        self._recent_metrics.append(float(metric_value))

    def should_reseed(self, update: int, *, force: bool = False) -> tuple[bool, str]:
        if force:
            return True, "forced"
        if self._cfg.reseed_every_updates > 0 and update % self._cfg.reseed_every_updates == 0:
            return True, "periodic"
        if self._cfg.reseed_on_plateau and self._is_plateau():
            return True, "plateau"
        return False, ""

    def next_seed_policy(self, update: int) -> str:
        if self._pool:
            return "shuffled_pool"
        if self._cfg.reseed_on_plateau and self._is_plateau():
            return "random_jump"
        if self._cfg.reseed_every_updates > 0 and update % self._cfg.reseed_every_updates == 0:
            return "random_jump"
        return "incremental"

    def reseed(self, update: int, reason: str, policy: str | None = None) -> SeedEvent:
        chosen_policy = policy or self.next_seed_policy(update)
        old_seed = self._next_seed
        if chosen_policy == "random_jump":
            self._next_seed = self._rng.randint(1, 2**31 - 1)
        elif chosen_policy == "shuffled_pool" and self._pool:
            if self._pool_index == 0:
                self._rng.shuffle(self._pool)
            self._next_seed = int(self._pool[self._pool_index % len(self._pool)])
            self._pool_index = (self._pool_index + 1) % len(self._pool)
        else:
            self._next_seed += 1
            chosen_policy = "incremental"
        return SeedEvent(
            update=int(update),
            old_seed=int(old_seed),
            new_seed=int(self._next_seed),
            reason=reason,
            policy=chosen_policy,
        )

    def advance(self, count: int = 1) -> int:
        self._next_seed += int(count)
        return self._next_seed

    @staticmethod
    def parse_seed_set(raw: object) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, str):
            text = raw.strip()
            if ".." in text:
                start_s, end_s = text.split("..", maxsplit=1)
                start = int(start_s)
                end = int(end_s)
                step = 1 if end >= start else -1
                return list(range(start, end + step, step))
            if "-" in text and text.count("-") == 1 and text.replace("-", "").isdigit():
                start_s, end_s = text.split("-", maxsplit=1)
                start = int(start_s)
                end = int(end_s)
                step = 1 if end >= start else -1
                return list(range(start, end + step, step))
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        if isinstance(raw, Iterable):
            return [int(v) for v in raw]
        return []

    def _is_plateau(self) -> bool:
        if len(self._recent_metrics) < max(self._cfg.plateau_window, 1):
            return False
        vals = list(self._recent_metrics)
        return (max(vals) - min(vals)) <= float(self._cfg.plateau_delta)
