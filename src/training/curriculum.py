from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax.numpy as jnp

import jax
from src.opponents.pool import (
    OPPONENT_FAMILY_COUNT,
    OPPONENT_FAMILY_IDS,
    OPPONENT_FAMILY_NAMES,
    OPPONENT_LATEST,
)

METRIC_KEYS = (
    "overall_win_rate",
    "win_rate_2p",
    "first_place_rate_4p",
    "average_reward",
    "survival_time",
    "score_share",
    "approx_kl",
    "episode_reward_mean",
)


class StageView(NamedTuple):
    stage_index: jnp.ndarray
    family_ids: jnp.ndarray
    family_probs: jnp.ndarray
    family_mask: jnp.ndarray
    snapshot_pool_ids: jnp.ndarray
    snapshot_valid_mask: jnp.ndarray
    snapshot_age_updates: jnp.ndarray
    historical_selection_probs: jnp.ndarray
    fallback_family_id: jnp.ndarray


@dataclass(slots=True)
class PromotionRule:
    metric: str = ""
    op: str = ">="
    value: float = 0.0
    window_updates: int = 1


@dataclass(slots=True)
class CurriculumStage:
    id: str
    opponent_families: dict[str, float]
    min_updates: int = 1
    cooldown_updates: int = 0
    promote_if: PromotionRule | None = None
    format_weights: dict[int, float] = field(default_factory=dict)


class CurriculumController:
    """Host-side staged curriculum controller that emits immutable JAX stage views."""

    def __init__(
        self,
        curriculum_cfg: Any,
        snapshot_cfg: Any | None = None,
        *,
        static_format_weights: dict[int, float] | None = None,
    ) -> None:
        self.enabled = bool(getattr(curriculum_cfg, "enabled", False))
        self.snapshot_selection = str(getattr(snapshot_cfg, "selection", "uniform"))
        self.static_format_weights = dict(static_format_weights or {2: 1.0, 4: 1.0})
        raw_stages = list(getattr(curriculum_cfg, "stages", []) or [])
        if not self.enabled or not raw_stages:
            raw_stages = [
                {
                    "id": "default_latest",
                    "opponent_families": {"latest": 1.0},
                }
            ]
        self.stages = [self._parse_stage(raw, index) for index, raw in enumerate(raw_stages)]
        self.stage_index = 0
        self.stage_start_update = 1
        self.cooldown_until = 0
        self.metric_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=128))

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    def current_phase_id(self) -> str:
        return self.stage.id

    def current_stage_id(self) -> str:
        return self.stage.id

    def current_format_weights(self) -> dict[int, float]:
        if self.stage.format_weights:
            return self.stage.format_weights
        return dict(self.static_format_weights)

    def state_dict(self) -> dict[str, object]:
        return {
            "stage_index": self.stage_index,
            "stage_start_update": self.stage_start_update,
            "cooldown_until": self.cooldown_until,
            "metric_history": {
                key: list(values) for key, values in self.metric_history.items()
            },
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        self.stage_index = min(
            max(int(payload.get("stage_index", 0)), 0), len(self.stages) - 1
        )
        self.stage_start_update = int(payload.get("stage_start_update", 1))
        self.cooldown_until = int(payload.get("cooldown_until", 0))
        raw_history = payload.get("metric_history", {})
        self.metric_history.clear()
        if isinstance(raw_history, dict):
            for key, values in raw_history.items():
                history = deque(maxlen=128)
                if isinstance(values, list):
                    history.extend(float(value) for value in values)
                self.metric_history[str(key)] = history

    def apply(self, _cfg: Any) -> None:
        return None

    def stage_view(
        self,
        update: int,
        *,
        snapshot_ids: Any,
        snapshot_valid_mask: Any,
        snapshot_updates: Any,
    ) -> StageView:
        weights = self._normalized_family_weights(self.stage.opponent_families)
        family_probs = jnp.asarray(weights, dtype=jnp.float32)
        valid_mask = jnp.asarray(snapshot_valid_mask, dtype=bool)
        ids = jnp.asarray(snapshot_ids, dtype=jnp.int32)
        updates = jnp.asarray(snapshot_updates, dtype=jnp.int32)
        ages = jnp.where(valid_mask, jnp.asarray(update, dtype=jnp.int32) - updates, 0)
        if self.snapshot_selection == "recent_biased":
            valid_updates = jnp.where(valid_mask, updates, jnp.max(updates))
            oldest = jnp.min(valid_updates)
            historical_probs = jnp.where(
                valid_mask,
                (updates - oldest + 1).astype(jnp.float32),
                0.0,
            )
        else:
            historical_probs = valid_mask.astype(jnp.float32)
        total = historical_probs.sum()
        historical_probs = jnp.where(
            total > 0.0,
            historical_probs / jnp.maximum(total, 1.0),
            jnp.zeros_like(historical_probs),
        )
        historical_probs = jnp.where(
            jnp.isfinite(historical_probs), historical_probs, 0.0
        )
        return StageView(
            stage_index=jnp.asarray(self.stage_index, dtype=jnp.int32),
            family_ids=OPPONENT_FAMILY_IDS,
            family_probs=family_probs,
            family_mask=family_probs > 0.0,
            snapshot_pool_ids=ids,
            snapshot_valid_mask=valid_mask,
            snapshot_age_updates=ages,
            historical_selection_probs=historical_probs,
            fallback_family_id=jnp.asarray(OPPONENT_LATEST, dtype=jnp.int32),
        )

    def stage_telemetry(self, view: StageView, update: int) -> dict[str, object]:
        probs = [float(x) for x in list(view.family_probs)]
        record: dict[str, object] = {
            "curriculum_stage_id": self.stage.id,
            "curriculum_stage_index": self.stage_index,
            "curriculum_stage_update": update,
            "curriculum_stage_dwell_updates": update - self.stage_start_update + 1,
        }
        for family, prob in zip(OPPONENT_FAMILY_NAMES, probs, strict=True):
            record[f"curriculum_family_prob_{family}"] = prob
        return record

    def update(self, update_idx: int, metrics: dict[str, float]) -> dict[str, Any] | None:
        for key, value in metrics.items():
            if value is not None:
                self.metric_history[str(key)].append(float(value))
        if update_idx < self.cooldown_until:
            return None
        dwell = update_idx - self.stage_start_update + 1
        if dwell < max(int(self.stage.min_updates), 1):
            return None
        rule = self.stage.promote_if
        if rule is None or self.stage_index >= len(self.stages) - 1:
            return None
        required = max(int(rule.window_updates), 1)
        window = list(self.metric_history[rule.metric])[-required:]
        if len(window) < required:
            return None
        value = sum(window) / float(len(window))
        if not self._compare(value, rule.op, rule.value):
            return {
                "event": "curriculum_stage_promotion_blocked",
                "update": update_idx,
                "stage": self.stage.id,
                "reason": "threshold",
                "metric": rule.metric,
                "metric_value": value,
            }
        previous = self.stage
        self.stage_index += 1
        self.stage_start_update = update_idx + 1
        self.cooldown_until = update_idx + max(int(previous.cooldown_updates), 0)
        return {
            "event": "curriculum_stage_promoted",
            "update": update_idx,
            "from_stage": previous.id,
            "to_stage": self.stage.id,
            "metric": rule.metric,
            "metric_value": value,
            "threshold": rule.value,
        }

    def _parse_stage(self, raw: dict[str, Any], index: int) -> CurriculumStage:
        promote = raw.get("promote_if") or None
        rule = None
        if promote:
            rule = PromotionRule(
                metric=str(promote.get("metric", "")),
                op=str(promote.get("op", ">=")),
                value=float(promote.get("value", 0.0)),
                window_updates=int(promote.get("window_updates", 1)),
            )
        return CurriculumStage(
            id=str(raw.get("id", f"stage_{index}")),
            opponent_families={
                str(key): float(value)
                for key, value in dict(raw.get("opponent_families", {"latest": 1.0})).items()
            },
            min_updates=int(raw.get("min_updates", raw.get("min_dwell_updates", 1))),
            cooldown_updates=int(raw.get("cooldown_updates", 0)),
            promote_if=rule,
            format_weights={
                int(key): float(value) for key, value in dict(raw.get("format_weights", {})).items()
            },
        )

    def _normalized_family_weights(self, weights: dict[str, float]) -> list[float]:
        values = [float(weights.get(name, 0.0)) for name in OPPONENT_FAMILY_NAMES]
        total = sum(values)
        if total <= 0.0:
            return [1.0] + [0.0] * (OPPONENT_FAMILY_COUNT - 1)
        return [value / total for value in values]

    def _compare(self, value: float, op: str, threshold: float) -> bool:
        if op == ">=":
            return value >= threshold
        if op == ">":
            return value > threshold
        if op == "<=":
            return value <= threshold
        if op == "<":
            return value < threshold
        return False


def default_stage_view(cfg: Any) -> StageView:
    opponents = getattr(cfg, "opponents", None)
    mode = getattr(opponents, "mode", None) if opponents is not None else None
    opponent = getattr(mode, "opponent", "self")
    family = "random" if opponent == "random" else "latest"
    controller = CurriculumController(
        type(
            "DefaultCurriculum",
            (),
            {
                "enabled": True,
                "stages": [{"id": f"default_{family}", "opponent_families": {family: 1.0}}],
            },
        )()
    )
    return controller.stage_view(
        0,
        snapshot_ids=jnp.zeros((1,), dtype=jnp.int32),
        snapshot_valid_mask=jnp.zeros((1,), dtype=bool),
        snapshot_updates=jnp.zeros((1,), dtype=jnp.int32),
    )
