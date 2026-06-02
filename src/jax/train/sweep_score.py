from __future__ import annotations

from collections import deque

from src.jax.preflight_calibration import WINDOW_UPDATES

PLANET_FLOW_MIN_DEMAND_MASS = 100.0
PLANET_FLOW_MIN_EMITTED_LAUNCHES = 50.0
PLANET_FLOW_MIN_MEAN_LAUNCHES = 0.05
PLANET_FLOW_MIN_ENTROPY = 1.0e-3
PLANET_FLOW_MAX_APPROX_KL = 0.15
PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE = 0.05
PLANET_FLOW_SWEEP_SCORE_INELIGIBLE = -1.0


def planet_flow_max_post_mask_unreachable_rate(
    thresholds: dict[str, object] | None = None,
) -> float:
    """Calibrated post-mask unreachable ceiling; falls back to construction default."""

    if thresholds is None:
        return PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE
    planet_flow = thresholds.get("planet_flow_learning_signal")
    if not isinstance(planet_flow, dict):
        return PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE
    value = planet_flow.get("max_post_mask_unreachable_demand_rate")
    if value is None:
        return PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE
    return float(value)


class WinRateTrendTracker:
    """Rolling overall_win_rate buffer for preflight-aligned trend metrics."""

    def __init__(self, *, window: int = WINDOW_UPDATES) -> None:
        self._window = max(int(window), 1)
        self._values: deque[float] = deque(maxlen=self._window * 4)

    def observe(self, overall_win_rate: float) -> None:
        self._values.append(float(overall_win_rate))

    def win_rate_delta(self) -> float | None:
        if len(self._values) < self._window:
            return None
        ordered = list(self._values)
        first = sum(ordered[: self._window]) / self._window
        last = sum(ordered[-self._window :]) / self._window
        return last - first


def planet_flow_sweep_score(
    *,
    win_rate_delta: float | None,
    mean_active_launches_per_turn: float | None,
    planet_flow_demanded_mass_sum: float | None,
    planet_flow_emitted_launch_count: float | None,
    entropy: float | None,
    approx_kl: float | None,
    planet_flow_unreachable_demand_rate: float | None = None,
    max_post_mask_unreachable_rate: float = PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE,
) -> float:
    """Composite W&B sweep objective aligned with learn-proof trend + activity floors."""

    if win_rate_delta is None:
        return PLANET_FLOW_SWEEP_SCORE_INELIGIBLE
    launches = mean_active_launches_per_turn
    demand = planet_flow_demanded_mass_sum
    emitted = planet_flow_emitted_launch_count
    ent = entropy
    kl = approx_kl
    if launches is None or demand is None or emitted is None or ent is None or kl is None:
        return PLANET_FLOW_SWEEP_SCORE_INELIGIBLE
    if (
        launches < PLANET_FLOW_MIN_MEAN_LAUNCHES
        or demand < PLANET_FLOW_MIN_DEMAND_MASS
        or emitted < PLANET_FLOW_MIN_EMITTED_LAUNCHES
        or ent < PLANET_FLOW_MIN_ENTROPY
        or kl > PLANET_FLOW_MAX_APPROX_KL
    ):
        return PLANET_FLOW_SWEEP_SCORE_INELIGIBLE
    unreachable = planet_flow_unreachable_demand_rate
    if unreachable is not None and unreachable > max_post_mask_unreachable_rate:
        return PLANET_FLOW_SWEEP_SCORE_INELIGIBLE
    return float(win_rate_delta)
