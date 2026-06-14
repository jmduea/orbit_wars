from __future__ import annotations

from collections import deque
from pathlib import Path

from src.jax.preflight_calibration import (
    WINDOW_UPDATES,
    default_calibration_json_path,
    load_thresholds,
)

PLANET_FLOW_MIN_DEMAND_MASS = 100.0
PREFLIGHT_SWEEP_SCORE_INELIGIBLE = -1.0
PLANET_FLOW_MIN_EMITTED_LAUNCHES = 50.0
PLANET_FLOW_MIN_MEAN_LAUNCHES = 0.05
PLANET_FLOW_MIN_ENTROPY = 1.0e-3
PLANET_FLOW_MAX_APPROX_KL = 0.15
PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE = 0.05


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


class MetricWindowTracker:
    """Rolling mean over the last ``window`` observations (preflight-aligned)."""

    def __init__(self, *, window: int = WINDOW_UPDATES) -> None:
        self._window = max(int(window), 1)
        self._values: deque[float] = deque(maxlen=self._window)

    def observe(self, value: float) -> None:
        self._values.append(float(value))

    def window_mean(self) -> float | None:
        if len(self._values) < self._window:
            return None
        return sum(self._values) / len(self._values)


class WinRateTrendTracker:
    """Rolling overall_win_rate buffer for preflight-aligned trend metrics."""

    def __init__(self, *, window: int = WINDOW_UPDATES) -> None:
        self._window = max(int(window), 1)
        self._values: deque[float] = deque(maxlen=self._window * 4)

    def observe(self, overall_win_rate: float) -> None:
        self._values.append(float(overall_win_rate))

    def rolling_window_means(self) -> list[float]:
        if len(self._values) < self._window:
            return []
        ordered = list(self._values)
        return [
            sum(ordered[index - self._window : index]) / self._window
            for index in range(self._window, len(ordered) + 1)
        ]

    def win_rate_window_mean(self) -> float | None:
        windows = self.rolling_window_means()
        if not windows:
            return None
        return windows[-1]

    def best_win_rate_window_mean(self) -> float | None:
        windows = self.rolling_window_means()
        if not windows:
            return None
        return max(windows)

    def win_rate_delta(self) -> float | None:
        windows = self.rolling_window_means()
        if not windows:
            return None
        return windows[-1] - windows[0]

    def win_rate_recovery_delta(self) -> float | None:
        windows = self.rolling_window_means()
        if not windows:
            return None
        prior_floor = min(windows[:-1]) if len(windows) > 1 else windows[0]
        return windows[-1] - prior_floor


def planet_flow_sweep_guardrail_reasons(
    *,
    win_rate_delta: float | None,
    mean_active_launches_per_turn: float | None,
    planet_flow_demanded_mass_sum: float | None,
    planet_flow_emitted_launch_count: float | None,
    entropy: float | None,
    approx_kl: float | None,
    planet_flow_unreachable_demand_rate: float | None = None,
    max_post_mask_unreachable_rate: float = PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE,
    max_approx_kl: float = PLANET_FLOW_MAX_APPROX_KL,
    min_entropy: float = PLANET_FLOW_MIN_ENTROPY,
) -> list[str]:
    """Structured ineligibility reasons for sweep shortlist audit (R12)."""

    reasons: list[str] = []
    if win_rate_delta is None:
        reasons.append("missing win_rate_delta_10 (need >= 10 updates)")
    launches = mean_active_launches_per_turn
    demand = planet_flow_demanded_mass_sum
    emitted = planet_flow_emitted_launch_count
    ent = entropy
    kl = approx_kl
    if launches is None:
        reasons.append("missing mean_active_launches_per_turn")
    elif launches < PLANET_FLOW_MIN_MEAN_LAUNCHES:
        reasons.append(
            f"mean_active_launches_per_turn {launches:.4f} < {PLANET_FLOW_MIN_MEAN_LAUNCHES:.4f}"
        )
    if demand is None:
        reasons.append("missing planet_flow_demanded_mass_sum")
    elif demand < PLANET_FLOW_MIN_DEMAND_MASS:
        reasons.append(
            f"planet_flow_demanded_mass_sum {demand:.1f} < {PLANET_FLOW_MIN_DEMAND_MASS:.1f}"
        )
    if emitted is None:
        reasons.append("missing planet_flow_emitted_launch_count")
    elif emitted < PLANET_FLOW_MIN_EMITTED_LAUNCHES:
        reasons.append(
            f"planet_flow_emitted_launch_count {emitted:.1f} < {PLANET_FLOW_MIN_EMITTED_LAUNCHES:.1f}"
        )
    if ent is None:
        reasons.append("missing entropy_window_mean")
    elif ent < min_entropy:
        reasons.append(f"entropy_window_mean {ent:.6f} < {min_entropy:.6f}")
    if kl is None:
        reasons.append("missing approx_kl_window_mean")
    elif kl > max_approx_kl:
        reasons.append(f"approx_kl_window_mean {kl:.4f} > {max_approx_kl:.4f}")
    unreachable = planet_flow_unreachable_demand_rate
    if unreachable is not None and unreachable > max_post_mask_unreachable_rate:
        reasons.append(
            "planet_flow_unreachable_demand_rate "
            f"{unreachable:.4f} > {max_post_mask_unreachable_rate:.4f}"
        )
    return reasons


def planet_flow_sweep_eval(
    *,
    win_rate_delta: float | None,
    mean_active_launches_per_turn: float | None,
    planet_flow_demanded_mass_sum: float | None,
    planet_flow_emitted_launch_count: float | None,
    entropy: float | None,
    approx_kl: float | None,
    planet_flow_unreachable_demand_rate: float | None = None,
    max_post_mask_unreachable_rate: float = PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE,
    max_approx_kl: float = PLANET_FLOW_MAX_APPROX_KL,
    min_entropy: float = PLANET_FLOW_MIN_ENTROPY,
) -> tuple[float, list[str]]:
    """Return sweep score and guardrail reasons in one eligibility pass."""

    reasons = planet_flow_sweep_guardrail_reasons(
        win_rate_delta=win_rate_delta,
        mean_active_launches_per_turn=mean_active_launches_per_turn,
        planet_flow_demanded_mass_sum=planet_flow_demanded_mass_sum,
        planet_flow_emitted_launch_count=planet_flow_emitted_launch_count,
        entropy=entropy,
        approx_kl=approx_kl,
        planet_flow_unreachable_demand_rate=planet_flow_unreachable_demand_rate,
        max_post_mask_unreachable_rate=max_post_mask_unreachable_rate,
        max_approx_kl=max_approx_kl,
        min_entropy=min_entropy,
    )
    if reasons:
        return PREFLIGHT_SWEEP_SCORE_INELIGIBLE, reasons
    assert win_rate_delta is not None
    return float(win_rate_delta), reasons


def planet_flow_preflight_score(
    *,
    win_rate_delta: float | None,
    mean_active_launches_per_turn: float | None,
    planet_flow_demanded_mass_sum: float | None,
    planet_flow_emitted_launch_count: float | None,
    entropy: float | None,
    approx_kl: float | None,
    planet_flow_unreachable_demand_rate: float | None = None,
    max_post_mask_unreachable_rate: float = PLANET_FLOW_MAX_POST_MASK_UNREACHABLE_RATE,
    max_approx_kl: float = PLANET_FLOW_MAX_APPROX_KL,
    min_entropy: float = PLANET_FLOW_MIN_ENTROPY,
) -> float:
    """Composite W&B sweep objective aligned with learn-proof trend + activity floors.

    ``approx_kl`` and ``entropy`` must be **last-window means** (same as preflight gates),
    not point samples from the latest PPO update.
    """

    score, _ = planet_flow_sweep_eval(
        win_rate_delta=win_rate_delta,
        mean_active_launches_per_turn=mean_active_launches_per_turn,
        planet_flow_demanded_mass_sum=planet_flow_demanded_mass_sum,
        planet_flow_emitted_launch_count=planet_flow_emitted_launch_count,
        entropy=entropy,
        approx_kl=approx_kl,
        planet_flow_unreachable_demand_rate=planet_flow_unreachable_demand_rate,
        max_post_mask_unreachable_rate=max_post_mask_unreachable_rate,
        max_approx_kl=max_approx_kl,
        min_entropy=min_entropy,
    )
    return score


def collect_planet_flow_sweep_metrics(
    *,
    win_rate_trend: WinRateTrendTracker,
    approx_kl_window: MetricWindowTracker | None,
    entropy_window: MetricWindowTracker | None,
    overall_win_rate: float,
    metrics_host: dict[str, object],
    rollout_scalars: dict[str, float | None],
    max_post_mask_unreachable_rate: float,
) -> dict[str, float]:
    """Build W&B sweep metrics from live training-loop trackers."""

    win_rate_trend.observe(overall_win_rate)
    metrics: dict[str, float] = {}
    win_rate_delta_10 = win_rate_trend.win_rate_delta()
    if win_rate_delta_10 is not None:
        metrics["win_rate_delta_10"] = float(win_rate_delta_10)

    approx_kl_wm: float | None = None
    entropy_wm: float | None = None
    if approx_kl_window is not None and "approx_kl" in metrics_host:
        approx_kl_window.observe(float(metrics_host["approx_kl"]))
        approx_kl_wm = approx_kl_window.window_mean()
        if approx_kl_wm is not None:
            metrics["approx_kl_window_mean"] = float(approx_kl_wm)
    if entropy_window is not None and "entropy" in metrics_host:
        entropy_window.observe(float(metrics_host["entropy"]))
        entropy_wm = entropy_window.window_mean()
        if entropy_wm is not None:
            metrics["entropy_window_mean"] = float(entropy_wm)

    metrics["preflight_sweep_score"] = planet_flow_preflight_score(
        win_rate_delta=win_rate_delta_10,
        mean_active_launches_per_turn=rollout_scalars.get(
            "mean_active_launches_per_turn"
        ),
        planet_flow_demanded_mass_sum=rollout_scalars.get(
            "planet_flow_demanded_mass_sum"
        ),
        planet_flow_emitted_launch_count=rollout_scalars.get(
            "planet_flow_emitted_launch_count"
        ),
        entropy=entropy_wm,
        approx_kl=approx_kl_wm,
        planet_flow_unreachable_demand_rate=rollout_scalars.get(
            "planet_flow_unreachable_demand_rate"
        ),
        max_post_mask_unreachable_rate=max_post_mask_unreachable_rate,
    )
    return metrics


def ssot_preflight_learning_signal_thresholds(
    thresholds: dict[str, object] | None = None,
) -> tuple[float, float, float]:
    """Gates 2–3 floors from committed preflight calibration (min delta, max KL, min entropy)."""

    if thresholds is None:
        repo_root = Path(__file__).resolve().parents[3]
        thresholds = load_thresholds(default_calibration_json_path(repo_root))
    learning = thresholds.get("learning_signal")
    if not isinstance(learning, dict):
        return 0.05, 0.15, 0.0001
    return (
        float(learning.get("min_win_rate_delta", 0.05)),
        float(learning.get("max_approx_kl", 0.15)),
        float(learning.get("min_entropy", 0.0001)),
    )


def preflight_sweep_score(
    *,
    win_rate_delta: float | None,
    approx_kl: float | None,
    entropy: float | None,
    win_rate_recovery_delta: float | None = None,
    min_win_rate_delta: float = 0.05,
    max_approx_kl: float = 0.15,
    min_entropy: float = 0.0001,
) -> float:
    """W&B sweep objective for short preflight trend, guarded by KL/entropy floors."""

    if win_rate_delta is None or approx_kl is None or entropy is None:
        return PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    candidate_delta = max(
        float(win_rate_delta),
        float(win_rate_recovery_delta)
        if win_rate_recovery_delta is not None
        else float("-inf"),
    )
    if (
        candidate_delta < float(min_win_rate_delta)
        or float(approx_kl) > float(max_approx_kl)
        or float(entropy) < float(min_entropy)
    ):
        return PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    return float(candidate_delta)

def collect_ssot_preflight_sweep_metrics(
    *,
    win_rate_trend: WinRateTrendTracker,
    approx_kl_window: MetricWindowTracker | None,
    entropy_window: MetricWindowTracker | None,
    overall_win_rate: float,
    metrics_host: dict[str, object],
    thresholds: dict[str, object] | None = None,
) -> dict[str, float]:
    """Build SSOT preflight sweep metrics from live training-loop trackers."""

    win_rate_trend.observe(overall_win_rate)
    metrics: dict[str, float] = {}
    win_rate_delta_10 = win_rate_trend.win_rate_delta()
    if win_rate_delta_10 is not None:
        metrics["win_rate_delta_10"] = float(win_rate_delta_10)
    win_rate_recovery_delta_10 = win_rate_trend.win_rate_recovery_delta()
    if win_rate_recovery_delta_10 is not None:
        metrics["win_rate_recovery_delta_10"] = float(win_rate_recovery_delta_10)
    win_rate_window_mean_10 = win_rate_trend.win_rate_window_mean()
    if win_rate_window_mean_10 is not None:
        metrics["win_rate_window_mean_10"] = float(win_rate_window_mean_10)
    win_rate_best_window_mean_10 = win_rate_trend.best_win_rate_window_mean()
    if win_rate_best_window_mean_10 is not None:
        metrics["win_rate_best_window_mean_10"] = float(win_rate_best_window_mean_10)

    approx_kl_wm: float | None = None
    entropy_wm: float | None = None
    if approx_kl_window is not None and "approx_kl" in metrics_host:
        approx_kl_window.observe(float(metrics_host["approx_kl"]))
        approx_kl_wm = approx_kl_window.window_mean()
        if approx_kl_wm is not None:
            metrics["approx_kl_window_mean"] = float(approx_kl_wm)
    if entropy_window is not None and "entropy" in metrics_host:
        entropy_window.observe(float(metrics_host["entropy"]))
        entropy_wm = entropy_window.window_mean()
        if entropy_wm is not None:
            metrics["entropy_window_mean"] = float(entropy_wm)

    min_delta, max_kl, min_ent = ssot_preflight_learning_signal_thresholds(thresholds)
    metrics["preflight_sweep_score"] = preflight_sweep_score(
        win_rate_delta=win_rate_delta_10,
        win_rate_recovery_delta=win_rate_recovery_delta_10,
        approx_kl=approx_kl_wm,
        entropy=entropy_wm,
        min_win_rate_delta=min_delta,
        max_approx_kl=max_kl,
        min_entropy=min_ent,
    )
    return metrics


def is_ssot_preflight_sweep(cfg: object) -> bool:
    """True when W&B tags mark a short SSOT preflight sweep agent run."""

    telemetry = getattr(cfg, "telemetry", None)
    wandb_cfg = getattr(telemetry, "wandb", None) if telemetry is not None else None
    tags = getattr(wandb_cfg, "tags", None) if wandb_cfg is not None else None
    if not tags:
        return False
    normalized = {str(tag).strip() for tag in tags}
    return "ssot_preflight" in normalized or "preflight" in normalized
