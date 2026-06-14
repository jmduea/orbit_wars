from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AcceleratorPreference:
    """Ordered Kaggle accelerator fallback policy."""

    # Single-GPU Kaggle machine shapes only, highest VRAM class first.
    accelerator_ids: tuple[str, ...] = (
        "NvidiaH100",
        "NvidiaRtxPro6000",
        "NvidiaTeslaA100",
        "NvidiaL4",
        "NvidiaL4X1",
        "NvidiaTeslaT4Highmem",
        "NvidiaTeslaT4",
        "NvidiaTeslaP100",
    )

    def candidates_after(self, failed: Sequence[str] = ()) -> tuple[str, ...]:
        failed_set = set(failed)
        return tuple(
            accelerator
            for accelerator in self.accelerator_ids
            if accelerator not in failed_set
        )

    def first_available(self, failed: Sequence[str] = ()) -> str | None:
        remaining = self.candidates_after(failed)
        return remaining[0] if remaining else None


@dataclass(frozen=True, slots=True)
class PopulationCandidate:
    """W&B sweep candidate materialized for local diagnostics."""

    candidate_id: str
    overrides: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Throughput calibration summary for a candidate worker."""

    selected_overrides: tuple[str, ...]
    samples_per_sec: float
    ppo_samples_per_sec: float
    stable: bool
    reason: str = ""

    @property
    def throughput_score(self) -> float:
        if not self.stable:
            return 0.0
        return min(float(self.samples_per_sec), float(self.ppo_samples_per_sec))


@dataclass(frozen=True, slots=True)
class ShortlistRow:
    """Candidate row used by W&B-backed promotion shortlist generation."""

    run_id: str
    name: str
    state: str
    checkpoint_artifact: str | None
    checkpoint_artifact_version: str | None = None
    checkpoint_artifact_aliases: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    config: Mapping[str, object] = field(default_factory=dict)

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.checkpoint_artifact)

    @property
    def is_finished(self) -> bool:
        return self.state.lower() in {"finished", "completed", "success"}

    @property
    def score(self) -> float:
        objective = float(
            self.metrics.get(
                "preflight_sweep_score",
                self.metrics.get("episode_reward_mean", 0.0),
            )
        )
        samples = float(self.metrics.get("samples_per_sec", 0.0))
        ppo_samples = float(self.metrics.get("ppo_samples_per_sec", 0.0))
        stability = 1.0 if self.is_finished and self.has_checkpoint else 0.0
        return (1000.0 * stability) + objective + 0.0001 * min(samples, ppo_samples)


def rank_shortlist(
    rows: Sequence[ShortlistRow], *, limit: int = 10
) -> list[ShortlistRow]:
    """Rank completed checkpointed candidates ahead of partial diagnostics."""

    return sorted(
        rows,
        key=lambda row: (
            row.is_finished,
            row.has_checkpoint,
            row.score,
            float(
                row.metrics.get(
                    "preflight_sweep_score",
                    row.metrics.get("episode_reward_mean", 0.0),
                )
            ),
        ),
        reverse=True,
    )[: max(int(limit), 0)]


def render_hydra_command(overrides: Sequence[str]) -> list[str]:
    """Render the stable Orbit Wars training entrypoint for a worker."""

    return ["uv", "run", "python", "-m", "src.train", *overrides]
