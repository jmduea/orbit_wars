"""Remote orchestration helpers for hosted Orbit Wars training."""

from .kaggle_cli import KaggleCli, KaggleKernelRef, KaggleKernelStatus
from .population import (
    AcceleratorPreference,
    CalibrationResult,
    PopulationCandidate,
    ShortlistRow,
    rank_shortlist,
)
from .throughput import HardwareProfile, estimate_training_overrides

__all__ = [
    "AcceleratorPreference",
    "CalibrationResult",
    "HardwareProfile",
    "KaggleCli",
    "KaggleKernelRef",
    "KaggleKernelStatus",
    "PopulationCandidate",
    "ShortlistRow",
    "estimate_training_overrides",
    "rank_shortlist",
]
