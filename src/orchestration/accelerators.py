from __future__ import annotations

DEFAULT_KAGGLE_ACCELERATOR = "NvidiaTeslaP100"

KAGGLE_TPU_V5E8 = "TpuV5E8"

# Kaggle CLI machine_shape values for the v5e-8 single-host slice.
KAGGLE_TPU_ACCELERATOR_IDS: tuple[str, ...] = (
    KAGGLE_TPU_V5E8,
    "TpuV5e8",
    "TpuV5E-8",
)

# Conservative single-GPU VRAM budgets for throughput sizing when smi is unavailable.
_GPU_MEMORY_GB_BY_ACCELERATOR: dict[str, float] = {
    "nvidiah100": 80.0,
    "nvidiartxpro6000": 48.0,
    "nvidiateslaa100": 40.0,
    "nvidial4": 24.0,
    "nvidial4x1": 24.0,
    "nvidiateslata4highmem": 16.0,
    "nvidiateslata4": 16.0,
    "nvidiateslap100": 16.0,
}


def is_tpu_accelerator(accelerator_id: str) -> bool:
    """Return True when the Kaggle accelerator id targets a TPU backend."""

    normalized = accelerator_id.strip().lower()
    if not normalized:
        return False
    if normalized in {item.lower() for item in KAGGLE_TPU_ACCELERATOR_IDS}:
        return True
    return normalized.startswith("tpu")


def default_memory_gb(accelerator_id: str, *, fallback: float = 16.0) -> float:
    """Return a conservative host-memory budget for throughput sizing."""

    if is_tpu_accelerator(accelerator_id):
        if accelerator_id.strip().lower() in {
            KAGGLE_TPU_V5E8.lower(),
            "tpuv5e8",
            "tpuv5e-8",
        }:
            return 384.0
        return 192.0
    normalized = accelerator_id.strip().lower()
    return _GPU_MEMORY_GB_BY_ACCELERATOR.get(normalized, fallback)
