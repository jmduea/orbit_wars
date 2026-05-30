"""Best-effort GPU memory sampling for training telemetry."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuMemorySnapshot:
    """Observed GPU memory use from the device driver or JAX allocator."""

    gpu_name: str
    memory_used_gb: float
    memory_total_gb: float
    source: str


class GpuMemoryTracker:
    """Track per-update GPU memory and a running peak for the training run."""

    def __init__(self) -> None:
        self._baseline = probe_gpu_memory()
        self._peak_used_gb = self._baseline.memory_used_gb if self._baseline else 0.0

    @property
    def available(self) -> bool:
        return self._baseline is not None

    def run_metadata(self) -> dict[str, float | str]:
        """Static GPU metadata for step-0 run records."""

        if self._baseline is None:
            return {}
        return {
            "gpu_name": self._baseline.gpu_name,
            "gpu_memory_total_gb": self._baseline.memory_total_gb,
        }

    def sample_update_metrics(self) -> dict[str, float]:
        """Sample current GPU use and return canonical update metrics."""

        snapshot = probe_gpu_memory()
        if snapshot is None:
            return {}
        self._peak_used_gb = max(self._peak_used_gb, snapshot.memory_used_gb)
        return {
            "gpu_memory_used_gb": snapshot.memory_used_gb,
            "gpu_memory_total_gb": snapshot.memory_total_gb,
            "gpu_memory_peak_gb": self._peak_used_gb,
        }


def probe_gpu_memory() -> GpuMemorySnapshot | None:
    """Return current GPU memory use, preferring ``nvidia-smi`` over JAX stats."""

    smi_snapshot = _probe_nvidia_smi()
    if smi_snapshot is not None:
        return smi_snapshot
    return _probe_jax_memory_stats()


def _probe_nvidia_smi() -> GpuMemorySnapshot | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    name, used_mib, total_mib = parts[0], parts[1], parts[2]
    try:
        used_gb = float(used_mib) / 1024.0
        total_gb = float(total_mib) / 1024.0
    except ValueError:
        return None
    return GpuMemorySnapshot(
        gpu_name=name,
        memory_used_gb=used_gb,
        memory_total_gb=total_gb,
        source="nvidia-smi",
    )


def _probe_jax_memory_stats() -> GpuMemorySnapshot | None:
    try:
        import jax
    except ImportError:
        return None
    devices = [device for device in jax.devices() if device.platform in {"gpu", "cuda"}]
    if not devices:
        return None
    device = devices[0]
    memory_stats = getattr(device, "memory_stats", None)
    if memory_stats is None:
        return None
    try:
        stats = memory_stats()
    except Exception:
        return None
    bytes_in_use = stats.get("bytes_in_use")
    bytes_limit = stats.get("bytes_limit")
    if bytes_in_use is None or bytes_limit is None:
        return None
    try:
        used_gb = float(bytes_in_use) / (1024.0**3)
        total_gb = float(bytes_limit) / (1024.0**3)
    except (TypeError, ValueError):
        return None
    device_str = str(device)
    gpu_name = device_str.replace("cuda:", "cuda")
    return GpuMemorySnapshot(
        gpu_name=gpu_name,
        memory_used_gb=used_gb,
        memory_total_gb=total_gb,
        source="jax-memory-stats",
    )
