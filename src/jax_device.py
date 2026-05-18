"""JAX device selection checks for Orbit Wars training and benchmarks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def nvidia_gpu_present() -> bool:
    """Return whether this host appears to expose NVIDIA GPU hardware."""

    if any(Path("/dev").glob("nvidia[0-9]*")):
        return True
    if Path("/proc/driver/nvidia/gpus").is_dir():
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if result.returncode == 0 and result.stdout.strip():
                return True
    return False


def configure_jax_runtime_for_host() -> None:
    """Set JAX/XLA runtime defaults before JAX is imported.

    CUDA-enabled JAX uses XLA's BFC allocator by default. Long attention-policy
    benchmarks can otherwise fail from allocator fragmentation on GPUs while XLA
    is compiling or autotuning large programs. ``cuda_malloc_async`` lets the
    CUDA driver reuse freed blocks more effectively and matches XLA's own OOM
    guidance. Hosts without visible NVIDIA hardware are pinned to CPU so CUDA
    plugins are not probed unnecessarily.
    """

    if nvidia_gpu_present():
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def configure_jax_platform_for_host() -> None:
    """Avoid probing CUDA plugins on hosts without visible NVIDIA hardware.

    Deprecated compatibility wrapper; new code should call
    :func:`configure_jax_runtime_for_host`.
    """

    configure_jax_runtime_for_host()


def ensure_cuda_jax_if_nvidia_present() -> None:
    """Fail fast when NVIDIA GPUs are present but JAX is running on CPU.

    Set ``ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1`` to explicitly bypass this
    guard for debugging or driver-maintenance sessions.
    """

    if os.environ.get("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA") == "1":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        return
    configure_jax_runtime_for_host()
    if not nvidia_gpu_present():
        return

    import jax

    devices = jax.devices()
    if any(device.platform == "gpu" for device in devices):
        return
    device_summary = ", ".join(str(device) for device in devices) or "no JAX devices"
    raise RuntimeError(
        "NVIDIA GPU hardware is present, but JAX did not initialize a CUDA "
        "backend and would fall back to CPU. Run `uv sync` to install the "
        "project's CUDA-enabled `jax[cuda13]` dependency, then verify that "
        "the NVIDIA driver is new enough for CUDA 13 and visible in this "
        f"environment. JAX devices: {device_summary}. Set "
        "ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1 only if CPU execution is "
        "intentional."
    )
