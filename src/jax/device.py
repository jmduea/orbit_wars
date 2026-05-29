"""JAX device selection checks for Orbit Wars training and benchmarks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _tpu_runtime_requested() -> bool:
    """Return True when the host should prefer JAX's TPU backend."""

    if os.environ.get("ORBIT_WARS_JAX_BACKEND", "").strip().lower() == "tpu":
        return True
    accelerator_id = os.environ.get("KAGGLE_ACCELERATOR_ID", "")
    if not accelerator_id:
        return False
    from src.orchestration.accelerators import is_tpu_accelerator

    return is_tpu_accelerator(accelerator_id)



def _nvidia_kaggle_requested() -> bool:
    accelerator_id = os.environ.get("KAGGLE_ACCELERATOR_ID", "")
    return accelerator_id.strip().lower().startswith("nvidia")


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
    guidance. Hosts without visible NVIDIA hardware are pinned to CPU unless a
    TPU backend was explicitly requested.
    """

    if nvidia_gpu_present():
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        # Do not use the generic JAX ``gpu`` alias on NVIDIA/Kaggle: it may
        # attempt ROCm initialization.  CUDA workers should be explicit.
        if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "gpu":
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["JAX_PLATFORMS"] = "cuda,cpu"
        elif not os.environ.get("JAX_PLATFORMS"):
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    elif _tpu_runtime_requested():
        return
    elif not os.environ.get("JAX_PLATFORMS"):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def configure_jax_platform_for_host() -> None:
    """Avoid probing CUDA plugins on hosts without visible NVIDIA hardware.

    Deprecated compatibility wrapper; new code should call
    :func:`configure_jax_runtime_for_host`.
    """

    configure_jax_runtime_for_host()


def ensure_jax_accelerator_backend() -> None:
    """Fail fast when the requested Kaggle accelerator backend is unavailable."""

    if os.environ.get("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA") == "1":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        return
    configure_jax_runtime_for_host()

    import jax

    devices = jax.devices()
    platforms = {device.platform for device in devices}
    device_summary = ", ".join(str(device) for device in devices) or "no JAX devices"

    if _tpu_runtime_requested():
        if "tpu" in platforms:
            return
        raise RuntimeError(
            "Kaggle TPU execution was requested, but JAX did not initialize a "
            "TPU backend. Re-run the worker bootstrap so it installs "
            "`jax[tpu]` from the libtpu release index, then verify the kernel "
            f"accelerator is set to TPU v5e-8. JAX devices: {device_summary}."
        )

    if not nvidia_gpu_present():
        return
    if {"gpu", "cuda"} & platforms:
        return
    raise RuntimeError(
        "NVIDIA GPU hardware is present, but JAX did not initialize a CUDA "
        "backend and would fall back to CPU. Ensure JAX_PLATFORMS is set to "
        "cuda,cpu (not gpu), and verify that the worker venv's CUDA wheel "
        "libraries plus Kaggle's NVIDIA driver libraries are visible in "
        f"LD_LIBRARY_PATH. JAX devices: {device_summary}. Set "
        "ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1 only if CPU execution is "
        "intentional."
    )


def ensure_cuda_jax_if_nvidia_present() -> None:
    """Deprecated compatibility wrapper for accelerator backend checks."""

    ensure_jax_accelerator_backend()
