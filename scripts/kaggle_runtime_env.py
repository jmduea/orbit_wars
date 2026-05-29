"""Kaggle/runtime environment helpers for Orbit Wars workers.

This module must be safe to import before JAX.  The critical rule for NVIDIA
Kaggle workers is: never set ``JAX_PLATFORMS=gpu``.  Newer JAX treats the
``gpu`` alias as a request to consider GPU backends broadly and may attempt
ROCm before/alongside CUDA.  Kaggle NVIDIA kernels should be pinned to
``cuda,cpu``.
"""

from __future__ import annotations

import os
from pathlib import Path

KAGGLE_RUNTIME_ENV_PATCH_VERSION = "cuda-platform-pin-v10"
_WORKER_VENV_ENV = "ORBIT_WARS_WORKER_VENV"
_DRIVER_DIR_CANDIDATES: tuple[str, ...] = (
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
    "/usr/local/cuda/compat",
    "/usr/local/cuda-12.8/compat",
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
)


def pin_jax_platform_from_kaggle(*, env: dict[str, str] | None = None) -> None:
    """Pin JAX's platform selection for Kaggle before importing JAX.

    ``JAX_PLATFORMS=gpu`` is intentionally avoided for NVIDIA workers because it
    can trigger ROCm initialization attempts.  Use ``cuda,cpu`` instead.
    """

    target = os.environ if env is None else env
    accelerator_id = target.get("KAGGLE_ACCELERATOR_ID", os.environ.get("KAGGLE_ACCELERATOR_ID", ""))
    backend = target.get("ORBIT_WARS_JAX_BACKEND", os.environ.get("ORBIT_WARS_JAX_BACKEND", "")).strip().lower()
    target.pop("JAX_PLATFORM_NAME", None)

    if backend == "tpu" or _accelerator_requests_tpu(accelerator_id):
        target["JAX_PLATFORMS"] = "tpu,cpu"
    elif backend == "cpu" or _truthy(target.get("ORBIT_WARS_FORCE_JAX_CPU", os.environ.get("ORBIT_WARS_FORCE_JAX_CPU", ""))):
        target["JAX_PLATFORMS"] = "cpu"
    elif _accelerator_requests_nvidia(accelerator_id) or _nvidia_runtime_visible():
        target["JAX_PLATFORMS"] = "cuda,cpu"
    else:
        # Avoid GPU plugin probing on plain CPU hosts.  Preserve an explicit user
        # setting if one was already provided.
        target.setdefault("JAX_PLATFORMS", "cpu")


def isolate_worker_python_env(*, env: dict[str, str] | None = None) -> None:
    """Make subprocesses prefer the managed worker venv and avoid user site dirs."""

    target = os.environ if env is None else env
    target["PYTHONNOUSERSITE"] = "1"
    target.pop("PYTHONHOME", None)
    venv = _worker_venv(target)
    if venv is None:
        return
    target["VIRTUAL_ENV"] = str(venv)
    target["UV_PROJECT_ENVIRONMENT"] = str(venv)
    bin_dir = venv / "bin"
    if bin_dir.exists():
        target["PATH"] = _prepend_path(str(bin_dir), target.get("PATH", ""))


def add_worker_cuda_library_path(*, env: dict[str, str] | None = None) -> None:
    """Expose venv CUDA wheel libraries plus Kaggle's NVIDIA driver libraries."""

    target = os.environ if env is None else env
    accelerator_id = target.get("KAGGLE_ACCELERATOR_ID", os.environ.get("KAGGLE_ACCELERATOR_ID", ""))
    if not (_accelerator_requests_nvidia(accelerator_id) or _nvidia_runtime_visible()):
        return
    venv = _worker_venv(target)
    dirs: list[str] = []
    if venv is not None:
        dirs.extend(_cuda_wheel_library_dirs(venv))
    dirs.extend(_cuda_driver_library_dirs(target.get("LD_LIBRARY_PATH", "")))
    if dirs:
        target["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(dirs))
    target.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def _worker_venv(env: dict[str, str]) -> Path | None:
    value = env.get(_WORKER_VENV_ENV, os.environ.get(_WORKER_VENV_ENV, "")).strip()
    if not value:
        return None
    return Path(value).resolve()


def _cuda_wheel_library_dirs(venv: Path) -> list[str]:
    dirs: list[str] = []
    for site_packages in sorted(venv.glob("lib/python*/site-packages")):
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                dirs.append(str(lib_dir.resolve()))
    return dirs


def _cuda_driver_library_dirs(existing_ld_library_path: str) -> list[str]:
    candidates = [
        *(item for item in existing_ld_library_path.split(os.pathsep) if item),
        *_DRIVER_DIR_CANDIDATES,
    ]
    result: list[str] = []
    for item in candidates:
        path = Path(item)
        if not path.is_dir():
            continue
        if any(path.glob("libcuda.so*")) or any(path.glob("libnvidia-ml.so*")):
            result.append(str(path.resolve()))
    return _dedupe(result)


def _nvidia_runtime_visible() -> bool:
    return Path("/usr/local/nvidia/lib64").is_dir() or any(Path("/dev").glob("nvidia[0-9]*"))


def _accelerator_requests_nvidia(accelerator_id: str) -> bool:
    return accelerator_id.strip().lower().startswith("nvidia")


def _accelerator_requests_tpu(accelerator_id: str) -> bool:
    return accelerator_id.strip().lower().startswith("tpu")


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prepend_path(item: str, existing: str) -> str:
    parts = [part for part in existing.split(os.pathsep) if part]
    return os.pathsep.join([item, *(part for part in parts if part != item)])


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
