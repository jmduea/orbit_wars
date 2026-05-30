"""Environment propagation for artifact worker subprocesses."""

from __future__ import annotations

import os

from src.jax.device import configure_jax_runtime_for_host


def bootstrap_artifact_worker_jax_env() -> None:
    """Apply JAX platform settings before the worker imports JAX modules."""

    if os.environ.get("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA") == "1":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.pop("JAX_PLATFORM_NAME", None)
        return
    configure_jax_runtime_for_host()


def artifact_worker_subprocess_env() -> dict[str, str]:
    """Environment dict for worker and nested validation subprocesses."""

    bootstrap_artifact_worker_jax_env()
    return dict(os.environ)
