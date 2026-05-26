#!/usr/bin/env python3
"""Kaggle-side W&B population worker entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.orchestration.population import render_hydra_command
from src.orchestration.throughput import (
    HardwareProfile,
    calibration_grid,
    estimate_training_overrides,
)


def main() -> None:
    _load_packaged_env()
    _ensure_uv_environment()
    summary: dict[str, Any] = {"diagnostics": diagnostics()}
    Path("worker-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _ensure_gpu(summary)
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    if not sweep_id:
        raise SystemExit("WANDB_SWEEP_ID is required for Kaggle population workers.")

    import wandb  # type: ignore

    def run_candidate() -> None:
        run = wandb.init(job_type="kaggle-population-worker")
        config = dict(wandb.config)
        observed = _hardware_profile(summary)
        calibration_overrides = estimate_training_overrides(
            observed,
            _prefixed_config(config, "model."),
            _prefixed_config(config, "task."),
        )
        variants = calibration_grid(calibration_overrides)
        wandb.log(
            {
                "kaggle_worker/observed_gpu": observed.gpu_name,
                "kaggle_worker/observed_memory_gb": observed.memory_gb,
                "kaggle_worker/calibration_variants": [list(item) for item in variants],
            },
            step=0,
        )
        overrides = _config_to_overrides(config)
        overrides.extend(calibration_overrides)
        overrides.extend(
            [
                "telemetry.wandb.enabled=true",
                "telemetry.wandb.log_artifacts=true",
                "artifacts.artifact_pipeline.docker_validation_async=false",
                "artifacts.replay.enabled=false",
            ]
        )
        command = render_hydra_command(tuple(overrides))
        print("worker command:", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False, text=True)
        run.summary["kaggle_worker_exit_code"] = completed.returncode
        run.finish(exit_code=completed.returncode)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    wandb.agent(sweep_id, function=run_candidate, count=1)


def diagnostics() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "nvidia_smi": _run_optional(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    }
    try:
        import jax

        payload["jax_devices"] = [str(device) for device in jax.devices()]
        payload["jax_platforms"] = sorted({device.platform for device in jax.devices()})
    except Exception as exc:
        payload["jax_error"] = repr(exc)
    return payload


def _ensure_gpu(summary: dict[str, Any]) -> None:
    if os.environ.get("ORBIT_WARS_KAGGLE_ALLOW_CPU") == "1":
        return
    platforms = set(summary.get("diagnostics", {}).get("jax_platforms", []))
    if "gpu" not in platforms:
        raise SystemExit(f"JAX GPU backend required for Kaggle worker. Platforms: {platforms}")


def _hardware_profile(summary: dict[str, Any]) -> HardwareProfile:
    smi = str(summary.get("diagnostics", {}).get("nvidia_smi", ""))
    gpu_name = smi.split(",", 1)[0].strip() if smi.strip() else "unknown"
    memory_gb = 16.0
    if "," in smi:
        memory_text = smi.split(",", 1)[1].strip().lower().replace("mib", "")
        try:
            memory_gb = float(memory_text) / 1024.0
        except ValueError:
            pass
    return HardwareProfile(
        accelerator_id=os.environ.get("KAGGLE_ACCELERATOR_ID", "unknown"),
        gpu_name=gpu_name,
        memory_gb=memory_gb,
    )


def _config_to_overrides(config: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for key, value in sorted(config.items()):
        if not isinstance(key, str) or "." not in key:
            continue
        if key.startswith("kaggle_worker."):
            continue
        overrides.append(f"{key}={_hydra_value(value)}")
    return overrides


def _prefixed_config(config: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key.removeprefix(prefix): value
        for key, value in config.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _run_optional(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return ""
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() or completed.stderr.strip()


def _load_packaged_env() -> None:
    env_path = Path("worker-env.json")
    if not env_path.exists():
        return
    payload = json.loads(env_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if value:
            os.environ.setdefault(str(key), str(value))


def _ensure_uv_environment() -> None:
    if shutil.which("uv") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv"],
            check=True,
            text=True,
        )
    if Path("pyproject.toml").exists():
        subprocess.run(["uv", "sync"], check=True, text=True)


if __name__ == "__main__":
    main()
