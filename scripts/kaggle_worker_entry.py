#!/usr/bin/env python3
"""Kaggle-side W&B population worker entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from src.orchestration.kaggle_jax import (  # noqa: E402
    format_bootstrap_failure,
    log_bootstrap_failure,
    sync_kaggle_worker_environment,
)

_JAX_BACKEND_MARKER = Path(".orbit_wars_jax_backend")
_WORKER_VENV_ENV = "ORBIT_WARS_WORKER_VENV"
_WORKER_VENV_READY_ENV = "ORBIT_WARS_WORKER_VENV_READY"
KAGGLE_WORKER_ENTRY_PATCH_VERSION = "subprocess-jax-diagnostics-v9"

_CUDA_DRIVER_LIBRARY_DIR_CANDIDATES: tuple[str, ...] = (
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
    "/usr/local/cuda/compat",
    "/usr/local/cuda-12.8/compat",
    "/usr/lib/x86_64-linux-gnu",
)

_HYDRA_CONFIG_GROUP_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "training",
        "opponents",
        "curriculum",
        "reward",
        "task",
        "telemetry",
        "artifacts",
        "output",
    }
)


def main() -> None:
    print(f"ORBIT_WARS_WORKER_ENTRY_PATCH={KAGGLE_WORKER_ENTRY_PATCH_VERSION}", flush=True)
    _load_packaged_env()
    standalone = _is_standalone_mode()
    wandb_secret: dict[str, object] | None
    if standalone:
        wandb_secret = {"skipped": True, "reason": "standalone worker mode"}
    else:
        wandb_secret = _load_wandb_api_key_from_kaggle_secret()
    summary: dict[str, Any] = {
        "status": "starting",
        "cwd": str(Path.cwd()),
        "env": _safe_worker_env(),
        "worker_mode": "standalone" if standalone else "wandb",
        "wandb_secret": wandb_secret,
    }
    _write_summary(summary)
    try:
        _ensure_uv_environment(summary)
        summary["diagnostics"] = diagnostics()
        _write_summary(summary)
        _ensure_accelerator(summary)
        if standalone:
            _run_standalone_worker(summary)
        else:
            _run_wandb_worker(summary)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = _exception_message(exc)
        summary.setdefault("exit_code", _exit_code(exc))
        _write_summary(summary)
        raise


def _run_standalone_worker(summary: dict[str, Any]) -> None:
    config = _load_standalone_config()
    summary["standalone_config"] = config
    summary["standalone_overrides"] = list(_config_to_overrides(config))
    _write_summary(summary)
    _run_training_candidate(summary, config, wandb_run=None)
    if summary.get("status") == "failed":
        raise SystemExit(int(summary.get("exit_code", 1)))
    summary.setdefault("exit_code", 0)
    summary["status"] = "standalone_complete"
    _write_summary(summary)


def _run_wandb_worker(summary: dict[str, Any]) -> None:
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    if not sweep_id:
        raise SystemExit(
            "WANDB_SWEEP_ID is required for Kaggle population workers."
        )
    if not os.environ.get("WANDB_API_KEY"):
        summary["wandb_secret_retry"] = _load_wandb_api_key_from_kaggle_secret()
        _write_summary(summary)

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "WANDB_API_KEY is not available after retrying Kaggle Secrets. "
            "Add or attach a Kaggle Secret named by WANDB_API_KEY_SECRET_NAME, "
            "or provide WANDB_API_KEY through the worker environment. "
            f"Secret status: {summary.get('wandb_secret_retry') or summary.get('wandb_secret')}"
        )
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    summary["sweep_id"] = sweep_id
    summary["wandb_agent"] = {
        "project": wandb_project or "",
        "entity": wandb_entity or "",
    }
    _write_summary(summary)

    import wandb  # type: ignore

    def run_candidate() -> None:
        run = wandb.init(job_type="kaggle-population-worker")
        run_finished = False
        summary["wandb_run"] = {
            "id": str(getattr(run, "id", "")),
            "name": str(getattr(run, "name", "")),
        }
        _write_summary(summary)
        try:
            _run_training_candidate(summary, dict(wandb.config), wandb_run=run)
            run.summary["kaggle_worker_exit_code"] = summary.get("exit_code", 0)
            run.finish(exit_code=int(summary.get("exit_code", 0)))
            run_finished = True
            if int(summary.get("exit_code", 0)) != 0:
                raise SystemExit(int(summary.get("exit_code", 1)))
        except BaseException as exc:
            summary["status"] = "failed"
            summary["error"] = _exception_message(exc)
            summary.setdefault("exit_code", _exit_code(exc))
            _write_summary(summary)
            if not run_finished:
                run.finish(exit_code=_exit_code(exc))
            raise

    wandb.agent(
        sweep_id,
        function=run_candidate,
        count=1,
        entity=wandb_entity,
        project=wandb_project,
    )
    if summary.get("status") == "failed":
        raise SystemExit(int(summary.get("exit_code", 1)))
    summary.setdefault("exit_code", 0)
    summary["status"] = "agent_complete"
    _write_summary(summary)


def _run_training_candidate(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    wandb_run: Any | None,
) -> None:
    from src.orchestration.throughput import (
        calibration_grid,
        estimate_training_overrides,
        finalize_rollout_shape_overrides,
    )

    standalone = wandb_run is None
    try:
        observed = _hardware_profile(summary)
        base_overrides = _config_to_overrides(config)
        if standalone:
            base_overrides.extend(_standalone_extra_overrides())
        hydra_overrides = tuple(base_overrides)
        calibration_overrides = estimate_training_overrides(
            observed,
            _prefixed_config(config, "model."),
            _prefixed_config(config, "task."),
            hydra_overrides=hydra_overrides,
        )
        variants = calibration_grid(
            calibration_overrides,
            hydra_overrides=hydra_overrides,
        )
        settings = _calibration_settings()
        calibration_results = _run_calibration(base_overrides, variants, settings)
        selected_overrides = finalize_rollout_shape_overrides(
            _select_calibration(
                calibration_results,
                calibration_overrides,
                allow_fallback=os.environ.get(
                    "ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK"
                )
                == "1",
            ),
            hydra_overrides,
        )
        summary["hardware"] = {
            "accelerator_id": observed.accelerator_id,
            "gpu_name": observed.gpu_name,
            "memory_gb": observed.memory_gb,
        }
        summary["calibration"] = {
            "settings": settings,
            "estimated_overrides": list(calibration_overrides),
            "results": calibration_results,
            "selected_overrides": list(selected_overrides),
        }
        _write_summary(summary)
        if wandb_run is not None:
            import wandb  # type: ignore

            wandb.log(
                {
                    "kaggle_worker/observed_gpu": observed.gpu_name,
                    "kaggle_worker/observed_memory_gb": observed.memory_gb,
                    "kaggle_worker/calibration_results": calibration_results,
                    "kaggle_worker/selected_overrides": list(selected_overrides),
                },
                step=0,
            )
        overrides = list(base_overrides)
        overrides.extend(selected_overrides)
        overrides = _apply_smoke_training_overrides(overrides)
        if standalone:
            overrides.extend(
                [
                    "telemetry.wandb.enabled=false",
                    "telemetry.wandb.log_artifacts=false",
                    "artifacts.artifact_pipeline.docker_validation_async=false",
                    "artifacts.replay.enabled=false",
                ]
            )
        else:
            overrides.extend(
                [
                    "telemetry.wandb.enabled=true",
                    "telemetry.wandb.log_artifacts=true",
                    "artifacts.artifact_pipeline.docker_validation_async=false",
                    "artifacts.replay.enabled=false",
                ]
            )
        command = _render_worker_train_command(tuple(overrides))
        summary["final_command"] = command
        summary["selected_overrides"] = overrides
        _write_summary(summary)
        print("worker command:", " ".join(command), flush=True)
        env = _subprocess_env()
        if wandb_run is not None:
            env.setdefault("WANDB_RUN_ID", wandb_run.id)
            env.setdefault("WANDB_RESUME", "allow")
        completed = subprocess.run(command, check=False, text=True, env=env)
        summary["exit_code"] = completed.returncode
        summary["checkpoint_paths"] = _collect_checkpoint_paths()
        summary["status"] = "completed" if completed.returncode == 0 else "failed"
        _write_summary(summary)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = _exception_message(exc)
        summary.setdefault("exit_code", _exit_code(exc))
        summary.setdefault("checkpoint_paths", _collect_checkpoint_paths())
        _write_summary(summary)
        raise



def diagnostics() -> dict[str, Any]:
    """Collect runtime diagnostics without importing JAX in this parent process.

    The verified JAX CUDA configuration only works reliably when LD_LIBRARY_PATH
    is present at process start. Importing JAX after mutating LD_LIBRARY_PATH in
    this already-running parent process can fail even when child processes work.
    Therefore all JAX/Flax checks run in a fresh worker-venv Python subprocess.
    """

    _activate_local_venv()
    payload: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": getattr(sys, "base_prefix", ""),
        "worker_entry_patch": KAGGLE_WORKER_ENTRY_PATCH_VERSION,
        "diagnostics_mode": "jax-subprocess-probe-v9",
        "accelerator_id": os.environ.get("KAGGLE_ACCELERATOR_ID", ""),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS", ""),
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "worker_venv": os.environ.get(_WORKER_VENV_ENV, ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "nvidia_smi": _run_optional(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ),
    }
    probe = _run_jax_diagnostics_subprocess()
    payload["jax_subprocess_probe"] = probe
    if int(probe.get("returncode", 1)) == 0:
        parsed = probe.get("parsed")
        if isinstance(parsed, dict):
            payload["jax_version"] = parsed.get("jax_version", "")
            payload["jax_default_backend"] = parsed.get("jax_default_backend", "")
            payload["jax_devices"] = list(parsed.get("jax_devices", []))
            payload["jax_cuda_devices"] = list(parsed.get("jax_cuda_devices", []))
            payload["jax_platforms"] = list(parsed.get("jax_platforms", []))
            payload["flax_version"] = parsed.get("flax_version", "")
            payload["flax_linen_import"] = parsed.get("flax_linen_import", "")
    else:
        payload["jax_error"] = str(
            probe.get("stderr_tail") or probe.get("stdout_tail") or probe
        )
    return payload


def _run_jax_diagnostics_subprocess() -> dict[str, object]:
    command = [str(_venv_python()), "-c", _JAX_DIAGNOSTICS_PROBE_CODE]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    result: dict[str, object] = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout, limit=6000),
        "stderr_tail": _tail(completed.stderr, limit=6000),
    }
    if completed.stdout:
        for line in reversed(completed.stdout.strip().splitlines()):
            try:
                result["parsed"] = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return result


_JAX_DIAGNOSTICS_PROBE_CODE = r"""
import json
import os
import sys


def platform_aliases(devices):
    platforms = set()
    for device in devices:
        platform = str(getattr(device, "platform", "") or "")
        if platform:
            platforms.add(platform)
        text = str(device).lower()
        if "cuda" in text:
            platforms.add("cuda")
            platforms.add("gpu")
        if platform == "gpu":
            platforms.add("cuda")
        if platform == "cuda":
            platforms.add("gpu")
    return sorted(platforms)

payload = {
    "probe_python": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": getattr(sys, "base_prefix", ""),
    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
    "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}
try:
    import flax
    import flax.linen as nn  # noqa: F401
    payload["flax_version"] = str(getattr(flax, "__version__", ""))
    payload["flax_linen_import"] = "ok"

    import jax
    devices = list(jax.devices())
    payload["jax_version"] = str(getattr(jax, "__version__", ""))
    payload["jax_default_backend"] = str(jax.default_backend())
    payload["jax_devices"] = [str(device) for device in devices]
    payload["jax_platforms"] = platform_aliases(devices)
    try:
        cuda_devices = list(jax.devices("cuda"))
    except Exception as exc:
        payload["jax_cuda_error"] = repr(exc)
        cuda_devices = []
    payload["jax_cuda_devices"] = [str(device) for device in cuda_devices]
    payload["jax_platforms"] = sorted(
        set(payload["jax_platforms"]) | set(platform_aliases(cuda_devices))
    )
    if cuda_devices:
        payload["jax_platforms"] = sorted(set(payload["jax_platforms"]) | {"cuda", "gpu"})
    x = jax.numpy.arange(8.0)
    payload["probe_sum"] = float(jax.numpy.sum(x))
    print(json.dumps(payload, sort_keys=True))
except BaseException as exc:
    payload["error"] = repr(exc)
    print(json.dumps(payload, sort_keys=True))
    raise
"""

def _ensure_accelerator(summary: dict[str, Any]) -> None:
    if os.environ.get("ORBIT_WARS_KAGGLE_ALLOW_CPU") == "1":
        return
    from src.orchestration.accelerators import is_tpu_accelerator

    diagnostics_payload = summary.get("diagnostics", {})
    platforms = set(diagnostics_payload.get("jax_platforms", []))
    accelerator_id = os.environ.get("KAGGLE_ACCELERATOR_ID", "")
    if is_tpu_accelerator(accelerator_id):
        if "tpu" not in platforms:
            raise SystemExit(
                f"JAX TPU backend required for Kaggle worker. Platforms: {platforms}; "
                f"diagnostics={diagnostics_payload}"
            )
        return
    if _accelerator_requests_nvidia(accelerator_id):
        if not ({"gpu", "cuda"} & platforms):
            raise SystemExit(
                f"JAX GPU/CUDA backend required for Kaggle worker. "
                f"Platforms: {platforms}; diagnostics={diagnostics_payload}"
            )
        return


def _hardware_profile(summary: dict[str, Any]) -> Any:
    from src.orchestration.accelerators import default_memory_gb, is_tpu_accelerator
    from src.orchestration.throughput import HardwareProfile

    accelerator_id = os.environ.get("KAGGLE_ACCELERATOR_ID", "unknown")
    if is_tpu_accelerator(accelerator_id):
        return HardwareProfile(
            accelerator_id=accelerator_id,
            gpu_name="tpu-v5e-8",
            memory_gb=default_memory_gb(accelerator_id),
        )

    smi = str(summary.get("diagnostics", {}).get("nvidia_smi", ""))
    gpu_name = smi.split(",", 1)[0].strip() if smi.strip() else "unknown"
    memory_gb = default_memory_gb(accelerator_id, fallback=16.0)
    if "," in smi:
        memory_text = smi.split(",", 1)[1].strip().lower().replace("mib", "")
        try:
            memory_gb = float(memory_text) / 1024.0
        except ValueError:
            pass
    return HardwareProfile(
        accelerator_id=accelerator_id,
        gpu_name=gpu_name,
        memory_gb=memory_gb,
    )


def _config_to_overrides(config: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for key, value in sorted(config.items()):
        if not isinstance(key, str):
            continue
        if key.startswith("kaggle_worker."):
            continue
        if "." not in key and key not in _HYDRA_CONFIG_GROUP_KEYS:
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


def _calibration_command(
    overrides: list[str], *, warmup: int, updates: int
) -> list[str]:
    return [
        str(_venv_python()),
        str(SCRIPT_DIR / "benchmark_jax_rl.py"),
        "--overrides",
        *overrides,
        "--warmup",
        str(warmup),
        "--updates",
        str(updates),
    ]


def _render_worker_train_command(overrides: tuple[str, ...]) -> list[str]:
    return [str(_venv_python()), "-m", "src.train", *overrides]


def _venv_python() -> Path:
    candidate = _worker_venv() / "bin" / "python"
    if not candidate.exists():
        raise SystemExit("Worker venv python is missing after bootstrap.")
    return candidate


def _run_calibration(
    base_overrides: list[str],
    variants: list[tuple[str, ...]],
    settings: dict[str, int],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    warmup = settings["warmup"]
    updates = settings["updates"]
    timeout_seconds = settings["timeout_seconds"]
    selected_variants = variants[: settings["max_variants"]]
    for index, variant in enumerate(selected_variants):
        overrides = [
            *_without_training_shape_overrides(base_overrides),
            *variant,
            "telemetry.wandb.enabled=false",
            "artifacts.artifact_pipeline.enabled=false",
            "artifacts.replay.enabled=false",
        ]
        command = _calibration_command(
            overrides,
            warmup=warmup,
            updates=updates,
        )
        print("calibration command:", " ".join(command), flush=True)
        result: dict[str, object] = {
            "index": index,
            "overrides": list(variant),
            "timeout_seconds": timeout_seconds,
        }
        env = _subprocess_env()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            result["returncode"] = 124
            result["error"] = f"calibration timed out after {timeout_seconds}s"
            result["stdout_tail"] = _tail(exc.stdout)
            result["stderr_tail"] = _tail(exc.stderr)
            results.append(result)
            continue
        result["returncode"] = completed.returncode
        result["stdout_tail"] = _tail(completed.stdout)
        result["stderr_tail"] = _tail(completed.stderr)
        if completed.returncode == 0:
            try:
                result.update(json.loads(completed.stdout.strip().splitlines()[-1]))
            except (IndexError, json.JSONDecodeError) as exc:
                result["error"] = f"failed to parse benchmark output: {exc}"
        else:
            result["error"] = _tail(completed.stderr or completed.stdout)
            print(
                f"calibration variant {index} failed (rc={completed.returncode}):",
                result["error"],
                flush=True,
            )
        results.append(result)
    return results


def _select_calibration(
    results: list[dict[str, object]],
    fallback: tuple[str, ...],
    *,
    allow_fallback: bool,
) -> tuple[str, ...]:
    successful = [
        result
        for result in results
        if int(result.get("returncode", 1)) == 0 and "samples_per_sec" in result
    ]
    if not successful:
        if not allow_fallback:
            raise SystemExit(
                "All calibration variants failed; set "
                "ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK=1 to use estimator fallback."
            )
        return fallback
    best = max(successful, key=lambda item: float(item.get("samples_per_sec", 0.0)))
    return tuple(str(item) for item in best.get("overrides", fallback))


def _calibration_settings() -> dict[str, int]:
    smoke = _is_smoke_run()
    return {
        "warmup": _env_int("ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP", 0, minimum=0),
        "updates": _env_int("ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES", 1, minimum=1),
        "max_variants": _env_int(
            "ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS",
            1 if smoke else 3,
            minimum=1,
        ),
        "timeout_seconds": _env_int(
            "ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS",
            600 if smoke else 1800,
            minimum=1,
        ),
    }


def _is_smoke_run() -> bool:
    return os.environ.get("ORBIT_WARS_KAGGLE_RUN_TYPE", "").strip().lower() == "smoke"


def _is_standalone_mode() -> bool:
    return (
        os.environ.get("ORBIT_WARS_KAGGLE_WORKER_MODE", "").strip().lower()
        == "standalone"
    )


def _load_standalone_config() -> dict[str, Any]:
    from src.orchestration.wandb_sweeps import load_standalone_config

    sweep_yaml = os.environ.get("WANDB_SWEEP_YAML", "").strip()
    if not sweep_yaml:
        raise SystemExit(
            "WANDB_SWEEP_YAML is required for standalone Kaggle workers."
        )
    path = Path(sweep_yaml)
    if not path.is_file():
        raise SystemExit(f"Packaged sweep YAML is missing: {path}")
    return load_standalone_config(path)


def _standalone_extra_overrides() -> list[str]:
    raw = os.environ.get("ORBIT_WARS_KAGGLE_STANDALONE_OVERRIDES", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise SystemExit(
            "ORBIT_WARS_KAGGLE_STANDALONE_OVERRIDES must be a JSON list of Hydra overrides."
        )
    return [str(item) for item in parsed if str(item).strip()]


def _collect_checkpoint_paths() -> list[str]:
    root = Path("outputs")
    if not root.exists():
        return []
    return sorted(str(path.resolve()) for path in root.rglob("jax_ckpt*.pkl"))


def _apply_smoke_training_overrides(overrides: list[str]) -> list[str]:
    """Apply short training defaults for smoke validation runs."""

    if not _is_smoke_run():
        return overrides
    filtered = [
        item
        for item in overrides
        if not item.startswith("training.total_updates=")
    ]
    filtered.append("training.total_updates=10")
    return filtered


def _without_training_shape_overrides(overrides: list[str]) -> list[str]:
    shape_keys = {
        "training.num_envs",
        "training.rollout_steps",
        "training.minibatch_size",
        "training.rollout_microbatch_envs",
        "training.update_chunk_rows_min",
        "training.update_chunk_rows_max",
    }
    return [item for item in overrides if item.split("=", 1)[0] not in shape_keys]


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


def _load_wandb_api_key_from_kaggle_secret() -> dict[str, object]:
    """Resolve the W&B key from Kaggle Secrets instead of packaging it.

    Kaggle's secrets service can be temporarily unavailable immediately after a
    CLI-pushed kernel starts. Retry because manual reruns often succeed once the
    service has initialized for the kernel.
    """

    if os.environ.get("WANDB_API_KEY"):
        return {"loaded": True, "source": "environment"}

    secret_name = os.environ.get("WANDB_API_KEY_SECRET_NAME", "WANDB_API_KEY").strip()
    if not secret_name:
        return {"loaded": False, "secret_name": "", "error": "no secret name set"}

    max_wait_seconds = _env_int(
        "ORBIT_WARS_KAGGLE_SECRET_WAIT_SECONDS",
        30 if _is_smoke_run() else 120,
        minimum=0,
    )
    sleep_seconds = 2.0
    deadline = time.monotonic() + max_wait_seconds
    attempts: list[dict[str, object]] = []

    while True:
        attempt_index = len(attempts) + 1
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
        except ImportError as exc:
            error = f"kaggle_secrets unavailable: {exc}"
            attempts.append({"attempt": attempt_index, "error": error})
            return {
                "loaded": False,
                "secret_name": secret_name,
                "attempts": attempts,
                "error": error,
            }

        try:
            value = UserSecretsClient().get_secret(secret_name)
        except Exception as exc:
            error = f"could not read Kaggle secret: {exc}"
            attempts.append({"attempt": attempt_index, "error": error})
            if time.monotonic() >= deadline:
                return {
                    "loaded": False,
                    "secret_name": secret_name,
                    "attempts": attempts,
                    "error": error,
                }
            print(
                f"WANDB secret unavailable on attempt {attempt_index}; "
                f"retrying in {sleep_seconds:.1f}s: {error}",
                flush=True,
            )
            time.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 1.5, 15.0)
            continue

        if not value:
            error = "Kaggle secret is empty"
            attempts.append({"attempt": attempt_index, "error": error})
            return {
                "loaded": False,
                "secret_name": secret_name,
                "attempts": attempts,
                "error": error,
            }

        os.environ["WANDB_API_KEY"] = str(value)
        return {
            "loaded": True,
            "secret_name": secret_name,
            "source": "kaggle_secret",
            "attempt_count": attempt_index,
        }

def _ensure_uv_environment(summary: dict[str, Any]) -> None:
    if os.environ.get(_WORKER_VENV_READY_ENV) == "1":
        _activate_local_venv()
        return
    if shutil.which("uv") is None:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv"],
            check=False,
            capture_output=True,
            text=True,
        )
        summary["uv_install"] = _completed_summary(completed)
        _write_summary(summary)
        if completed.returncode != 0:
            raise SystemExit("Failed to install uv inside Kaggle worker.")
    if Path("pyproject.toml").exists():
        _reset_worker_venv()
        sync = _bootstrap_uv_environment(os.environ.get("KAGGLE_ACCELERATOR_ID", ""))
        summary["uv_sync"] = {
            "returncode": sync["returncode"],
            "tpu_backend": sync["tpu_backend"],
            "steps": sync["steps"],
        }
        _write_summary(summary)
        if int(sync["returncode"]) != 0:
            log_bootstrap_failure(sync["steps"])
            raise SystemExit(format_bootstrap_failure(sync["steps"]))
        _activate_local_venv()
        _reexec_with_venv_python()


def _reexec_with_venv_python() -> None:
    """Run the worker under the verified managed venv.

    The CUDA JAX probe succeeded with a specific venv + LD_LIBRARY_PATH. Carry
    that exact class of environment into the re-executed worker instead of
    re-applying generic runtime helpers.
    """

    venv_python = _venv_python().resolve()
    if Path(sys.executable).resolve() == venv_python:
        os.environ[_WORKER_VENV_READY_ENV] = "1"
        _activate_local_venv()
        return
    env = _subprocess_env()
    env[_WORKER_VENV_READY_ENV] = "1"
    script_path = Path(__file__).resolve()
    os.execve(str(venv_python), [str(venv_python), str(script_path), *sys.argv[1:]], env)


def _bootstrap_uv_environment(accelerator_id: str) -> dict[str, object]:
    """Install project deps before importing ``src`` (Kaggle has no venv yet)."""

    sync = sync_kaggle_worker_environment(accelerator_id)
    return {
        "returncode": sync.returncode,
        "tpu_backend": sync.tpu_backend,
        "steps": [dict(step) for step in sync.steps],
    }


def _reset_worker_venv() -> None:
    """Recreate the worker venv every run; Kaggle working dirs persist stale JAX installs."""

    venv = _worker_venv()
    if venv.exists():
        shutil.rmtree(venv)
    backend = (
        "tpu"
        if _accelerator_requests_tpu(os.environ.get("KAGGLE_ACCELERATOR_ID", ""))
        else "gpu"
    )
    _JAX_BACKEND_MARKER.write_text(f"{backend}\n", encoding="utf-8")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    _apply_verified_worker_runtime_env(env=env)
    return env


def _accelerator_requests_tpu(accelerator_id: str) -> bool:
    normalized = accelerator_id.strip().lower()
    return bool(normalized) and normalized.startswith("tpu")


def _accelerator_requests_nvidia(accelerator_id: str) -> bool:
    normalized = accelerator_id.strip().lower()
    return bool(normalized) and normalized.startswith("nvidia")


def _activate_local_venv() -> None:
    _apply_verified_worker_runtime_env(env=os.environ)
    for site_packages in sorted(_worker_venv().glob("lib/python*/site-packages")):
        site_text = str(site_packages.resolve())
        if site_text not in sys.path:
            sys.path.insert(0, site_text)


def _apply_verified_worker_runtime_env(*, env: dict[str, str]) -> None:
    """Apply the runtime env that matches the successful JAX CUDA probe."""

    venv = _worker_venv().resolve()
    bin_dir = venv / "bin"
    env[_WORKER_VENV_ENV] = str(venv)
    env["VIRTUAL_ENV"] = str(venv)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    if bin_dir.exists():
        env["PATH"] = _prepend_path(str(bin_dir), env.get("PATH", ""))

    accelerator_id = env.get("KAGGLE_ACCELERATOR_ID", os.environ.get("KAGGLE_ACCELERATOR_ID", ""))
    env.pop("JAX_PLATFORM_NAME", None)
    if _accelerator_requests_tpu(accelerator_id):
        env["JAX_PLATFORMS"] = "tpu,cpu"
    elif _accelerator_requests_nvidia(accelerator_id):
        env["JAX_PLATFORMS"] = "cuda,cpu"
        merged = [
            *_cuda_wheel_library_dirs(venv),
            *_cuda_driver_library_dirs(env.get("LD_LIBRARY_PATH", "")),
        ]
        if merged:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(merged))
    elif os.environ.get("ORBIT_WARS_FORCE_JAX_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        env["JAX_PLATFORMS"] = "cpu"


def _cuda_wheel_library_dirs(venv: Path) -> list[str]:
    dirs: list[str] = []
    for site_packages in sorted(venv.glob("lib/python*/site-packages")):
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                dirs.append(str(lib_dir.resolve()))
    return _dedupe(dirs)


def _cuda_driver_library_dirs(existing_ld_library_path: str = "") -> list[str]:
    candidates = [*_CUDA_DRIVER_LIBRARY_DIR_CANDIDATES]
    candidates.extend(item for item in existing_ld_library_path.split(os.pathsep) if item)
    candidates.extend(item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item)
    result: list[str] = []
    for item in candidates:
        path = Path(item)
        if not path.exists() or not path.is_dir():
            continue
        if list(path.glob("libcuda.so*")) or list(path.glob("libnvidia-ml.so*")):
            result.append(str(path.resolve()))
    return _dedupe(result)


def _jax_platform_aliases(devices: list[object]) -> list[str]:
    platforms: set[str] = set()
    for device in devices:
        platform = str(getattr(device, "platform", "") or "")
        if platform:
            platforms.add(platform)
        device_text = str(device).lower()
        if "cuda" in device_text:
            platforms.add("cuda")
            platforms.add("gpu")
        if platform == "gpu":
            platforms.add("cuda")
        if platform == "cuda":
            platforms.add("gpu")
    return sorted(platforms)


def _prepend_path(prefix: str, existing: str) -> str:
    return os.pathsep.join(_dedupe([prefix, *(item for item in existing.split(os.pathsep) if item)]))


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _worker_venv() -> Path:
    return Path(os.environ.get(_WORKER_VENV_ENV, ".venv"))


def _completed_summary(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    return {
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _write_summary(summary: dict[str, Any]) -> None:
    Path("worker-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _safe_worker_env() -> dict[str, str]:
    keys = (
        "KAGGLE_ACCELERATOR_ID",
        "ORBIT_WARS_KAGGLE_ALLOW_CPU",
        "ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK",
        "ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS",
        "ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS",
        "ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES",
        "ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP",
        "ORBIT_WARS_KAGGLE_RUN_TYPE",
        "ORBIT_WARS_KAGGLE_STANDALONE_OVERRIDES",
        "ORBIT_WARS_KAGGLE_TRUST_BASE_JAX",
        "ORBIT_WARS_KAGGLE_WORKER_MODE",
        "WANDB_ENTITY",
        "WANDB_API_KEY_SECRET_NAME",
        "WANDB_PROJECT",
        "WANDB_RESUME",
        "WANDB_SWEEP_ID",
        "WANDB_SWEEP_YAML",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(value, minimum)


def _tail(text: object, *, limit: int = 2000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return str(text).strip()[-limit:]


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        return str(exc.code)
    return repr(exc)


def _exit_code(exc: BaseException) -> int:
    if isinstance(exc, SystemExit) and isinstance(exc.code, int):
        return exc.code
    return 1


if __name__ == "__main__":
    main()
