from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from src.orchestration.remote_package import hydra_overrides_from_worker_env

RemoteHost = Literal["kaggle", "colab"]

WORKER_SUMMARY_NAME = "worker-summary.json"
WORKER_ENV_NAME = "worker-env.json"
WORKER_VENV_ENV = "ORBIT_WARS_WORKER_VENV"
WORKER_VENV_READY_ENV = "ORBIT_WARS_WORKER_VENV_READY"
REMOTE_WORKER_PATCH_VERSION = "shared-bootstrap-v1"


def load_packaged_env(path: Path | None = None) -> None:
    """Load packaged worker environment values without overwriting existing env."""

    env_path = path or Path(WORKER_ENV_NAME)
    if not env_path.exists():
        return
    payload = json.loads(env_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, list):
            os.environ.setdefault(str(key), json.dumps(value))
            continue
        if value:
            os.environ.setdefault(str(key), str(value))


def write_summary(summary: dict[str, Any], path: Path | None = None) -> None:
    """Persist worker progress to ``worker-summary.json``."""

    target = path or Path(WORKER_SUMMARY_NAME)
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def load_hydra_overrides() -> list[str]:
    """Return Hydra overrides from ``HYDRA_OVERRIDES`` in the active environment."""

    raw = os.environ.get("HYDRA_OVERRIDES", "").strip()
    if not raw:
        return []
    return hydra_overrides_from_worker_env({"HYDRA_OVERRIDES": raw})


def render_train_command(
    overrides: tuple[str, ...],
    *,
    host: RemoteHost,
    venv_python: Path | None = None,
) -> list[str]:
    """Render the training subprocess command for a remote host."""

    if host == "colab":
        return ["uv", "run", "ow", "train", *overrides]
    python = venv_python or default_venv_python()
    return [str(python), "-m", "src.train", *overrides]


def collect_checkpoint_paths(root: Path | None = None) -> list[str]:
    """Collect checkpoint paths produced under ``outputs/``."""

    search_root = root or Path("outputs")
    if not search_root.exists():
        return []
    return sorted(str(path.resolve()) for path in search_root.rglob("jax_ckpt*.pkl"))


def run_train_subprocess(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the training command and return the completed process."""

    return subprocess.run(
        command,
        check=False,
        text=True,
        env=env or os.environ.copy(),
    )


def ensure_uv_available(summary: dict[str, Any]) -> None:
    """Install ``uv`` when it is missing on the remote VM."""

    if shutil.which("uv") is not None:
        return
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "uv"],
        check=False,
        capture_output=True,
        text=True,
    )
    summary["uv_install"] = completed_summary(completed)
    write_summary(summary)
    if completed.returncode != 0:
        raise SystemExit("Failed to install uv inside remote worker.")


def sync_colab_worker_environment() -> dict[str, object]:
    """Run ``uv sync --group dev`` for Colab workers."""

    command = ["uv", "sync", "--group", "dev"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def run_jax_gpu_check(
    *,
    host: RemoteHost,
    summary: dict[str, Any],
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Verify JAX sees an accelerator; return diagnostics payload."""

    diagnostics_payload = run_jax_diagnostics_subprocess()
    summary["diagnostics"] = diagnostics_payload
    write_summary(summary)

    if allow_cpu:
        return diagnostics_payload

    if host == "colab":
        _require_colab_gpu(diagnostics_payload)
    else:
        _require_kaggle_accelerator(diagnostics_payload)
    return diagnostics_payload


def run_colab_worker() -> None:
    """Bootstrap and run a Colab standalone worker from ``worker-env.json``."""

    print(f"ORBIT_WARS_REMOTE_WORKER_PATCH={REMOTE_WORKER_PATCH_VERSION}", flush=True)
    load_packaged_env()
    summary: dict[str, Any] = {
        "status": "starting",
        "cwd": str(Path.cwd()),
        "host": "colab",
        "worker_mode": os.environ.get("ORBIT_WARS_COLAB_WORKER_MODE", "standalone"),
        "env": safe_worker_env(
            keys=(
                "ORBIT_WARS_COLAB_WORKER_MODE",
                "ORBIT_WARS_COLAB_TRUST_BASE_JAX",
                "HYDRA_OVERRIDES",
            )
        ),
    }
    write_summary(summary)
    try:
        ensure_uv_available(summary)
        sync_result = sync_colab_worker_environment()
        summary["uv_sync"] = sync_result
        write_summary(summary)
        if int(sync_result.get("returncode", 1)) != 0:
            raise SystemExit("uv sync failed inside Colab worker.")

        allow_cpu = os.environ.get("ORBIT_WARS_COLAB_ALLOW_CPU", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        run_jax_gpu_check(host="colab", summary=summary, allow_cpu=allow_cpu)

        overrides = tuple(load_hydra_overrides())
        summary["hydra_overrides"] = list(overrides)
        command = render_train_command(overrides, host="colab")
        summary["final_command"] = command
        write_summary(summary)
        print("worker command:", " ".join(command), flush=True)

        completed = run_train_subprocess(command)
        summary["exit_code"] = completed.returncode
        summary["checkpoint_paths"] = collect_checkpoint_paths()
        summary["status"] = "completed" if completed.returncode == 0 else "failed"
        write_summary(summary)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

        summary["status"] = "colab_complete"
        write_summary(summary)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = exception_message(exc)
        summary.setdefault("exit_code", exit_code(exc))
        summary.setdefault("checkpoint_paths", collect_checkpoint_paths())
        write_summary(summary)
        raise


def run_jax_diagnostics_subprocess() -> dict[str, Any]:
    """Run JAX diagnostics in a fresh subprocess to avoid parent LD_LIBRARY_PATH issues."""

    command = [sys.executable, "-c", _JAX_DIAGNOSTICS_PROBE_CODE]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    payload: dict[str, Any] = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": tail(completed.stdout, limit=6000),
        "stderr_tail": tail(completed.stderr, limit=6000),
        "remote_worker_patch": REMOTE_WORKER_PATCH_VERSION,
    }
    if completed.stdout:
        for line in reversed(completed.stdout.strip().splitlines()):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload.update(parsed)
                break
    if int(payload["returncode"]) != 0 and "jax_error" not in payload:
        payload["jax_error"] = str(
            payload.get("stderr_tail") or payload.get("stdout_tail") or payload
        )
    return payload


def default_venv_python() -> Path:
    """Return the managed worker venv python executable."""

    candidate = worker_venv() / "bin" / "python"
    if not candidate.exists():
        raise SystemExit("Worker venv python is missing after bootstrap.")
    return candidate


def worker_venv() -> Path:
    return Path(os.environ.get(WORKER_VENV_ENV, ".venv"))


def safe_worker_env(*, keys: tuple[str, ...]) -> dict[str, str]:
    return {key: os.environ[key] for key in keys if key in os.environ}


def completed_summary(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    return {
        "returncode": completed.returncode,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def tail(text: object, *, limit: int = 2000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return str(text).strip()[-limit:]


def exception_message(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        return str(exc.code)
    return repr(exc)


def exit_code(exc: BaseException) -> int:
    if isinstance(exc, SystemExit) and isinstance(exc.code, int):
        return exc.code
    return 1


def _require_colab_gpu(diagnostics_payload: dict[str, Any]) -> None:
    platforms = set(diagnostics_payload.get("jax_platforms", []))
    if {"gpu", "cuda"} & platforms:
        return
    raise SystemExit(
        "JAX GPU/CUDA backend required for Colab worker. "
        f"Platforms: {sorted(platforms)}; diagnostics={diagnostics_payload}"
    )


def _require_kaggle_accelerator(diagnostics_payload: dict[str, Any]) -> None:
    from src.orchestration.accelerators import is_tpu_accelerator

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


def _accelerator_requests_nvidia(accelerator_id: str) -> bool:
    normalized = accelerator_id.strip().lower()
    return bool(normalized) and normalized.startswith("nvidia")


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
