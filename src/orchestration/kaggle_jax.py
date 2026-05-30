from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from src.orchestration.accelerators import is_tpu_accelerator

LIBTPU_FIND_LINKS = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
KAGGLE_WORKER_VENV_ENV = "ORBIT_WARS_WORKER_VENV"
KAGGLE_JAX_BOOTSTRAP_PATCH_VERSION = "trust-base-jax-v9"

# Kaggle base image currently reports working JAX CUDA 0.7.2 on T4.
# Keep the worker venv aligned with that unless explicitly overridden.
# Override without code changes via:
#   ORBIT_WARS_KAGGLE_JAX_CUDA_SPEC='jax[cuda12]==<version>'
# or:
#   ORBIT_WARS_KAGGLE_JAX_CUDA_VERSION='<version>'
KAGGLE_DEFAULT_JAX_CUDA_VERSION = "0.7.2"
KAGGLE_DEFAULT_JAX_CUDA_EXTRA = "cuda12"

KAGGLE_CUDA_JAX_PLUGINS: tuple[str, ...] = (
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
)
KAGGLE_CUDA13_PACKAGES: tuple[str, ...] = (
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "nvidia-cublas-cu13",
    "nvidia-cuda-cccl-cu13",
    "nvidia-cuda-cupti-cu13",
    "nvidia-cuda-nvcc-cu13",
    "nvidia-cuda-nvrtc-cu13",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cudnn-cu13",
    "nvidia-cufft-cu13",
    "nvidia-cusolver-cu13",
    "nvidia-cusparse-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink-cu13",
    "nvidia-nvshmem-cu13",
)
KAGGLE_TPU_JAX_PACKAGES: tuple[str, ...] = ("libtpu",)

# Driver libraries are mounted by the Kaggle/NVIDIA host. CUDA runtime libraries
# may come from either the base image or pip wheels, but libcuda/libnvidia-ml do
# not come from pip wheels.
KAGGLE_CUDA_DRIVER_LIBRARY_DIR_CANDIDATES: tuple[str, ...] = (
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
    "/usr/local/cuda/compat",
    "/usr/lib/x86_64-linux-gnu",
)


@dataclass(frozen=True, slots=True)
class UvEnvironmentSync:
    """Result summary for Kaggle-side dependency setup."""

    returncode: int
    tpu_backend: bool
    steps: tuple[dict[str, object], ...]


def first_failed_bootstrap_step(
    steps: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> dict[str, object] | None:
    """Return the first bootstrap step with a non-zero return code."""

    for step in steps:
        returncode = step.get("returncode")
        if returncode is None:
            continue
        if int(returncode) != 0:
            return step
    return None


def format_bootstrap_failure(
    steps: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> str:
    """Render a concise bootstrap failure message for worker exits."""

    failed = first_failed_bootstrap_step(steps)
    if failed is None:
        return "Worker bootstrap failed during dependency setup."
    name = str(failed.get("name", "unknown"))
    stderr_tail = str(failed.get("stderr_tail", "")).strip()
    stdout_tail = str(failed.get("stdout_tail", "")).strip()
    detail = stderr_tail or stdout_tail or "no subprocess output captured"
    return f"Worker bootstrap failed at step {name!r}: {detail}"


def log_bootstrap_failure(
    steps: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> None:
    """Print bootstrap step summaries to stdout for Kaggle kernel logs."""

    print("bootstrap steps:", flush=True)
    for step in steps:
        name = str(step.get("name", "unknown"))
        returncode = step.get("returncode")
        optional = " optional" if step.get("optional") else ""
        print(f"  - {name}: returncode={returncode}{optional}", flush=True)
    failed = first_failed_bootstrap_step(steps)
    if failed is None:
        return
    name = str(failed.get("name", "unknown"))
    print(f"bootstrap failed at step: {name}", flush=True)
    for key in (
        "stdout_tail",
        "stderr_tail",
        "installed_jax_packages",
        "installed_nvidia_packages",
        "cuda_wheel_library_dirs",
        "cuda_driver_library_dirs",
        "plugin_dirs",
        "jax_cuda_spec",
    ):
        text = str(failed.get(key, "")).strip()
        if text:
            print(f"bootstrap {key}:", text, flush=True)


def jax_platform_for_accelerator(accelerator_id: str) -> str | None:
    """Return the JAX platform pin for a Kaggle accelerator, if any."""

    if is_tpu_accelerator(accelerator_id):
        return "tpu"
    if os.environ.get("ORBIT_WARS_FORCE_JAX_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "cpu"
    normalized = accelerator_id.strip().lower()
    if normalized.startswith("nvidia"):
        return "cuda"
    return None


def sync_kaggle_worker_environment(accelerator_id: str) -> UvEnvironmentSync:
    """Prepare the Kaggle worker environment using a two-phase JAX bootstrap.

    NVIDIA policy:
      1. Probe the base Kaggle Python for JAX CUDA before touching deps.
      2. Run project dependency sync.
      3. Probe the actual worker interpreter used by training.
      4. If worker JAX CUDA is missing, install one pinned JAX CUDA package spec.
      5. Verify CUDA or fall back to JAX CPU only when the run mode allows it.

    TPU policy remains separate because TPU needs libtpu/JAX TPU packages.
    """

    print(
        f"ORBIT_WARS_JAX_BOOTSTRAP_PATCH={KAGGLE_JAX_BOOTSTRAP_PATCH_VERSION}",
        flush=True,
    )
    tpu_backend = is_tpu_accelerator(accelerator_id)
    nvidia_backend = accelerator_id.strip().lower().startswith("nvidia")
    steps: list[dict[str, object]] = []
    trust_base_jax = False

    if nvidia_backend:
        print("bootstrap starting step: probe_base_python_jax_cuda", flush=True)
        base_probe = _probe_base_python_jax_cuda()
        steps.append(_optional_probe_step(base_probe))
        _print_probe_summary(base_probe)
        trust_base_jax = _trust_base_jax() and int(base_probe.get("returncode", 1)) == 0
        if trust_base_jax:
            print(
                "bootstrap trust-base-jax: using Kaggle base-image JAX CUDA; "
                "skipping pinned reinstall unless worker probe fails",
                flush=True,
            )

    print("bootstrap starting step: ensure_worker_venv", flush=True)
    venv_step = _ensure_worker_venv(system_site_packages=trust_base_jax and nvidia_backend)
    steps.append(venv_step)
    if int(venv_step.get("returncode", 1)) != 0:
        return UvEnvironmentSync(
            returncode=int(venv_step.get("returncode", 1)),
            tpu_backend=tpu_backend,
            steps=tuple(steps),
        )

    print("bootstrap starting step: uv_sync", flush=True)
    sync = subprocess.run(
        ["uv", "sync", "--no-dev"],
        check=False,
        capture_output=True,
        text=True,
        env=_uv_bootstrap_env(),
    )
    print(f"bootstrap finished step: uv_sync returncode={sync.returncode}", flush=True)
    steps.append(_completed_step("uv_sync", sync))
    if sync.returncode != 0:
        return UvEnvironmentSync(
            returncode=sync.returncode,
            tpu_backend=tpu_backend,
            steps=tuple(steps),
        )

    if tpu_backend:
        print("bootstrap starting step: uv_pip_uninstall_cuda_jax_plugins", flush=True)
        uninstall = _uninstall_cuda_jax_plugins()
        steps.append(_completed_step("uv_pip_uninstall_cuda_jax_plugins", uninstall))
        print("bootstrap starting step: uv_pip_install_jax_tpu", flush=True)
        reinstall = _install_jax_tpu()
        steps.append(_completed_step("uv_pip_install_jax_tpu", reinstall))
        if reinstall.returncode != 0:
            return UvEnvironmentSync(
                returncode=reinstall.returncode,
                tpu_backend=True,
                steps=tuple(steps),
            )
        print("bootstrap starting step: verify_tpu_jax", flush=True)
        verify_tpu = _probe_worker_jax_platform("tpu", name="verify_tpu_jax")
        steps.append(verify_tpu)
        return UvEnvironmentSync(
            returncode=int(verify_tpu.get("returncode", 1)),
            tpu_backend=True,
            steps=tuple(steps),
        )

    if not nvidia_backend:
        # CPU or unknown accelerator: uv sync is enough. Verify CPU JAX only for
        # clearer logs, but do not force a CUDA stack.
        print("bootstrap starting step: verify_worker_jax_cpu", flush=True)
        verify_cpu = _probe_worker_jax_platform("cpu", name="verify_worker_jax_cpu")
        steps.append(verify_cpu)
        return UvEnvironmentSync(
            returncode=int(verify_cpu.get("returncode", 1)),
            tpu_backend=False,
            steps=tuple(steps),
        )

    print("bootstrap starting step: probe_existing_worker_jax_cuda", flush=True)
    existing_worker_cuda = _probe_worker_jax_platform(
        "cuda",
        name="probe_existing_worker_jax_cuda",
    )
    steps.append(existing_worker_cuda)
    _print_probe_summary(existing_worker_cuda)
    if int(existing_worker_cuda.get("returncode", 1)) == 0:
        _pin_current_process_for_jax_cuda()
        return UvEnvironmentSync(returncode=0, tpu_backend=False, steps=tuple(steps))

    if trust_base_jax:
        print(
            "bootstrap trust-base-jax: worker CUDA probe failed after slim sync; "
            "falling back to pinned JAX CUDA install",
            flush=True,
        )

    print("bootstrap starting step: uv_pip_install_pinned_jax_cuda", flush=True)
    install_stack = _install_pinned_jax_cuda_stack()
    install_step = _completed_step("uv_pip_install_pinned_jax_cuda", install_stack)
    install_step["jax_cuda_spec"] = _jax_cuda_package_spec()
    install_step["cuda_wheel_library_dirs"] = os.pathsep.join(_cuda_wheel_library_dirs())
    install_step["cuda_driver_library_dirs"] = os.pathsep.join(_cuda_driver_library_dirs())
    steps.append(install_step)
    if install_stack.returncode != 0:
        return _maybe_fallback_to_cpu_or_fail(steps=steps, tpu_backend=False)

    print("bootstrap starting step: uv_pip_purge_cuda13_packages", flush=True)
    purge_cuda13 = _purge_cuda13_packages()
    purge_step = _completed_step("uv_pip_purge_cuda13_packages", purge_cuda13)
    purge_step["optional"] = True
    # Do not fail the bootstrap only because a package was not installed.
    purge_step["returncode"] = 0
    steps.append(purge_step)

    print("bootstrap starting step: verify_pinned_worker_jax_cuda", flush=True)
    verify_cuda = _probe_worker_jax_platform(
        "cuda",
        name="verify_pinned_worker_jax_cuda",
    )
    verify_cuda["jax_cuda_spec"] = _jax_cuda_package_spec()
    steps.append(verify_cuda)
    _print_probe_summary(verify_cuda)
    if int(verify_cuda.get("returncode", 1)) == 0:
        _pin_current_process_for_jax_cuda()
        return UvEnvironmentSync(returncode=0, tpu_backend=False, steps=tuple(steps))

    return _maybe_fallback_to_cpu_or_fail(steps=steps, tpu_backend=False)


def _maybe_fallback_to_cpu_or_fail(
    *, steps: list[dict[str, object]], tpu_backend: bool
) -> UvEnvironmentSync:
    if not _allow_jax_cpu_fallback():
        return UvEnvironmentSync(returncode=1, tpu_backend=tpu_backend, steps=tuple(steps))

    print("bootstrap starting step: verify_worker_jax_cpu_fallback", flush=True)
    verify_cpu = _probe_worker_jax_platform(
        "cpu",
        name="verify_worker_jax_cpu_fallback",
    )
    steps.append(verify_cpu)
    if int(verify_cpu.get("returncode", 1)) != 0:
        return UvEnvironmentSync(returncode=1, tpu_backend=tpu_backend, steps=tuple(steps))

    os.environ["ORBIT_WARS_FORCE_JAX_CPU"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("JAX_PLATFORM_NAME", None)
    print("bootstrap warning: falling back to JAX CPU for this run", flush=True)
    return UvEnvironmentSync(returncode=0, tpu_backend=tpu_backend, steps=tuple(steps))


def _allow_jax_cpu_fallback() -> bool:
    value = os.environ.get("ORBIT_WARS_ALLOW_JAX_CPU_FALLBACK", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    run_type = os.environ.get("ORBIT_WARS_KAGGLE_RUN_TYPE", "").strip().lower()
    return run_type in {"smoke", "debug", "preflight", "cpu", "diagnostic"}


def _pin_current_process_for_jax_cuda() -> None:
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    _set_ld_library_path_for_current_process()


def _trust_base_jax() -> bool:
    """Return True when the worker should inherit Kaggle base-image JAX CUDA."""

    value = os.environ.get("ORBIT_WARS_KAGGLE_TRUST_BASE_JAX", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_worker_venv(*, system_site_packages: bool) -> dict[str, object]:
    """Create the worker venv before ``uv sync`` when it does not exist."""

    venv = _venv_path()
    if venv.exists():
        return {
            "name": "ensure_worker_venv",
            "returncode": 0,
            "stdout_tail": "venv already exists",
            "stderr_tail": "",
            "system_site_packages": system_site_packages,
        }
    command = ["uv", "venv"]
    if system_site_packages:
        command.append("--system-site-packages")
    command.append(str(venv))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_uv_bootstrap_env(),
    )
    step = _completed_step("ensure_worker_venv", completed)
    step["system_site_packages"] = system_site_packages
    return step


def _install_jax_tpu() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(_venv_python()),
            "-U",
            "jax[tpu]",
            "--find-links",
            LIBTPU_FIND_LINKS,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_uv_bootstrap_env(),
    )


def _install_pinned_jax_cuda_stack() -> subprocess.CompletedProcess[str]:
    """Install one pinned JAX CUDA package spec into the worker environment."""

    spec = _jax_cuda_package_spec()
    installer = os.environ.get("ORBIT_WARS_KAGGLE_JAX_INSTALLER", "uv").strip().lower()
    if installer in {"pip", "python-pip", "python"}:
        command = [
            str(_venv_python()),
            "-m",
            "pip",
            "install",
            "-U",
            "--force-reinstall",
            spec,
        ]
    else:
        command = [
            "uv",
            "pip",
            "install",
            "-p",
            str(_venv_python()),
            "-U",
            "--force-reinstall",
            spec,
        ]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_gpu_bootstrap_env(),
    )


def _jax_cuda_package_spec() -> str:
    explicit = os.environ.get("ORBIT_WARS_KAGGLE_JAX_CUDA_SPEC", "").strip()
    if explicit:
        return explicit
    version = os.environ.get(
        "ORBIT_WARS_KAGGLE_JAX_CUDA_VERSION",
        KAGGLE_DEFAULT_JAX_CUDA_VERSION,
    ).strip()
    extra = os.environ.get(
        "ORBIT_WARS_KAGGLE_JAX_CUDA_EXTRA",
        KAGGLE_DEFAULT_JAX_CUDA_EXTRA,
    ).strip()
    if version:
        return f"jax[{extra}]=={version}"
    return f"jax[{extra}]"


def _purge_cuda13_packages() -> subprocess.CompletedProcess[str]:
    """Remove CUDA 13 packages left behind by pyproject/lock resolution."""

    return subprocess.run(
        [
            "uv",
            "pip",
            "uninstall",
            "-p",
            str(_venv_python()),
            *KAGGLE_CUDA13_PACKAGES,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_gpu_bootstrap_env(),
    )


def _probe_base_python_jax_cuda() -> dict[str, object]:
    return _probe_jax_platform(
        sys.executable,
        "cuda",
        name="probe_base_python_jax_cuda",
        env=_system_jax_probe_env("cuda"),
    )


def _probe_worker_jax_platform(platform: str, *, name: str) -> dict[str, object]:
    return _probe_jax_platform(
        _venv_python(),
        platform,
        name=name,
        env=_worker_jax_probe_env(platform),
    )


def _probe_jax_platform(
    python_executable: str | Path,
    platform: str,
    *,
    name: str,
    env: dict[str, str],
) -> dict[str, object]:
    command = [str(python_executable), "-c", _JAX_PLATFORM_PROBE_CODE, platform]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(os.environ.get("ORBIT_WARS_JAX_PROBE_TIMEOUT_SECONDS", "90")),
    )
    result = _completed_step(name, completed)
    result["python"] = str(python_executable)
    result["platform"] = platform
    result["installed_jax_packages"] = _installed_packages_summary(prefixes=("jax",))
    result["installed_nvidia_packages"] = _installed_packages_summary(prefixes=("nvidia-",), limit=4000)
    result["cuda_wheel_library_dirs"] = os.pathsep.join(_cuda_wheel_library_dirs())
    result["cuda_driver_library_dirs"] = os.pathsep.join(_cuda_driver_library_dirs())
    result["plugin_dirs"] = _jax_plugin_dir_names()
    return result


def _print_probe_summary(step: dict[str, object]) -> None:
    stdout = str(step.get("stdout_tail", "")).strip()
    stderr = str(step.get("stderr_tail", "")).strip()
    if stdout:
        print(f"bootstrap {step.get('name')} stdout_tail:", stdout, flush=True)
    if stderr:
        print(f"bootstrap {step.get('name')} stderr_tail:", stderr, flush=True)


_JAX_PLATFORM_PROBE_CODE = textwrap.dedent(
    r'''
    from __future__ import annotations

    import ctypes
    import os
    import shutil
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    platform = sys.argv[1]
    if platform == "cuda":
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    elif platform == "tpu":
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ["JAX_PLATFORMS"] = "tpu,cpu"
    elif platform == "cpu":
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        raise SystemExit(f"unsupported probe platform: {platform}")

    print("probe_platform=" + platform)
    print("probe_python=" + sys.executable)
    print("JAX_PLATFORMS=" + os.environ.get("JAX_PLATFORMS", ""))
    print("LD_LIBRARY_PATH=" + os.environ.get("LD_LIBRARY_PATH", ""))

    if platform == "cuda":
        nvidia_smi = shutil.which("nvidia-smi")
        print("nvidia_smi=" + str(nvidia_smi))
        if nvidia_smi:
            smi = subprocess.run(
                [nvidia_smi, "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            print("nvidia_smi_returncode=" + str(smi.returncode))
            if smi.stdout.strip():
                print("nvidia_smi_stdout=" + smi.stdout.strip())
            if smi.stderr.strip():
                print("nvidia_smi_stderr=" + smi.stderr.strip())

        venv = Path(os.environ.get("ORBIT_WARS_WORKER_VENV", ".venv"))
        for lib_dir in sorted(venv.glob("lib/python*/site-packages/nvidia/*/lib")):
            print("nvidia_wheel_lib_dir=" + str(lib_dir))
        for candidate in (
            "/usr/local/nvidia/lib64",
            "/usr/local/nvidia/lib",
            "/usr/local/cuda/compat",
            "/usr/lib/x86_64-linux-gnu",
        ):
            path = Path(candidate)
            if path.exists() and (
                list(path.glob("libcuda.so*")) or list(path.glob("libnvidia-ml.so*"))
            ):
                print("nvidia_driver_lib_dir=" + str(path))

        for lib in (
            "libcuda.so.1",
            "libnvidia-ml.so.1",
            "libcusparse.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcusolver.so.11",
        ):
            try:
                ctypes.CDLL(lib)
                print("loaded=" + lib)
            except OSError as exc:
                print("failed_to_load=" + lib + ": " + str(exc))

    try:
        import jax
        import jax.numpy as jnp

        print("jax=" + jax.__version__)
        if os.environ.get("ORBIT_WARS_JAX_PROBE_IMPORT_FLAX") == "1":
            import flax
            import flax.linen as nn  # noqa: F401

            print("flax=" + getattr(flax, "__version__", "unknown"))
            print("flax_linen_import=ok")
        print("default_backend=" + jax.default_backend())
        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"no {platform} devices returned by jax.devices")
        print("devices=" + ",".join(str(device) for device in devices))
        x = jnp.arange(8, dtype=jnp.float32)
        y = jnp.sum(x).block_until_ready()
        print("probe_sum=" + str(float(y)))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    '''
).strip()


def _system_jax_probe_env(platform: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(KAGGLE_WORKER_VENV_ENV, str(_venv_path().resolve()))
    env.pop("JAX_PLATFORM_NAME", None)
    env["JAX_PLATFORMS"] = "cuda,cpu" if platform == "cuda" else platform
    return env


def _worker_jax_probe_env(platform: str) -> dict[str, str]:
    if platform == "cuda":
        env = _gpu_bootstrap_env()
    else:
        env = _uv_bootstrap_env()
        env.pop("JAX_PLATFORM_NAME", None)
        if platform == "tpu":
            env["JAX_PLATFORMS"] = "tpu,cpu"
        elif platform == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
        else:
            env["JAX_PLATFORMS"] = platform
    # The training path imports flax.linen immediately, so worker probes must
    # validate JAX/Flax compatibility, not only CUDA device visibility.
    env["ORBIT_WARS_JAX_PROBE_IMPORT_FLAX"] = "1"
    return env


def _uv_bootstrap_env() -> dict[str, str]:
    env = os.environ.copy()
    venv = _venv_path().resolve()
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    env["VIRTUAL_ENV"] = str(venv)
    env[KAGGLE_WORKER_VENV_ENV] = str(venv)
    # Package installation should not inherit stale JAX backend pins.
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("JAX_PLATFORMS", None)
    return env


def _gpu_bootstrap_env() -> dict[str, str]:
    env = _uv_bootstrap_env()
    env.pop("JAX_PLATFORM_NAME", None)
    env["JAX_PLATFORMS"] = "cuda,cpu"
    wheel_dirs = _cuda_wheel_library_dirs()
    driver_dirs = _cuda_driver_library_dirs()
    merged = [*wheel_dirs, *driver_dirs]
    if merged:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(merged))
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def _set_ld_library_path_for_current_process() -> None:
    merged = [*_cuda_wheel_library_dirs(), *_cuda_driver_library_dirs()]
    if merged:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(merged))


def _add_cuda_wheel_library_path(env: dict[str, str], *, venv: Path) -> None:
    """Prepend CUDA wheel library dirs from *venv* into ``env`` LD_LIBRARY_PATH."""

    wheel_dirs: list[str] = []
    for site_packages in sorted(venv.glob("lib/python*/site-packages")):
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                wheel_dirs.append(str(lib_dir.resolve()))
    existing = env.get("LD_LIBRARY_PATH", "")
    merged = [*wheel_dirs]
    if existing:
        merged.append(existing)
    if merged:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(merged))


def _cuda_wheel_library_dirs() -> list[str]:
    dirs: list[str] = []
    venv = _venv_path().resolve()
    for site_packages in sorted(venv.glob("lib/python*/site-packages")):
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                dirs.append(str(lib_dir.resolve()))
    return _dedupe(dirs)


def _cuda_driver_library_dirs() -> list[str]:
    candidates: list[str] = [*KAGGLE_CUDA_DRIVER_LIBRARY_DIR_CANDIDATES]
    candidates.extend(
        item
        for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if item
    )
    result: list[str] = []
    for item in candidates:
        path = Path(item)
        if not path.exists() or not path.is_dir():
            continue
        if list(path.glob("libcuda.so*")) or list(path.glob("libnvidia-ml.so*")):
            result.append(str(path.resolve()))
    return _dedupe(result)


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _installed_packages_summary(
    *, prefixes: tuple[str, ...], limit: int = 2000
) -> str:
    if not _venv_python().exists():
        return ""
    pip_list = subprocess.run(
        ["uv", "pip", "list", "-p", str(_venv_python())],
        check=False,
        capture_output=True,
        text=True,
        env=_uv_bootstrap_env(),
    )
    if pip_list.returncode != 0:
        return _tail(pip_list.stderr or pip_list.stdout, limit=limit)
    lower_prefixes = tuple(prefix.lower() for prefix in prefixes)
    return _tail(
        "\n".join(
            line
            for line in pip_list.stdout.splitlines()
            if line.lower().startswith(lower_prefixes)
        ),
        limit=limit,
    )


def _jax_plugin_dir_names(*, venv: Path | None = None) -> str:
    venv = venv or _venv_path()
    names: list[str] = []
    for root in sorted(venv.glob("lib/python*/site-packages/jax_plugins")):
        if not root.exists():
            continue
        names.extend(item.name for item in root.iterdir() if item.is_dir())
    return ",".join(sorted(names))


def _venv_path() -> Path:
    return Path(os.environ.get(KAGGLE_WORKER_VENV_ENV, ".venv"))


def _venv_python() -> Path:
    return _venv_path() / "bin" / "python"


def _uninstall_cuda_jax_plugins() -> subprocess.CompletedProcess[str]:
    """Remove CUDA JAX plugins so TPU libtpu can register cleanly."""

    return subprocess.run(
        [
            "uv",
            "pip",
            "uninstall",
            "-p",
            str(_venv_python()),
            *KAGGLE_CUDA_JAX_PLUGINS,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_uv_bootstrap_env(),
    )


def _optional_probe_step(step: dict[str, object]) -> dict[str, object]:
    result = dict(step)
    result["optional"] = True
    # Keep diagnostics without causing first_failed_bootstrap_step() to report
    # the optional base-environment probe as the bootstrap failure.
    result["observed_returncode"] = result.get("returncode")
    result["returncode"] = 0
    return result


def _completed_step(
    name: str, completed: subprocess.CompletedProcess[str]
) -> dict[str, object]:
    return {
        "name": name,
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _tail(text: str | None, *, limit: int = 2000) -> str:
    if not text:
        return ""
    return text.strip()[-limit:]
