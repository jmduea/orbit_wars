from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from src.orchestration.accelerators import (
    KAGGLE_TPU_V5E8,
    default_memory_gb,
    is_tpu_accelerator,
)
from src.orchestration.kaggle_jax import (
    _add_cuda_wheel_library_path,
    first_failed_bootstrap_step,
    format_bootstrap_failure,
    jax_platform_for_accelerator,
    log_bootstrap_failure,
    sync_kaggle_worker_environment,
)

_WORKER_MODULE = None


def _worker_module():
    global _WORKER_MODULE
    if _WORKER_MODULE is None:
        worker_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "kaggle_worker_entry.py"
        )
        spec = importlib.util.spec_from_file_location(
            "kaggle_worker_entry_test", worker_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _WORKER_MODULE = module
    return _WORKER_MODULE


def test_sync_kaggle_worker_environment_installs_tpu_jax(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setenv("ORBIT_WARS_WORKER_VENV", str(tmp_path / "worker_venv"))

    def fake_run(command, **kwargs):
        commands.append(list(command))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("src.orchestration.kaggle_jax.subprocess.run", fake_run)

    sync = sync_kaggle_worker_environment(KAGGLE_TPU_V5E8)

    assert sync.returncode == 0
    assert sync.tpu_backend is True
    step_names = [step["name"] for step in sync.steps]
    assert "ensure_worker_venv" in step_names
    assert "uv_sync" in step_names
    assert any(cmd[:3] == ["uv", "sync", "--no-dev"] for cmd in commands)
    assert any(cmd[:2] == ["uv", "venv"] for cmd in commands)
    assert any(cmd[:3] == ["uv", "pip", "uninstall"] for cmd in commands)
    assert any("jax[tpu]" in cmd for cmd in commands)


def test_sync_kaggle_worker_environment_installs_cuda_jax_on_gpu(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setenv("ORBIT_WARS_WORKER_VENV", str(tmp_path / "worker_venv"))

    def fake_run(command, **kwargs):
        commands.append(list(command))

        class Result:
            returncode = 0
            stdout = (
                "Version: 0.10.0\n"
                if list(command)[:3] == ["uv", "pip", "show"]
                and list(command)[-1] == "jax"
                else ""
            )
            stderr = ""

        return Result()

    monkeypatch.setattr("src.orchestration.kaggle_jax.subprocess.run", fake_run)

    sync = sync_kaggle_worker_environment("NvidiaH100")

    assert sync.returncode == 0
    assert sync.tpu_backend is False
    step_names = [step["name"] for step in sync.steps]
    assert "probe_base_python_jax_cuda" in step_names
    assert "ensure_worker_venv" in step_names
    assert "uv_sync" in step_names
    assert any(cmd[:3] == ["uv", "sync", "--no-dev"] for cmd in commands)
    assert any(cmd[:2] == ["uv", "venv"] for cmd in commands)
    assert "--system-site-packages" in next(
        cmd for cmd in commands if cmd[:2] == ["uv", "venv"]
    )
    assert step_names[-1] in {
        "probe_existing_worker_jax_cuda",
        "verify_pinned_worker_jax_cuda",
    }
    assert not any("jax[cuda12]" in " ".join(cmd) for cmd in commands)


def test_cuda_wheel_library_path_includes_nvidia_lib_dirs(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    cusparse = venv / "lib/python3.12/site-packages/nvidia/cusparse/lib"
    cublas = venv / "lib/python3.12/site-packages/nvidia/cublas/lib"
    cusparse.mkdir(parents=True)
    cublas.mkdir(parents=True)
    env = {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}

    _add_cuda_wheel_library_path(env, venv=venv)

    entries = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(cublas.resolve()) in entries
    assert str(cusparse.resolve()) in entries
    assert entries[-1] == "/usr/local/cuda/lib64"


def test_log_bootstrap_failure_prints_installed_nvidia_packages(capsys) -> None:
    log_bootstrap_failure(
        [
            {
                "name": "verify_gpu_jax_plugins",
                "returncode": 1,
                "stderr_tail": "Unable to load cuSPARSE",
                "stdout_tail": "",
                "installed_nvidia_packages": "nvidia-cusparse-cu12 12.5.10.65",
            }
        ]
    )

    captured = capsys.readouterr().out
    assert "Unable to load cuSPARSE" in captured
    assert "bootstrap installed_nvidia_packages:" in captured
    assert "nvidia-cusparse-cu12 12.5.10.65" in captured


def test_jax_platform_for_accelerator() -> None:
    assert jax_platform_for_accelerator(KAGGLE_TPU_V5E8) == "tpu"
    assert jax_platform_for_accelerator("NvidiaH100") == "cuda"
    assert jax_platform_for_accelerator("") is None


def test_default_memory_gb_for_tpu_v5e8() -> None:
    assert default_memory_gb(KAGGLE_TPU_V5E8) == 384.0
    assert is_tpu_accelerator(KAGGLE_TPU_V5E8)


def test_default_memory_gb_for_single_gpu_accelerators() -> None:
    assert default_memory_gb("NvidiaH100") == 80.0
    assert default_memory_gb("NvidiaTeslaA100") == 40.0
    assert default_memory_gb("NvidiaTeslaT4") == 16.0


def test_format_bootstrap_failure_names_failed_step() -> None:
    steps = [
        {"name": "uv_sync", "returncode": 0, "stderr_tail": "", "stdout_tail": ""},
        {
            "name": "verify_gpu_jax_plugins",
            "returncode": 1,
            "stderr_tail": "xla_cuda13 plugin still present",
            "stdout_tail": "",
        },
    ]

    assert first_failed_bootstrap_step(steps) == steps[1]
    message = format_bootstrap_failure(steps)
    assert "verify_gpu_jax_plugins" in message
    assert "xla_cuda13 plugin still present" in message


def test_log_bootstrap_failure_prints_step_summary(capsys) -> None:
    log_bootstrap_failure(
        [
            {
                "name": "uv_sync",
                "returncode": 1,
                "stderr_tail": "network error",
                "stdout_tail": "",
            }
        ]
    )

    captured = capsys.readouterr().out
    assert "bootstrap steps:" in captured
    assert "uv_sync: returncode=1" in captured
    assert "bootstrap failed at step: uv_sync" in captured
    assert "network error" in captured


def test_kaggle_worker_bootstrap_uv_sync_avoids_src_import(
    monkeypatch, tmp_path: Path
) -> None:
    worker = _worker_module()
    commands: list[list[str]] = []
    monkeypatch.setenv("ORBIT_WARS_WORKER_VENV", str(tmp_path / "worker_venv"))

    def fake_run(command, **kwargs):
        commands.append(list(command))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("src.orchestration.kaggle_jax.subprocess.run", fake_run)

    result = worker._bootstrap_uv_environment("TpuV6E8")

    assert result["returncode"] == 0
    assert result["tpu_backend"] is True
    assert any(cmd[:3] == ["uv", "sync", "--no-dev"] for cmd in commands)
    assert any(cmd[:2] == ["uv", "venv"] for cmd in commands)
    assert any(cmd[:3] == ["uv", "pip", "uninstall"] for cmd in commands)
    assert any("jax[tpu]" in cmd for cmd in commands)


def test_kaggle_worker_bootstrap_failure_logs_failed_step(monkeypatch, capsys) -> None:
    worker = _worker_module()

    def fake_sync(accelerator_id: str):
        from src.orchestration.kaggle_jax import UvEnvironmentSync

        return UvEnvironmentSync(
            returncode=1,
            tpu_backend=False,
            steps=(
                {
                    "name": "verify_gpu_jax_plugins",
                    "returncode": 1,
                    "stderr_tail": "xla_cuda13 plugin still present",
                    "stdout_tail": "",
                },
            ),
        )

    monkeypatch.setattr(worker, "sync_kaggle_worker_environment", fake_sync)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        worker.Path, "exists", lambda self: self.name == "pyproject.toml"
    )
    monkeypatch.setattr(worker, "_reset_worker_venv", lambda: None)
    monkeypatch.setattr(worker, "_activate_local_venv", lambda: None)
    monkeypatch.setattr(worker, "_reexec_with_venv_python", lambda: None)
    monkeypatch.setattr(worker, "write_summary", lambda summary: None)

    with pytest.raises(SystemExit, match="verify_gpu_jax_plugins"):
        worker._ensure_uv_environment({"status": "starting"})

    captured = capsys.readouterr().out
    assert "bootstrap failed at step: verify_gpu_jax_plugins" in captured


def test_kaggle_worker_loads_wandb_key_from_kaggle_secret(monkeypatch) -> None:
    worker = _worker_module()

    class FakeSecretsClient:
        def get_secret(self, name: str) -> str:
            assert name == "CUSTOM_WANDB_SECRET"
            return "resolved-secret"

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY_SECRET_NAME", "CUSTOM_WANDB_SECRET")
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        types.SimpleNamespace(UserSecretsClient=FakeSecretsClient),
    )

    result = worker._load_wandb_api_key_from_kaggle_secret()

    assert result["loaded"] is True
    assert result["source"] == "kaggle_secret"
    assert os.environ["WANDB_API_KEY"] == "resolved-secret"


def test_kaggle_worker_smoke_mode_applies_short_training_overrides() -> None:
    worker = _worker_module()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("ORBIT_WARS_KAGGLE_RUN_TYPE", "smoke")
        overrides = worker._apply_smoke_training_overrides(
            ["training.total_updates=500", "training.num_envs=8"]
        )
        settings = worker._calibration_settings()
        assert overrides == ["training.num_envs=8", "training.total_updates=10"]
        assert settings["max_variants"] == 1
        assert settings["timeout_seconds"] == 600
    finally:
        monkeypatch.undo()


def test_kaggle_worker_standalone_mode_skips_wandb_secret(monkeypatch) -> None:
    worker = _worker_module()
    called = {"count": 0}

    def fake_load_secret() -> dict[str, object]:
        called["count"] += 1
        return {"loaded": True, "source": "should-not-run"}

    monkeypatch.setenv("ORBIT_WARS_KAGGLE_WORKER_MODE", "standalone")
    monkeypatch.setattr(worker, "_load_wandb_api_key_from_kaggle_secret", fake_load_secret)
    monkeypatch.setattr(worker, "load_packaged_env", lambda: None)
    monkeypatch.setattr(worker, "write_summary", lambda summary: None)
    monkeypatch.setattr(
        worker,
        "_ensure_uv_environment",
        lambda summary: summary.update({"uv": "skipped"}),
    )
    monkeypatch.setattr(worker, "diagnostics", lambda: {"jax_platforms": ["cuda", "gpu"]})
    monkeypatch.setattr(worker, "_ensure_accelerator", lambda summary: None)
    monkeypatch.setattr(
        worker,
        "_run_standalone_worker",
        lambda summary: summary.update({"status": "standalone_complete", "exit_code": 0}),
    )

    worker.main()

    assert called["count"] == 0


def test_kaggle_worker_is_standalone_mode() -> None:
    worker = _worker_module()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.delenv("ORBIT_WARS_KAGGLE_WORKER_MODE", raising=False)
        assert worker._is_standalone_mode() is False
        monkeypatch.setenv("ORBIT_WARS_KAGGLE_WORKER_MODE", "standalone")
        assert worker._is_standalone_mode() is True
    finally:
        monkeypatch.undo()


def test_kaggle_worker_config_to_overrides_includes_hydra_groups() -> None:
    worker = _worker_module()
    overrides = worker._config_to_overrides(
        {
            "model": "transformer_factorized",
            "training": "2p4p_16_rotate",
            "curriculum": "self_play_staged",
            "training.lr": 0.0003,
            "kaggle_worker.selected_overrides": ["ignored=true"],
        }
    )

    assert "model=transformer_factorized" in overrides
    assert "training=2p4p_16_rotate" in overrides
    assert "curriculum=self_play_staged" in overrides
    assert "training.lr=0.0003" in overrides
    assert all(not item.startswith("kaggle_worker.") for item in overrides)
