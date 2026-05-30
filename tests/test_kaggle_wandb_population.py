from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts import kaggle_wandb_population as launcher
from src.orchestration.accelerators import KAGGLE_TPU_V5E8, is_tpu_accelerator
from src.orchestration.kaggle_cli import KaggleCli, KaggleKernelRef
from src.orchestration.kernel_package import render_kernel_package
from src.orchestration.population import (
    AcceleratorPreference,
    ShortlistRow,
    rank_shortlist,
    render_hydra_command,
)
from src.orchestration.throughput import (
    HardwareProfile,
    calibration_grid,
    estimate_training_overrides,
    finalize_rollout_shape_overrides,
    largest_compatible_microbatch,
    rollout_group_env_counts,
)
from src.orchestration.wandb_sweeps import add_population_metadata


def test_accelerator_preference_ordered_fallback() -> None:
    pref = AcceleratorPreference(("A", "B", "C"))

    assert pref.first_available(["A"]) == "B"
    assert pref.candidates_after(["A", "C"]) == ("B",)
    assert pref.first_available(["A", "B", "C"]) is None


def test_default_accelerator_preference_prefers_single_gpu_vram() -> None:
    pref = AcceleratorPreference()

    assert pref.accelerator_ids[0] == "NvidiaH100"
    assert "Tpu" not in pref.accelerator_ids[0]
    assert all(not item.lower().startswith("tpu") for item in pref.accelerator_ids)


def test_is_tpu_accelerator_recognizes_kaggle_ids() -> None:
    assert is_tpu_accelerator(KAGGLE_TPU_V5E8)
    assert is_tpu_accelerator("TpuV3-8")
    assert not is_tpu_accelerator("NvidiaTeslaT4")


def test_render_hydra_command_keeps_train_entrypoint() -> None:
    command = render_hydra_command(("training.total_updates=5",))

    assert command == [
        "uv",
        "run",
        "python",
        "-m",
        "src.train",
        "training.total_updates=5",
    ]


def test_kaggle_cli_push_renders_accelerator_and_timeout(tmp_path: Path) -> None:
    calls = []

    def fake_runner(command, *, cwd=None):
        calls.append((list(command), cwd))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    cli = KaggleCli(runner=fake_runner)
    cli.push(tmp_path, accelerator="NvidiaTeslaT4", timeout_seconds=123)

    assert calls == [
        (
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(tmp_path),
                "--accelerator",
                "NvidiaTeslaT4",
                "--timeout",
                "123",
            ],
            tmp_path,
        )
    ]


def test_kaggle_status_normalizes_running() -> None:
    ref = KaggleKernelRef.parse("owner/slug")
    cli = KaggleCli(
        runner=lambda command, *, cwd=None: subprocess.CompletedProcess(
            command, 0, stdout="Kernel is running", stderr=""
        )
    )

    status = cli.status(ref)

    assert status.normalized == "running"


def test_estimate_training_overrides_scales_down_heavier_models() -> None:
    light = estimate_training_overrides(
        HardwareProfile("gpu", "test", 24),
        {"hidden_size": 128, "planet_transformer_layers": 1},
        {"feature_history_steps": 2, "trajectory_shield_horizon": 10},
    )
    heavy = estimate_training_overrides(
        HardwareProfile("gpu", "test", 24),
        {"hidden_size": 224, "planet_transformer_layers": 3},
        {"feature_history_steps": 10, "trajectory_shield_horizon": 30},
    )

    light_envs = _override_int(light, "training.num_envs")
    heavy_envs = _override_int(heavy, "training.num_envs")
    assert heavy_envs < light_envs
    assert len(calibration_grid(heavy)) == 3


def test_mixed_format_overrides_keep_microbatch_compatible_with_group_envs() -> None:
    hydra_overrides = ("format=2p_4p_16env",)
    overrides = estimate_training_overrides(
        HardwareProfile("gpu", "test", 24),
        {"hidden_size": 140, "planet_transformer_layers": 1},
        {"feature_history_steps": 2, "trajectory_shield_horizon": 20},
        hydra_overrides=hydra_overrides,
    )

    assert not any(item.startswith("training.num_envs=") for item in overrides)
    micro = _override_int(overrides, "training.rollout_microbatch_envs")
    assert all(
        count % micro == 0 for count in rollout_group_env_counts(hydra_overrides)
    )
    for variant in calibration_grid(overrides, hydra_overrides=hydra_overrides):
        variant_micro = _override_int(variant, "training.rollout_microbatch_envs")
        assert all(
            count % variant_micro == 0
            for count in rollout_group_env_counts(hydra_overrides)
        )


def test_finalize_rollout_shape_overrides_repairs_bad_microbatch() -> None:
    hydra_overrides = ("format=2p_4p_16env",)
    repaired = finalize_rollout_shape_overrides(
        (
            "training.num_envs=24",
            "training.rollout_microbatch_envs=12",
            "training.rollout_steps=128",
        ),
        hydra_overrides,
    )

    assert not any(item.startswith("training.num_envs=") for item in repaired)
    micro = _override_int(repaired, "training.rollout_microbatch_envs")
    assert micro == largest_compatible_microbatch(
        12, rollout_group_env_counts(hydra_overrides)
    )


def test_shortlist_ranks_finished_checkpointed_runs_first() -> None:
    rows = [
        ShortlistRow(
            run_id="bad",
            name="bad",
            state="running",
            checkpoint_artifact=None,
            metrics={"episode_reward_mean": 100.0},
        ),
        ShortlistRow(
            run_id="good",
            name="good",
            state="finished",
            checkpoint_artifact="checkpoint:v1",
            metrics={"episode_reward_mean": 1.0, "samples_per_sec": 1000.0},
        ),
    ]

    assert rank_shortlist(rows, limit=1)[0].run_id == "good"


def test_add_population_metadata_preserves_existing_tags() -> None:
    sweep = {"parameters": {"telemetry.wandb.tags": {"value": ["base"]}}}

    result = add_population_metadata(sweep, group="group", tags=("kaggle", "base"))

    assert result["parameters"]["telemetry.wandb.group"]["value"] == "group"
    assert result["parameters"]["telemetry.wandb.tags"]["value"] == ["base", "kaggle"]


def test_render_kernel_package_writes_metadata_and_env(tmp_path: Path) -> None:
    worker_source = tmp_path / "worker.py"
    worker_source.write_text("print('worker')\n", encoding="utf-8")

    package = render_kernel_package(
        package_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        worker_source=worker_source,
        env={"WANDB_SWEEP_ID": "abc"},
    )

    metadata = package.metadata_path.read_text(encoding="utf-8")
    env = (package.package_dir / "worker-env.json").read_text(encoding="utf-8")
    assert '"id": "owner/kernel"' in metadata
    assert '"WANDB_SWEEP_ID": "abc"' in env


def test_prepare_uses_kaggle_secret_name_instead_of_raw_wandb_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "raw-local-key")
    args = argparse.Namespace(
        work_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        sweep_yaml=Path("conf/sweeps/wandb/kaggle_population_long.yaml"),
        project="orbit_wars",
        entity="entity",
    )

    package = launcher._prepare(args, sweep_id="sweep", accelerator="NvidiaH100")
    env_text = (package.package_dir / "worker-env.json").read_text(encoding="utf-8")
    env = json.loads(env_text)

    assert env["WANDB_API_KEY_SECRET_NAME"] == "WANDB_API_KEY"
    assert env["ORBIT_WARS_WORKER_VENV"] == "/tmp/orbit_wars_worker_venv"
    assert "WANDB_API_KEY" not in env
    assert "raw-local-key" not in env_text


def test_render_kernel_package_rewrites_gpu_jax_dependency(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "conf").mkdir()
    (repo / "scripts").mkdir()
    (repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    for script in (
        "kaggle_worker_entry.py",
        "benchmark_jax_rl.py",
        "kaggle_runtime_env.py",
    ):
        (repo / "scripts" / script).write_text("print('ok')\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    (repo / "uv.lock").write_text("stale cuda13 lock\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'dependencies = [',
                '  "jax[cuda13]; sys_platform == \'linux\' and platform_machine == \'x86_64\'",',
                '  "jax; sys_platform != \'linux\' or platform_machine != \'x86_64\'",',
                "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    package = render_kernel_package(
        package_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        worker_source=repo / "scripts" / "kaggle_worker_entry.py",
        env={"WANDB_SWEEP_ID": "abc"},
        repo_root=repo,
        accelerator="NvidiaH100",
    )

    pyproject = (package.package_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert "jax[cuda13]" not in pyproject
    assert (
        '"jax; sys_platform == \'linux\' and platform_machine == \'x86_64\'"'
        in pyproject
    )
    assert not (package.package_dir / "uv.lock").exists()


def test_render_kernel_package_marks_tpu_metadata(tmp_path: Path) -> None:
    worker_source = tmp_path / "worker.py"
    worker_source.write_text("print('worker')\n", encoding="utf-8")

    package = render_kernel_package(
        package_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        worker_source=worker_source,
        env={"WANDB_SWEEP_ID": "abc"},
        accelerator=KAGGLE_TPU_V5E8,
    )

    metadata = package.metadata_path.read_text(encoding="utf-8")
    assert '"enable_gpu": false' in metadata
    assert '"enable_tpu": true' in metadata


def _override_int(overrides: tuple[str, ...], key: str) -> int:
    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            return int(item.removeprefix(prefix))
    raise AssertionError(f"missing override {key}")
