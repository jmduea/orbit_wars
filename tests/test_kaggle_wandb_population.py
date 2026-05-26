from __future__ import annotations

import subprocess
from pathlib import Path

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
)
from src.orchestration.wandb_sweeps import add_population_metadata


def test_accelerator_preference_ordered_fallback() -> None:
    pref = AcceleratorPreference(("A", "B", "C"))

    assert pref.first_available(["A"]) == "B"
    assert pref.candidates_after(["A", "C"]) == ("B",)
    assert pref.first_available(["A", "B", "C"]) is None


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


def _override_int(overrides: tuple[str, ...], key: str) -> int:
    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            return int(item.removeprefix(prefix))
    raise AssertionError(f"missing override {key}")
