from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest
from hydra.errors import MissingConfigException

from src.orchestration import kaggle_runner as launcher
from src.orchestration.accelerators import KAGGLE_TPU_V5E8, is_tpu_accelerator
from src.orchestration.kaggle_cli import (
    KaggleCli,
    KaggleKernelRef,
    parse_kernel_ref_from_text,
    resolve_kaggle_username,
)
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
from src.orchestration.wandb_sweeps import (
    _checkpoint_sort_key,
    add_population_metadata,
    resolve_standalone_parameters,
    resolve_wandb_group_from_sweep,
)


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
    hydra_overrides = ("training=2p4p_32_split",)
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
        assert not any(item.startswith("training.num_envs=") for item in variant)


def test_rotate_training_profile_calibration_grid_uses_resolved_env_budget() -> None:
    hydra_overrides = ("training=2p4p_16_rotate",)
    overrides = estimate_training_overrides(
        HardwareProfile("gpu", "test", 16),
        {"hidden_size": 140, "planet_transformer_layers": 1},
        {"feature_history_steps": 2, "trajectory_shield_horizon": 20},
        hydra_overrides=hydra_overrides,
    )

    assert not any(item.startswith("training.num_envs=") for item in overrides)
    variants = calibration_grid(overrides, hydra_overrides=hydra_overrides)
    assert variants
    for variant in variants:
        assert not any(item.startswith("training.num_envs=") for item in variant)
        variant_micro = _override_int(variant, "training.rollout_microbatch_envs")
        assert all(
            count % variant_micro == 0
            for count in rollout_group_env_counts(hydra_overrides)
        )


def test_finalize_rollout_shape_overrides_repairs_bad_microbatch() -> None:
    hydra_overrides = ("training=2p4p_32_split",)
    repaired = finalize_rollout_shape_overrides(
        (
            "training.num_envs=24",
            "training.rollout_microbatch_envs=12",
            "training.rollout_steps=128",
        ),
        hydra_overrides,
    )

    micro = _override_int(repaired, "training.rollout_microbatch_envs")
    assert micro == largest_compatible_microbatch(
        12, rollout_group_env_counts(hydra_overrides)
    )
    assert _override_int(repaired, "training.num_envs") == 24


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


def test_resolve_wandb_group_from_sweep_prefers_fixed_group() -> None:
    sweep = {
        "parameters": {
            "telemetry.wandb.group": {"value": "sweep_group"},
            "output.campaign": {"value": "campaign_slug"},
        }
    }

    assert resolve_wandb_group_from_sweep(sweep) == "sweep_group"


def test_resolve_wandb_group_from_sweep_falls_back_to_yaml_stem(tmp_path: Path) -> None:
    sweep = {"parameters": {"output.campaign": {"value": "campaign_slug"}}}
    sweep_path = tmp_path / "throughput_2p.yaml"

    assert resolve_wandb_group_from_sweep(sweep, sweep_yaml_path=sweep_path) == (
        "campaign_slug"
    )


def test_checkpoint_sort_key_prefers_best_over_latest() -> None:
    best = _checkpoint_sort_key(
        {"aliases": ("best", "promoted"), "update": 10, "version": "v3"}
    )
    latest = _checkpoint_sort_key(
        {"aliases": ("latest",), "update": 99, "version": "v9"}
    )

    assert best > latest


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
        sweep_yaml=Path("conf/sweeps/wandb/kaggle_runner_long.yaml"),
        project="orbit_wars",
        entity="entity",
        run_type=None,
        no_wandb=False,
        standalone_overrides=[],
    )

    package = launcher.prepare(args, sweep_id="sweep", accelerator="NvidiaH100")
    env_text = (package.package_dir / "worker-env.json").read_text(encoding="utf-8")
    env = json.loads(env_text)

    assert env["WANDB_API_KEY_SECRET_NAME"] == "WANDB_API_KEY"
    assert env["ORBIT_WARS_WORKER_VENV"] == "/tmp/orbit_wars_worker_venv"
    assert env["ORBIT_WARS_KAGGLE_TRUST_BASE_JAX"] == "1"
    assert "WANDB_API_KEY" not in env
    assert "raw-local-key" not in env_text


def test_prepare_benchmark_run_type_sets_calibration_grid(tmp_path: Path) -> None:
    args = argparse.Namespace(
        work_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        sweep_yaml=Path("conf/sweeps/wandb/kaggle_runner_mvp.yaml"),
        project="orbit_wars",
        entity="entity",
        run_type="benchmark",
        no_wandb=True,
        standalone_overrides=[],
        calibration_max_variants=None,
        calibration_warmup=None,
        calibration_updates=None,
        calibration_timeout_seconds=None,
    )

    package = launcher.prepare(args, sweep_id=None, accelerator="NvidiaTeslaP100")
    env = json.loads((package.package_dir / "worker-env.json").read_text(encoding="utf-8"))

    assert env["ORBIT_WARS_KAGGLE_RUN_TYPE"] == "benchmark"
    assert env["ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS"] == "3"
    assert env["ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP"] == "2"
    assert env["ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES"] == "30"
    assert env["ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS"] == "3600"


def test_prepare_standalone_without_run_type_omits_calibration_caps(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        work_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        sweep_yaml=Path("conf/sweeps/wandb/kaggle_runner_mvp.yaml"),
        project="orbit_wars",
        entity="entity",
        run_type=None,
        no_wandb=True,
        standalone_overrides=[],
        calibration_max_variants=None,
        calibration_warmup=None,
        calibration_updates=None,
        calibration_timeout_seconds=None,
    )

    package = launcher.prepare(args, sweep_id=None, accelerator="NvidiaTeslaP100")
    env = json.loads((package.package_dir / "worker-env.json").read_text(encoding="utf-8"))

    assert env["ORBIT_WARS_KAGGLE_WORKER_MODE"] == "standalone"
    assert "ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS" not in env


def test_prepare_smoke_run_type_sets_short_defaults(tmp_path: Path) -> None:
    args = argparse.Namespace(
        work_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        sweep_yaml=Path("conf/sweeps/wandb/kaggle_runner_mvp.yaml"),
        project="orbit_wars",
        entity="entity",
        run_type="smoke",
        no_wandb=False,
        standalone_overrides=[],
        calibration_max_variants=None,
        calibration_warmup=None,
        calibration_updates=None,
        calibration_timeout_seconds=None,
    )

    package = launcher.prepare(args, sweep_id="sweep", accelerator="NvidiaH100")
    env = json.loads((package.package_dir / "worker-env.json").read_text(encoding="utf-8"))

    assert env["ORBIT_WARS_KAGGLE_RUN_TYPE"] == "smoke"
    assert env["ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS"] == "1"
    assert env["ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK"] == "1"


def test_prepare_standalone_mode_omits_wandb_secrets(tmp_path: Path) -> None:
    args = argparse.Namespace(
        work_dir=tmp_path / "pkg",
        kernel_id="owner/kernel",
        title="Worker",
        sweep_yaml=Path("conf/sweeps/wandb/kaggle_runner_mvp.yaml"),
        project="orbit_wars",
        entity="entity",
        run_type="smoke",
        no_wandb=True,
        standalone_overrides=["training.total_updates=12"],
    )

    package = launcher.prepare(args, sweep_id=None, accelerator="NvidiaTeslaP100")
    env_text = (package.package_dir / "worker-env.json").read_text(encoding="utf-8")
    env = json.loads(env_text)

    assert env["ORBIT_WARS_KAGGLE_WORKER_MODE"] == "standalone"
    assert env["WANDB_SWEEP_YAML"] == "conf/sweeps/wandb/kaggle_runner_mvp.yaml"
    assert "WANDB_SWEEP_ID" not in env
    assert "WANDB_API_KEY_SECRET_NAME" not in env
    assert "WANDB_PROJECT" not in env
    assert json.loads(env["ORBIT_WARS_KAGGLE_STANDALONE_OVERRIDES"]) == [
        "training.total_updates=12"
    ]
    assert "WANDB_API_KEY" not in env_text


def test_resolve_standalone_parameters_picks_first_values() -> None:
    parameters = {
        "training.gamma": {"values": [0.99, 0.999]},
        "training.lr": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.0006,
        },
        "model": {"value": "transformer_factorized"},
    }

    resolved = resolve_standalone_parameters(parameters)

    assert resolved["training.gamma"] == 0.99
    assert resolved["training.lr"] == 0.0001
    assert resolved["model"] == "transformer_factorized"


def test_default_sweep_points_at_conf_mvp_yaml() -> None:
    assert launcher.DEFAULT_SWEEP.name == "kaggle_runner_mvp.yaml"
    assert launcher.DEFAULT_SWEEP.parts[-3:] == ("sweeps", "wandb", "kaggle_runner_mvp.yaml")


def test_run_launch_rejects_invalid_hydra_override_before_push(monkeypatch) -> None:
    pushed = False

    def fake_push(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal pushed
        pushed = True
        return {"command": [], "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(launcher, "_push_kernel", fake_push)

    args = launcher.PackageRequest(
        work_dir=launcher.DEFAULT_WORK_DIR,
        kernel_id="owner/kernel",
        title="test",
        sweep_yaml=launcher.DEFAULT_SWEEP,
        run_type="smoke",
        no_wandb=True,
        standalone_overrides=["training=2p4p_32_splitb"],
        accelerators=("NvidiaTeslaP100",),
        dry_run=False,
    )

    with pytest.raises(MissingConfigException, match="2p4p_32_splitb"):
        launcher.run_launch(args)

    assert not pushed


def test_run_preflight_rejects_invalid_hydra_override() -> None:
    args = launcher.PackageRequest(
        work_dir=launcher.DEFAULT_WORK_DIR,
        kernel_id="owner/kernel",
        title="test",
        sweep_yaml=launcher.DEFAULT_SWEEP,
        no_wandb=True,
        standalone_overrides=["training=2p4p_32_splitb"],
    )

    with pytest.raises(MissingConfigException, match="2p4p_32_split"):
        launcher.run_preflight(args)


def test_parse_kernel_ref_from_text_reads_kaggle_code_url() -> None:
    ref = parse_kernel_ref_from_text(
        "Kernel pushed to https://www.kaggle.com/code/jonduea/orbit-wars-kaggle-runner"
    )

    assert ref == "jonduea/orbit-wars-kaggle-runner"


def test_resolve_kaggle_username_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "from-env")

    assert resolve_kaggle_username() == "from-env"


def test_resolve_kaggle_username_reads_kaggle_json(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    config_dir = tmp_path / "kaggle-config"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(
        '{"username":"from-json","key":"secret"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    assert resolve_kaggle_username() == "from-json"


def test_resolve_kaggle_username_reads_credentials_json(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    config_dir = tmp_path / "kaggle-config"
    config_dir.mkdir()
    (config_dir / "credentials.json").write_text(
        '{"username":"from-oauth","access_token":"token"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    assert resolve_kaggle_username() == "from-oauth"


def test_default_kernel_id_uses_resolved_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    config_dir = tmp_path / "kaggle-config"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(
        '{"username":"jonduea","key":"secret"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    assert launcher.default_kernel_id() == "jonduea/orbit-wars-kaggle-runner"


def test_print_success_checklist_uses_kernel_from_push_stdout(capsys) -> None:
    launcher._print_success(
        {
            "stdout": (
                "Your kernel is accessible at: "
                "https://www.kaggle.com/code/jonduea/orbit-wars-kaggle-runner"
            ),
            "stderr": "",
        },
        kernel_id="replace-me/orbit-wars-kaggle-runner",
        ledger=Path("/tmp/ledger.jsonl"),
        standalone=True,
    )

    captured = capsys.readouterr().out
    assert '"kernel_id": "jonduea/orbit-wars-kaggle-runner"' in captured
    assert (
        '"kernel_url": "https://www.kaggle.com/code/jonduea/orbit-wars-kaggle-runner"'
        in captured
    )
    assert (
        '"verify_output": "uv run ow train kaggle sync '
        'jonduea/orbit-wars-kaggle-runner --force"'
        in captured
    )


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
    assert "jax" not in pyproject.lower()
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
