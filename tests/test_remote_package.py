from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestration.kernel_package import render_kernel_package
from src.orchestration.remote_package import (
    DEFAULT_MAP_POOL_RELATIVE,
    RemotePackageOptions,
    copy_repo_payload,
    hydra_overrides_from_worker_env,
    payload_archive,
    render_remote_tarball,
    tarball_member_names,
    worker_env_with_hydra_overrides,
    write_worker_env,
)


def _seed_repo(repo: Path, *, include_map_pool: bool = False) -> None:
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "conf" / "task").mkdir(parents=True)
    (repo / "conf" / "task" / "base.yaml").write_text("name: base\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    for script in (
        "colab_worker_entry.py",
        "kaggle_worker_entry.py",
        "benchmark_jax_rl.py",
        "kaggle_runtime_env.py",
    ):
        (repo / "scripts" / script).write_text("print('ok')\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'orbit-wars'\n",
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text("lock\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    (repo / "outputs" / "campaigns").mkdir(parents=True)
    (repo / "outputs" / "campaigns" / "secret.txt").write_text("secret\n", encoding="utf-8")
    if include_map_pool:
        map_pool = repo / DEFAULT_MAP_POOL_RELATIVE
        map_pool.parent.mkdir(parents=True, exist_ok=True)
        map_pool.write_bytes(b"map-pool-bytes")


def test_tarball_includes_core_paths_and_excludes_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package_dir = tmp_path / "pkg"
    _seed_repo(repo)

    copy_repo_payload(repo_root=repo, package_dir=package_dir)
    payload = payload_archive(package_dir)
    members = tarball_member_names(payload)

    assert any(member.startswith("src/") for member in members)
    assert any(member.startswith("conf/") for member in members)
    assert "pyproject.toml" in members
    assert not any(member.startswith("outputs/") for member in members)


def test_map_pool_included_when_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package_dir = tmp_path / "pkg"
    _seed_repo(repo, include_map_pool=True)

    warnings_list = copy_repo_payload(repo_root=repo, package_dir=package_dir)
    payload = payload_archive(package_dir)
    members = tarball_member_names(payload)

    assert str(DEFAULT_MAP_POOL_RELATIVE) in members
    assert warnings_list == []


def test_map_pool_missing_warns_for_task_map_pool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package_dir = tmp_path / "pkg"
    _seed_repo(repo, include_map_pool=False)

    with pytest.warns(UserWarning, match="task=map_pool requested"):
        warnings_list = copy_repo_payload(
            repo_root=repo,
            package_dir=package_dir,
            options=RemotePackageOptions(hydra_overrides=("task=map_pool",)),
        )

    assert warnings_list
    assert not (package_dir / DEFAULT_MAP_POOL_RELATIVE).exists()


def test_worker_env_hydra_overrides_round_trip(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    overrides = ["training.total_updates=10", "task=shield_cheap"]
    env = worker_env_with_hydra_overrides(overrides, extra={"ORBIT_WARS_COLAB_WORKER_MODE": "standalone"})
    env_path = write_worker_env(package_dir=package_dir, env=env)
    loaded = json.loads(env_path.read_text(encoding="utf-8"))

    assert hydra_overrides_from_worker_env(loaded) == overrides
    assert loaded["ORBIT_WARS_COLAB_WORKER_MODE"] == "standalone"


def test_render_remote_tarball_writes_worker_env_and_archive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package_dir = tmp_path / "pkg"
    tarball_path = tmp_path / "orbit_wars.tgz"
    _seed_repo(repo, include_map_pool=True)

    result = render_remote_tarball(
        repo_root=repo,
        package_dir=package_dir,
        tarball_path=tarball_path,
        env=worker_env_with_hydra_overrides(["training.total_updates=3"]),
    )

    assert tarball_path.is_file()
    assert result.worker_env_path.is_file()
    members = tarball_member_names(tarball_path.read_bytes())
    assert "worker-env.json" in members
    assert str(DEFAULT_MAP_POOL_RELATIVE) in members


def test_render_kernel_package_regression_matches_kaggle_payload_layout(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _seed_repo(repo, include_map_pool=True)
    (repo / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'dependencies = [',
                '  "jax[cuda13]; sys_platform == \'linux\' and platform_machine == \'x86_64\'",',
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
    assert (package.package_dir / DEFAULT_MAP_POOL_RELATIVE).is_file()
    assert (package.package_dir / "scripts" / "kaggle_worker_entry.py").exists()
    assert "embedded payload" in package.worker_path.read_text(encoding="utf-8")
