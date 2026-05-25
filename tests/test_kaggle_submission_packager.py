from __future__ import annotations

import argparse
import pickle
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_kaggle_docker_submission import (
    IN_CONTAINER_VALIDATOR,
    MAIN_TEMPLATE,
    ValidationError,
    _to_plain_data,
    build_submission_package,
    export_runtime_artifact,
    validate_tarball_layout,
)
from src.config.schema import TrainConfig


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(
            candidate_count=8,
            ship_bucket_count=8,
            max_fleets=256,
            player_count=2,
            max_ships=400.0,
            feature_history_steps=1,
            trajectory_shield_enabled=True,
            trajectory_shield_hit_mode="selected_target",
            trajectory_shield_horizon=500,
            trajectory_shield_epsilon=1e-6,
        ),
        reward=SimpleNamespace(
            reward_capture_planet=0.0,
            reward_ship_delta=0.0,
            reward_production_delta=0.0,
            reward_terminal_scale=1.0,
            early_terminal_reward_shaping_enabled=True,
            early_terminal_reward_shaping_horizon=500,
            terminal_reward_mode="binary_win",
        ),
        model=SimpleNamespace(
            architecture="gnn_pointer",
            hidden_size=16,
            attention_heads=4,
            max_moves_k=3,
            gnn_k_neighbors=3,
            gnn_message_passing_layers=1,
            normalize_observations=True,
            obs_norm_clip=10.0,
        ),
        training=SimpleNamespace(enable_gradient_checkpointing=False),
    )


def _write_fake_checkpoint(path: Path) -> None:
    payload = {
        "update": 7,
        "params": {"dense": {"kernel": np.zeros((2, 2), dtype=np.float32)}},
        "opt_state": {"must_not_ship": True},
        "rng_key": np.asarray([1, 2], dtype=np.uint32),
        "config": _fake_config(),
        "feature_metadata": {
            "feature_history_steps": 1,
            "self_feature_dim": 30,
            "candidate_feature_dim": 24,
            "global_feature_dim": 20,
        },
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)


def test_export_runtime_artifact_strips_training_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    _write_fake_checkpoint(checkpoint)

    artifact = export_runtime_artifact(checkpoint)

    assert artifact["checkpoint_update"] == 7
    assert artifact["config"]["model"]["architecture"] == "gnn_pointer"
    assert "opt_state" not in artifact
    assert "rng_key" not in artifact
    assert artifact["params"]["dense"]["kernel"].shape == (2, 2)


def test_plain_data_tolerates_old_pickled_dataclass_missing_new_field() -> None:
    cfg = TrainConfig()
    delattr(cfg.artifacts.artifact_pipeline, "replay_backend")

    data = _to_plain_data(cfg)

    assert "artifact_pipeline" in data["artifacts"]
    assert "replay_backend" not in data["artifacts"]["artifact_pipeline"]
    assert data["model"]["architecture"] == "gnn_pointer"


def test_build_submission_package_has_kaggle_root_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    output_dir = tmp_path / "package"
    _write_fake_checkpoint(checkpoint)
    args = argparse.Namespace(
        checkpoint=checkpoint,
        output_dir=output_dir,
        keep_staging=False,
        skip_docker=True,
        docker_image="unused",
        seed=42,
        player_count="both",
        timeout_seconds=1.0,
    )

    package_path = build_submission_package(args)

    assert package_path.is_file()
    with tarfile.open(package_path, "r:gz") as archive:
        names = set(archive.getnames())
        manifest = archive.extractfile("manifest.json")
        assert manifest is not None
        manifest_text = manifest.read().decode("utf-8")
    assert "main.py" in names
    assert "runtime_artifact.pkl" in names
    assert "manifest.json" in names
    assert "src/__init__.py" in names
    assert "src/jax/policy.py" in names
    assert "src/game/trajectory_shield.py" in names
    assert str(checkpoint.parent) not in manifest_text
    assert "source_checkpoint_sha256" in manifest_text


def test_validate_tarball_layout_rejects_traversal(tmp_path: Path) -> None:
    package_path = tmp_path / "bad.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("bad", encoding="utf-8")
    with tarfile.open(package_path, "w:gz") as archive:
        archive.add(payload, arcname="../payload.txt")

    with pytest.raises(ValidationError, match="Unsafe archive member"):
        validate_tarball_layout(package_path)


def test_validate_tarball_layout_rejects_symlink(tmp_path: Path) -> None:
    package_path = tmp_path / "bad-link.tar.gz"
    main_path = tmp_path / "main.py"
    main_path.write_text("def agent(obs):\n    return []\n", encoding="utf-8")
    with tarfile.open(package_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")
        link = tarfile.TarInfo("linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/target"
        archive.addfile(link)

    with pytest.raises(ValidationError, match="Unsafe archive member type"):
        validate_tarball_layout(package_path)


def test_embedded_runtime_templates_compile() -> None:
    compile(MAIN_TEMPLATE, "generated_main.py", "exec")
    compile(IN_CONTAINER_VALIDATOR, "validate_submission.py", "exec")
    assert "_agent_v1" not in MAIN_TEMPLATE
    assert "encoding_version" not in MAIN_TEMPLATE
    assert "empty_feature_history" in MAIN_TEMPLATE
    assert "FeatureExtractor" in MAIN_TEMPLATE


def test_export_runtime_artifact_accepts_gnn_pointer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "jax_ckpt_v2.pkl"
    config = _fake_config()
    config.model.architecture = "gnn_pointer"
    payload = {
        "update": 3,
        "params": {
            "params": {"planet_enc_0": {"kernel": np.zeros((13, 16), dtype=np.float32)}}
        },
        "config": config,
        "feature_metadata": {
            "schema_version": 4,
            "planet_feature_dim": 13,
            "edge_feature_dim": 18,
            "global_feature_dim": 46,
            "feature_history_steps": 1,
            "ship_feature_scale": 1000.0,
            "edge_layout": "top_k_per_source",
            "edge_k": 3,
            "intercept_anchors": (1.0, 6.0),
        },
    }
    with checkpoint.open("wb") as file:
        pickle.dump(payload, file)

    artifact = export_runtime_artifact(checkpoint)

    assert artifact["config"]["model"]["architecture"] == "gnn_pointer"
    assert artifact["feature_metadata"]["schema_version"] == 4
