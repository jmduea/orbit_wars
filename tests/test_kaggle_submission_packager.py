from __future__ import annotations

import argparse
import ast
import pickle
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_kaggle_docker_submission import (
    CONFIG_TEMPLATE,
    IN_CONTAINER_VALIDATOR,
    MAIN_TEMPLATE,
    RUNTIME_FILES,
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
            pointer_decoder="joint_flat",
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
        per_step_seconds=1.0,
        overage_budget_seconds=60.0,
        episode_steps=500,
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
    assert len(names) == len(set(names))
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
    assert "jitted_encode" in MAIN_TEMPLATE
    assert "_initialize_submission()" in MAIN_TEMPLATE
    assert "compile_batched_feature_encode" in MAIN_TEMPLATE
    assert "deterministic_eval=True" in MAIN_TEMPLATE
    assert "StepTimingBudget" in IN_CONTAINER_VALIDATOR


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
            "schema_version": 5,
            "planet_feature_dim": 13,
            "edge_feature_dim": 19,
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
    assert artifact["feature_metadata"]["schema_version"] == 5


def test_export_runtime_artifact_includes_factorized_pointer_decoder(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt_factorized.pkl"
    config = _fake_config()
    config.model.architecture = "gnn_pointer"
    config.model.pointer_decoder = "factorized_topk"
    payload = {
        "update": 5,
        "params": {
            "params": {"planet_enc_0": {"kernel": np.zeros((13, 16), dtype=np.float32)}}
        },
        "config": config,
        "feature_metadata": {
            "schema_version": 5,
            "planet_feature_dim": 13,
            "edge_feature_dim": 19,
            "global_feature_dim": 46,
            "feature_history_steps": 1,
            "ship_feature_scale": 1000.0,
            "edge_layout": "top_k_per_source",
            "edge_k": 3,
            "intercept_anchors": (1.0, 6.0),
            "pointer_decoder": "factorized_topk",
            "action_layout_version": 2,
        },
    }
    with checkpoint.open("wb") as file:
        pickle.dump(payload, file)

    artifact = export_runtime_artifact(checkpoint)

    assert artifact["feature_metadata"]["pointer_decoder"] == "factorized_topk"
    assert artifact["feature_metadata"]["action_layout_version"] == 2
    assert artifact["config"]["model"]["pointer_decoder"] == "factorized_topk"


def test_export_runtime_artifact_rejects_pointer_decoder_metadata_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt_mismatch.pkl"
    config = _fake_config()
    config.model.pointer_decoder = "joint_flat"
    payload = {
        "update": 1,
        "params": {"dense": {"kernel": np.zeros((2, 2), dtype=np.float32)}},
        "config": config,
        "feature_metadata": {
            "schema_version": 5,
            "pointer_decoder": "factorized_topk",
            "action_layout_version": 2,
        },
    }
    with checkpoint.open("wb") as file:
        pickle.dump(payload, file)

    with pytest.raises(ValidationError, match="pointer_decoder"):
        export_runtime_artifact(checkpoint)


def test_build_submission_package_includes_factorized_runtime_modules(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "jax_ckpt_factorized.pkl"
    config = _fake_config()
    config.model.pointer_decoder = "factorized_topk"
    payload = {
        "update": 2,
        "params": {"dense": {"kernel": np.zeros((2, 2), dtype=np.float32)}},
        "config": config,
        "feature_metadata": {
            "schema_version": 5,
            "pointer_decoder": "factorized_topk",
            "action_layout_version": 2,
        },
    }
    with checkpoint.open("wb") as file:
        pickle.dump(payload, file)
    args = argparse.Namespace(
        checkpoint=checkpoint,
        output_dir=tmp_path / "package",
        keep_staging=False,
        skip_docker=True,
        docker_image="unused",
        seed=42,
        player_count="both",
        per_step_seconds=1.0,
        overage_budget_seconds=60.0,
        episode_steps=500,
    )

    package_path = build_submission_package(args)

    with tarfile.open(package_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "src/jax/action_codec.py" in names
    assert "src/jax/decoder_carry.py" in names
    assert "src/jax/decoders/factorized_topk_pointer.py" in names
    assert "src/jax/distributional_value.py" in names
    assert "src/jax/encoders/_types.py" in names
    assert "src/jax/encoders/remat.py" in names
    assert "src/jax/factored_sequence_scan.py" in names
    assert "src/jax/ship_action.py" in names
    assert "src/features/catalog/planet.py" in names
    assert "src/features/catalog/edge.py" in names


def test_runtime_files_cover_static_submission_import_closure() -> None:
    packaged = {Path("src") / filename for filename in RUNTIME_FILES}
    packaged.update(
        {
            Path("src/__init__.py"),
            Path("src/config/__init__.py"),
            Path("src/config/schema.py"),
        }
    )
    ignored = {
        Path("src/artifacts/__init__.py"),
    }

    needed: set[Path] = set()
    for filename in RUNTIME_FILES:
        path = Path("src") / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_names.append(node.module)
            for module_name in module_names:
                if not module_name.startswith("src."):
                    continue
                parts = module_name.split(".")
                for index in range(2, len(parts) + 1):
                    module_path = Path(*parts[:index])
                    py_file = module_path.with_suffix(".py")
                    init_file = module_path / "__init__.py"
                    if py_file.exists():
                        needed.add(py_file)
                    if init_file.exists():
                        needed.add(init_file)

    assert sorted(needed - packaged - ignored) == []


def test_embedded_runtime_config_template_matches_runtime_fields() -> None:
    namespace: dict[str, object] = {}
    exec(CONFIG_TEMPLATE, namespace)

    template_task = namespace["TaskConfig"]()
    template_model = namespace["ModelConfig"]()
    runtime_task = TrainConfig().task
    runtime_model = TrainConfig().model

    for field in (
        "ship_action_mode",
        "edge_rank_mode",
        "value_bins",
        "value_max",
        "decoder_carry",
    ):
        template = template_task if hasattr(template_task, field) else template_model
        runtime = runtime_task if hasattr(runtime_task, field) else runtime_model
        assert getattr(template, field) == getattr(runtime, field)
