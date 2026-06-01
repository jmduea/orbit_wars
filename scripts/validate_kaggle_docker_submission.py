from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax

DOCKER_IMAGE = "gcr.io/kaggle-images/python-simulations"
RUNTIME_FORMAT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUNTIME_FILES = (
    "features/__init__.py",
    "features/catalog/__init__.py",
    "features/catalog/_core.py",
    "features/catalog/_types.py",
    "features/catalog/edge.py",
    "features/catalog/global_.py",
    "features/catalog/planet.py",
    "features/registry.py",
    "features/schema_api.py",
    "game/__init__.py",
    "game/constants.py",
    "game/shield_config.py",
    "game/shield.py",
    "jax/rewards.py",
    "game/types.py",
    "jax/__init__.py",
    "jax/action_codec.py",
    "jax/action_sampling.py",
    "jax/decoder_carry.py",
    "jax/decoders/__init__.py",
    "jax/decoders/factorized_topk_pointer.py",
    "jax/distributional_value.py",
    "jax/encoders/__init__.py",
    "jax/encoders/_types.py",
    "jax/encoders/planet_encoder_common.py",
    "jax/encoders/planet_graph_transformer.py",
    "jax/encoders/remat.py",
    "jax/factored_sequence_scan.py",
    "jax/launch_hygiene.py",
    "jax/feature_primitives.py",
    "jax/features.py",
    "jax/shield/__init__.py",
    "jax/shield/trajectory.py",
    "jax/policy.py",
    "jax/submission_runtime.py",
    "jax/env.py",
    "jax/map_pool/__init__.py",
    "jax/map_pool/comets.py",
    "jax/map_pool/home_assignment.py",
    "jax/rollout/__init__.py",
    "jax/rollout/types.py",
    "jax/ship_action.py",
    "opponents/__init__.py",
    "opponents/jax_actions/__init__.py",
    "opponents/jax_actions/builders.py",
    "artifacts/checkpoint_compat.py",
    "artifacts/timing.py",
)
TRAINING_ONLY_IMPORTS = (
    "hydra",
    "omegaconf",
    "wandb",
    "optax",
    "src.train",
    "src.jax.train",
)


class ValidationError(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        super().__init__(f"{phase}: {message}")
        self.phase = phase
        self.message = message


def main() -> int:
    args = parse_args()
    try:
        package_path = build_submission_package(args)
        print(f"package_path={package_path}")
        if args.skip_docker:
            print("docker_validation=skipped")
            return 0
        run_docker_validation(package_path, args)
    except ValidationError as exc:
        print(
            json.dumps({"ok": False, "phase": exc.phase, "error": exc.message}),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - last-resort CLI guard
        print(
            json.dumps({"ok": False, "phase": "unexpected_failure", "error": str(exc)}),
            file=sys.stderr,
        )
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and validate an Orbit Wars Kaggle submission in Kaggle Docker."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/kaggle_submission")
    )
    parser.add_argument("--docker-image", default=DOCKER_IMAGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--player-count",
        choices=("2", "4", "both"),
        default="both",
        help="Self-play player counts to validate inside Docker.",
    )
    parser.add_argument(
        "--per-step-seconds",
        type=float,
        default=1.0,
        help="Target per-turn inference budget after setup (Kaggle default: 1s).",
    )
    parser.add_argument(
        "--overage-budget-seconds",
        type=float,
        default=60.0,
        help="Total allowed cumulative per-step overrun across one episode.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=500,
        help="Maximum episode steps for Docker self-play validation.",
    )
    parser.add_argument("--skip-docker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def build_submission_package(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise ValidationError(
            "checkpoint_missing", f"Checkpoint does not exist: {checkpoint_path}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    artifact = export_runtime_artifact(checkpoint_path)
    write_runtime_package(staging_dir, artifact)
    package_path = output_dir / "submission.tar.gz"
    if package_path.exists():
        package_path.unlink()
    create_tarball(staging_dir, package_path)
    validate_tarball_layout(package_path)
    if not args.keep_staging:
        shutil.rmtree(staging_dir)
    return package_path


def export_runtime_artifact(checkpoint_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with checkpoint_path.open("rb") as file:
            checkpoint = pickle.load(file)
    except Exception as exc:
        raise ValidationError("checkpoint_load_failed", str(exc)) from exc

    if not isinstance(checkpoint, Mapping) or "params" not in checkpoint:
        raise ValidationError(
            "checkpoint_schema_failed", "Checkpoint must be a mapping with params"
        )

    config = checkpoint.get("config")
    config_dict = _to_plain_data(config)
    if not isinstance(config_dict, dict):
        raise ValidationError(
            "checkpoint_schema_failed", "Checkpoint config could not be serialized"
        )
    task_cfg = _require_dict(config_dict, "task")
    reward_cfg = _require_dict(config_dict, "reward")
    model_cfg = _require_dict(config_dict, "model")
    training_cfg = _require_dict(config_dict, "training")
    architecture = str(model_cfg.get("architecture", "")).strip().lower()
    if architecture not in {"planet_graph_transformer"}:
        raise ValidationError(
            "unsupported_architecture", f"Unsupported architecture: {architecture!r}"
        )

    feature_metadata = _to_plain_data(checkpoint.get("feature_metadata", {}))
    if not isinstance(feature_metadata, dict):
        feature_metadata = {}
    stored_decoder = feature_metadata.get("pointer_decoder")
    model_decoder = (
        str(model_cfg.get("pointer_decoder", "factorized_topk")).strip().lower()
    )
    if stored_decoder is not None and str(stored_decoder) != model_decoder:
        raise ValidationError(
            "checkpoint_schema_failed",
            "Checkpoint model.pointer_decoder disagrees with feature_metadata.pointer_decoder",
        )

    params = checkpoint["params"]
    if isinstance(params, Mapping) and set(params.keys()) == {"params"}:
        params = params["params"]
    params = jax.tree_util.tree_map(lambda value: jax.device_get(value), params)
    artifact = {
        "format_version": RUNTIME_FORMAT_VERSION,
        "exported_at_unix": time.time(),
        "source_checkpoint_name": checkpoint_path.name,
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_update": int(checkpoint.get("update", -1)),
        "params": params,
        "config": {
            "task": task_cfg,
            "reward": reward_cfg,
            "model": model_cfg,
            "training": {
                "enable_gradient_checkpointing": bool(
                    training_cfg.get("enable_gradient_checkpointing", False)
                )
            },
        },
        "feature_metadata": feature_metadata,
        "export_seconds": time.perf_counter() - start,
    }
    return artifact


def write_runtime_package(staging_dir: Path, artifact: dict[str, Any]) -> None:
    (staging_dir / "main.py").write_text(MAIN_TEMPLATE, encoding="utf-8")
    with (staging_dir / "runtime_artifact.pkl").open("wb") as file:
        pickle.dump(artifact, file)
    runtime_files = RUNTIME_FILES
    manifest = {
        "format_version": RUNTIME_FORMAT_VERSION,
        "checkpoint_update": artifact["checkpoint_update"],
        "source_checkpoint_name": artifact["source_checkpoint_name"],
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
        "architecture": artifact["config"]["model"].get("architecture"),
        "pointer_decoder": artifact["feature_metadata"].get("pointer_decoder")
        or artifact["config"]["model"].get("pointer_decoder"),
        "action_layout_version": artifact["feature_metadata"].get(
            "action_layout_version"
        ),
        "feature_metadata": artifact["feature_metadata"],
        "forbidden_runtime_imports": TRAINING_ONLY_IMPORTS,
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    package_dir = staging_dir / "src"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "from .config import TrainConfig\n", encoding="utf-8"
    )
    config_dir = package_dir / "config"
    config_dir.mkdir()
    (config_dir / "__init__.py").write_text(CONFIG_TEMPLATE, encoding="utf-8")
    (config_dir / "schema.py").write_text(CONFIG_TEMPLATE, encoding="utf-8")
    repo_src = Path(__file__).resolve().parents[1] / "src"
    for filename in runtime_files:
        destination = package_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_src / filename, destination)
    artifacts_init = package_dir / "artifacts" / "__init__.py"
    artifacts_init.write_text(
        '"""Submission runtime artifacts package."""\n', encoding="utf-8"
    )
    opponents_init = package_dir / "opponents" / "__init__.py"
    opponents_init.write_text(
        '"""Submission runtime opponents package."""\n', encoding="utf-8"
    )


def create_tarball(staging_dir: Path, package_path: Path) -> None:
    with tarfile.open(package_path, "w:gz") as archive:
        for path in sorted(staging_dir.rglob("*")):
            archive.add(path, arcname=path.relative_to(staging_dir), recursive=False)


def validate_tarball_layout(package_path: Path) -> None:
    with tarfile.open(package_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        for member in members:
            _validate_tar_member(member, Path("."), phase_error=ValidationError)
        if "main.py" not in names:
            raise ValidationError(
                "package_layout_failed", "submission.tar.gz must contain root main.py"
            )


def _resolve_docker_cli() -> str:
    """Return a usable docker CLI path or raise with WSL/Docker Desktop hints."""
    docker = shutil.which("docker")
    if docker is not None:
        return docker

    usr_bin = Path("/usr/bin/docker")
    if usr_bin.is_symlink() and not usr_bin.exists():
        try:
            target = os.readlink(usr_bin)
        except OSError:
            target = "<unknown>"
        raise ValidationError(
            "docker_unavailable",
            "Docker CLI symlink at /usr/bin/docker points to a missing target "
            f"({target}). On WSL with Docker Desktop: open Docker Desktop → Settings → "
            "Resources → WSL Integration, enable this distro, then verify `docker info` "
            "in a fresh shell.",
        )

    if Path("/var/run/docker.sock").exists():
        raise ValidationError(
            "docker_unavailable",
            "Docker socket exists at /var/run/docker.sock but the docker CLI was not found on PATH. "
            "Install the docker CLI or enable Docker Desktop WSL integration.",
        )

    raise ValidationError("docker_unavailable", "docker executable was not found")


def run_docker_validation(package_path: Path, args: argparse.Namespace) -> None:
    docker_cli = _resolve_docker_cli()
    if str(args.docker_image).startswith("-"):
        raise ValidationError(
            "docker_image_invalid", "Docker image must not start with '-'"
        )
    docker_info = subprocess.run(
        [docker_cli, "info", "--format", "{{.ServerVersion}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if docker_info.returncode != 0:
        message = (
            docker_info.stderr or docker_info.stdout or "docker daemon is not reachable"
        ).strip()
        raise ValidationError("docker_unavailable", message)
    replay_output_dir = args.output_dir.resolve() / "replays"
    replay_output_dir.mkdir(parents=True, exist_ok=True)
    replay_output_dir.chmod(0o777)
    with tempfile.TemporaryDirectory(
        prefix="orbit-wars-kaggle-docker-"
    ) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        validator_path = temp_dir / "validate_submission.py"
        validator_path.write_text(IN_CONTAINER_VALIDATOR, encoding="utf-8")
        command = [
            docker_cli,
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "-v",
            f"{package_path.resolve()}:/work/submission.tar.gz:ro",
            "-v",
            f"{validator_path.resolve()}:/work/validate_submission.py:ro",
            "-v",
            f"{replay_output_dir}:/work/replays:rw",
            args.docker_image,
            "python",
            "/work/validate_submission.py",
            "--package",
            "/work/submission.tar.gz",
            "--seed",
            str(args.seed),
            "--player-count",
            args.player_count,
            "--per-step-seconds",
            str(args.per_step_seconds),
            "--overage-budget-seconds",
            str(args.overage_budget_seconds),
            "--episode-steps",
            str(args.episode_steps),
            "--replay-output-dir",
            "/work/replays",
            "--checkpoint-update",
            str(_checkpoint_update_from_package_manifest(package_path)),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            if completed.returncode == 137:
                raise ValidationError(
                    "docker_container_killed",
                    "docker exited 137; container was killed, commonly by OOM or an external stop",
                )
            raise ValidationError(
                "docker_validation_failed", f"docker exited {completed.returncode}"
            )


def _checkpoint_update_from_package_manifest(package_path: Path) -> int:
    try:
        with tarfile.open(package_path, "r:gz") as archive:
            manifest_file = archive.extractfile("manifest.json")
            if manifest_file is None:
                return -1
            manifest = json.loads(manifest_file.read().decode("utf-8"))
    except Exception:
        return -1
    update = manifest.get("checkpoint_update")
    return int(update) if isinstance(update, int) else -1


def _validate_tar_member(
    member: tarfile.TarInfo, root: Path, *, phase_error: type[Exception]
) -> None:
    path = Path(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise phase_error(
            "package_layout_failed", f"Unsafe archive member: {member.name}"
        )
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise phase_error(
            "package_layout_failed", f"Unsafe archive member type: {member.name}"
        )
    if member.size > 512 * 1024 * 1024:
        raise phase_error(
            "package_layout_failed", f"Archive member too large: {member.name}"
        )
    target = (root / path).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise phase_error(
            "package_layout_failed",
            f"Archive member escapes extraction root: {member.name}",
        )


def _require_dict(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValidationError(
            "checkpoint_schema_failed", f"Checkpoint config missing {key}"
        )
    return value


def _to_plain_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_plain_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if hasattr(value, field.name)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _to_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_submission_root() -> Path:
    """Return the directory containing packaged submission artifacts.

    Mirrors the resolver embedded in generated ``main.py``. Kaggle loads agents via
    ``exec()`` without ``__file__``; ``kaggle_environments`` appends ``dirname(main.py)``
    to ``sys.path`` before executing the agent module.
    """

    for entry in sys.path:
        candidate = Path(entry)
        if (candidate / "runtime_artifact.pkl").is_file():
            return candidate
    cwd = Path.cwd()
    if (cwd / "runtime_artifact.pkl").is_file():
        return cwd
    return cwd


def import_submission_kaggle_exec(extract_dir: Path) -> tuple[Any, float]:
    """Load ``main.py`` the way ``kaggle_environments.agent.get_last_callable`` does."""

    started = time.perf_counter()
    main_path = extract_dir / "main.py"
    raw = main_path.read_text(encoding="utf-8")
    env: dict[str, Any] = {}
    exec_dir = str(extract_dir.resolve())
    sys.path.append(exec_dir)
    try:
        exec(compile(raw, str(main_path), "exec"), env)
    except Exception as exc:
        raise ValidationError(
            "submission_import_failed", f"kaggle exec: {exc}"
        ) from exc
    finally:
        if sys.path and sys.path[-1] == exec_dir:
            sys.path.pop()

    agent = env.get("agent")
    if not callable(agent):
        raise ValidationError(
            "submission_import_failed",
            "kaggle exec: main.py does not expose callable agent",
        )
    state = env.get("_STATE")
    if state is None or not state.get("ready"):
        raise ValidationError(
            "setup_failed",
            "kaggle exec: submission setup must finish at import time",
        )
    return agent, time.perf_counter() - started


CONFIG_TEMPLATE = """from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskConfig:
    candidate_count: int = 8
    ship_bucket_count: int = 8
    ship_action_mode: str = "buckets"
    max_fleets: int = 256
    player_count: int = 2
    max_ships: float = 400.0
    ship_feature_scale: float = 1000.0
    feature_history_steps: int = 1
    trajectory_shield_mode: str = "cheap"
    trajectory_shield_hit_mode: str = "selected_target"
    trajectory_shield_horizon: int = 500
    trajectory_shield_epsilon: float = 1e-6
    intercept_anchors: tuple[float, ...] = (1.0, 3.0, 6.0)
    edge_rank_mode: str = "snapshot"


@dataclass(slots=True)
class RewardConfig:
    reward_capture_planet: float = 0.0
    reward_ship_delta: float = 0.0
    reward_production_delta: float = 0.0
    reward_terminal_scale: float = 1.0
    early_terminal_reward_shaping_enabled: bool = True
    early_terminal_reward_shaping_horizon: int = 500
    terminal_reward_mode: str = "binary_win"


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "planet_graph_transformer"
    value_head: str = "shared"
    value_bins: int = 51
    value_max: float = 1.0
    pointer_decoder: str = "factorized_topk"
    decoder_carry: bool = False
    hidden_size: int = 128
    attention_heads: int = 4
    max_moves_k: int = 3
    gnn_k_neighbors: int = 5
    gnn_message_passing_layers: int = 2
    planet_transformer_layers: int = 2
    spatial_attention_bias: bool = True
    normalize_observations: bool = True
    obs_norm_clip: float = 10.0


@dataclass(slots=True)
class TrainingConfig:
    enable_gradient_checkpointing: bool = False


@dataclass(slots=True)
class TrainConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    field_names = {item.name for item in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in field_names})


def config_from_plain(data: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        task=_from_dict(TaskConfig, data.get("task", {})),
        reward=_from_dict(RewardConfig, data.get("reward", {})),
        model=_from_dict(ModelConfig, data.get("model", {})),
        training=_from_dict(TrainingConfig, data.get("training", {})),
    )
"""


MAIN_TEMPLATE = r"""from __future__ import annotations

import json
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp

from src.config import config_from_plain
from src.jax.policy import build_jax_policy


def _submission_root():
    from pathlib import Path

    import sys

    for entry in sys.path:
        candidate = Path(entry)
        if (candidate / "runtime_artifact.pkl").is_file():
            return candidate
    cwd = Path(".")
    if (cwd / "runtime_artifact.pkl").is_file():
        return cwd
    return cwd


_ROOT = _submission_root()
_ARTIFACT_PATH = _ROOT / "runtime_artifact.pkl"
_MANIFEST_PATH = _ROOT / "manifest.json"
_STATE = None


def _load_state():
    global _STATE
    if _STATE is not None:
        return _STATE
    started = time.perf_counter()
    with _ARTIFACT_PATH.open("rb") as file:
        artifact = pickle.load(file)
    cfg = config_from_plain(artifact["config"])
    from src.jax.submission_runtime import apply_feature_metadata_to_model_config

    cfg = apply_feature_metadata_to_model_config(cfg, artifact.get("feature_metadata"))
    params = artifact["params"]
    policy = build_jax_policy(cfg)
    _warm_policy(policy, params, cfg)
    from src.jax.features import empty_feature_history

    history = empty_feature_history(cfg.task)
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    _STATE = {
        "cfg": cfg,
        "params": params,
        "policy": policy,
        "history": history,
        "manifest": manifest,
        "load_seconds": time.perf_counter() - started,
        "ready": False,
    }
    return _STATE


def _warm_policy(policy, params, cfg) -> None:
    from src.jax.policy import make_synthetic_turn_batch

    batch = make_synthetic_turn_batch(1, cfg.task)
    player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)
    outputs = policy.apply(
        {"params": params},
        batch,
        player_count=player_count,
    )
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), outputs)


def _act_on_obs(obs: Any, state: dict[str, Any]) -> list[list[float | int]]:
    from src.jax.submission_runtime import moves_from_jax_action

    game = state["parse_game"](obs)
    if not bool(jnp.any(game.planets.active)):
        return []
    game_batched, batch_batched = state["jitted_encode"](game, state["history"])
    action = state["jitted_act"](
        game_batched,
        batch_batched,
        jax.random.PRNGKey(0),
    )
    jax.block_until_ready(action.valid)
    state["history"] = state["jitted_append_history"](state["history"], game)
    return moves_from_jax_action(action, env_index=0)


def _warm_submission_path(state: dict[str, Any]) -> None:
    # Compile the full encode -> policy -> shield path before the first live obs.
    from kaggle_environments import make

    player_count = max(2, int(state["cfg"].task.player_count))
    env = make(
        "orbit_wars",
        configuration={"seed": 0, "episodeSteps": 500},
        debug=True,
    )
    env.run([lambda _obs: [] for _ in range(player_count)])
    obs = env.steps[0][0].observation
    _act_on_obs(obs, state)


def _initialize_submission() -> None:
    state = _load_state()
    if state.get("ready"):
        return
    from functools import partial

    from src.jax.submission_runtime import (
        compile_batched_feature_encode,
        compile_feature_history_append,
        compile_shielded_policy_act,
        jax_game_from_observation_fast,
    )

    task_cfg = state["cfg"].task
    state["parse_game"] = partial(
        jax_game_from_observation_fast,
        max_fleet_slots=int(task_cfg.max_fleets),
    )
    state["jitted_encode"] = compile_batched_feature_encode(task_cfg)
    state["jitted_append_history"] = compile_feature_history_append(task_cfg)
    state["jitted_act"] = compile_shielded_policy_act(
        state["policy"],
        {"params": state["params"]},
        state["cfg"],
        deterministic=True,
        deterministic_eval=True,
    )
    _warm_submission_path(state)
    state["ready"] = True


def agent(obs: Any) -> list[list[float | int]]:
    if _STATE is None or not _STATE.get("ready"):
        raise RuntimeError("submission setup must finish before the first observation")
    return _act_on_obs(obs, _STATE)


_initialize_submission()
"""


IN_CONTAINER_VALIDATOR = r"""from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    try:
        dependency_probe()
        with tempfile.TemporaryDirectory(prefix="submission-") as extract_text:
            extract_dir = Path(extract_text)
            extract_package(args.package, extract_dir)
            import_submission_kaggle_exec(extract_dir)
            module, import_seconds = import_submission(extract_dir)
            artifact_load_seconds = setup_probe(module)
            first_action = first_action_probe(
                module,
                args.per_step_seconds,
                args.overage_budget_seconds,
                args.episode_steps,
                extract_dir,
            )
            results = []
            player_counts = [2, 4] if args.player_count == "both" else [int(args.player_count)]
            for player_count in player_counts:
                modules = [import_submission(extract_dir, module_name=f"submission_main_p{idx}_{player_count}")[0] for idx in range(player_count)]
                results.append(run_episode(
                    modules,
                    player_count,
                    args.seed,
                    args.per_step_seconds,
                    args.overage_budget_seconds,
                    args.episode_steps,
                    args.replay_output_dir,
                    args.checkpoint_update,
                    extract_dir,
                ))
            print(json.dumps({
                "ok": True,
                "phase": "complete",
                "docker_image": "gcr.io/kaggle-images/python-simulations",
                "package": str(args.package),
                "import_seconds": import_seconds,
                "setup_seconds": artifact_load_seconds,
                "first_action": first_action,
                "episodes": results,
            }, indent=2))
            return 0
    except PhaseError as exc:
        print(json.dumps({"ok": False, "phase": exc.phase, "error": exc.message}), file=sys.stderr)
        return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--player-count", choices=("2", "4", "both"), default="both")
    parser.add_argument("--per-step-seconds", type=float, default=1.0)
    parser.add_argument("--overage-budget-seconds", type=float, default=60.0)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--replay-output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-update", type=int, default=-1)
    return parser.parse_args()


class PhaseError(RuntimeError):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = phase
        self.message = message


def dependency_probe():
    try:
        import flax
        import jax
        import numpy
        from kaggle_environments import make
        make("orbit_wars", debug=True)
    except Exception as exc:
        raise PhaseError("dependency_probe_failed", str(exc)) from exc
    print(json.dumps({
        "phase": "dependency_probe",
        "jax": getattr(jax, "__version__", "unknown"),
        "flax": getattr(flax, "__version__", "unknown"),
        "numpy": getattr(numpy, "__version__", "unknown"),
        "devices": [str(device) for device in jax.devices()],
    }))


def extract_package(package_path: Path, extract_dir: Path):
    try:
        with tarfile.open(package_path, "r:gz") as archive:
            for member in archive.getmembers():
                validate_member(member, extract_dir)
            archive.extractall(extract_dir, filter="data")
    except PhaseError:
        raise
    except Exception as exc:
        raise PhaseError("package_layout_failed", str(exc)) from exc
    if not (extract_dir / "main.py").is_file():
        raise PhaseError("package_layout_failed", "missing root main.py")


def import_submission_kaggle_exec(extract_dir: Path) -> tuple[Any, float]:
    started = time.perf_counter()
    main_path = extract_dir / "main.py"
    raw = main_path.read_text(encoding="utf-8")
    env: dict[str, Any] = {}
    exec_dir = str(extract_dir.resolve())
    sys.path.append(exec_dir)
    try:
        exec(compile(raw, str(main_path), "exec"), env)
    except Exception as exc:
        raise PhaseError("submission_import_failed", f"kaggle exec: {exc}") from exc
    finally:
        if sys.path and sys.path[-1] == exec_dir:
            sys.path.pop()
    agent = env.get("agent")
    if not callable(agent):
        raise PhaseError(
            "submission_import_failed",
            "kaggle exec: main.py does not expose callable agent",
        )
    state = env.get("_STATE")
    if state is None or not state.get("ready"):
        raise PhaseError(
            "setup_failed",
            "kaggle exec: submission setup must finish at import time",
        )
    return agent, time.perf_counter() - started


def validate_member(member, extract_dir: Path):
    path = Path(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise PhaseError("package_layout_failed", f"unsafe member {member.name}")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise PhaseError("package_layout_failed", f"unsafe member type {member.name}")
    if member.size > 512 * 1024 * 1024:
        raise PhaseError("package_layout_failed", f"archive member too large {member.name}")
    target = (extract_dir / path).resolve()
    root = extract_dir.resolve()
    if target != root and root not in target.parents:
        raise PhaseError("package_layout_failed", f"member escapes extraction root {member.name}")


def import_submission(extract_dir: Path, module_name="submission_main"):
    started = time.perf_counter()
    sys.path.insert(0, str(extract_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, extract_dir / "main.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PhaseError("submission_import_failed", str(exc)) from exc
    if not callable(getattr(module, "agent", None)):
        raise PhaseError("submission_import_failed", "main.py does not expose callable agent")
    return module, time.perf_counter() - started


def load_timing_module(extract_dir: Path):
    src_root = extract_dir / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from artifacts.timing import StepTimingBudget, TournamentTimingError

    return StepTimingBudget, TournamentTimingError


def record_timing(timing, elapsed: float, *, timing_error_type) -> None:
    try:
        timing.record(elapsed)
    except timing_error_type as exc:
        raise PhaseError("overage_budget_failed", str(exc)) from exc


def setup_probe(module) -> float:
    started = time.perf_counter()
    state = getattr(module, "_STATE", None)
    if state is None or not state.get("ready"):
        raise PhaseError(
            "setup_failed",
            "submission setup must finish at import time before the first observation",
        )
    return time.perf_counter() - started


def first_action_probe(
    module,
    per_step_seconds: float,
    overage_budget_seconds: float,
    episode_steps: int,
    extract_dir: Path,
) -> dict[str, float | int]:
    from kaggle_environments import make

    StepTimingBudget, TournamentTimingError = load_timing_module(extract_dir)
    env = make("orbit_wars", configuration={"seed": 123, "episodeSteps": episode_steps}, debug=True)
    env.run([lambda obs: [] for _ in range(2)])
    obs = env.steps[0][0].observation
    timing = StepTimingBudget(per_step_seconds, overage_budget_seconds)
    started = time.perf_counter()
    try:
        action = module.agent(obs)
    except Exception as exc:
        raise PhaseError("first_action_failed", str(exc)) from exc
    elapsed = time.perf_counter() - started
    record_timing(timing, elapsed, timing_error_type=TournamentTimingError)
    validate_action(action)
    return {"action_seconds": elapsed, **timing.summary()}


def run_episode(
    modules,
    player_count: int,
    seed: int,
    per_step_seconds: float,
    overage_budget_seconds: float,
    episode_steps: int,
    replay_output_dir,
    checkpoint_update: int,
    extract_dir: Path,
):
    from kaggle_environments import make

    StepTimingBudget, TournamentTimingError = load_timing_module(extract_dir)
    timing = StepTimingBudget(per_step_seconds, overage_budget_seconds)

    def make_timed_agent(module):
        def timed_agent(obs):
            started = time.perf_counter()
            action = module.agent(obs)
            elapsed = time.perf_counter() - started
            record_timing(timing, elapsed, timing_error_type=TournamentTimingError)
            validate_action(action)
            return action

        return timed_agent

    agents = [make_timed_agent(module) for module in modules]

    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": episode_steps}, debug=True)
    try:
        env.run(agents)
    except PhaseError:
        raise
    except Exception as exc:
        raise PhaseError(f"episode_failed_{player_count}p", str(exc)) from exc
    final = env.steps[-1]
    statuses = [getattr(step, "status", None) for step in final]
    rewards = [getattr(step, "reward", None) for step in final]
    if any(status not in ("DONE", "ACTIVE", None) for status in statuses):
        raise PhaseError(f"episode_failed_{player_count}p", f"bad statuses: {statuses}")
    replay_html_path = None
    if replay_output_dir is not None:
        replay_output_dir.mkdir(parents=True, exist_ok=True)
        update_label = f"{checkpoint_update:06d}" if checkpoint_update >= 0 else "unknown"
        replay_html_path = replay_output_dir / f"replay_u{update_label}_{player_count}p.html"
        replay_html_path.write_text(env.render(mode="html"), encoding="utf-8")
    return {
        "player_count": player_count,
        "statuses": statuses,
        "rewards": rewards,
        "episode_steps": episode_steps,
        "replay_html_path": str(replay_html_path) if replay_html_path is not None else None,
        **timing.summary(),
    }


def validate_action(action):
    if not isinstance(action, list):
        raise PhaseError("invalid_action_failed", f"action must be list, got {type(action).__name__}")
    for move in action:
        if not isinstance(move, list) or len(move) != 3:
            raise PhaseError("invalid_action_failed", f"invalid move shape: {move!r}")
        source_id, angle, ships = move
        if not isinstance(source_id, int):
            raise PhaseError("invalid_action_failed", f"source id must be int: {move!r}")
        if not isinstance(angle, (int, float)) or not math.isfinite(float(angle)):
            raise PhaseError("invalid_action_failed", f"angle must be finite number: {move!r}")
        if not isinstance(ships, int) or ships <= 0:
            raise PhaseError("invalid_action_failed", f"ships must be positive int: {move!r}")


if __name__ == "__main__":
    raise SystemExit(main())
"""


if __name__ == "__main__":
    raise SystemExit(main())
