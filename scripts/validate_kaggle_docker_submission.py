from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
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
    "constants.py",
    "feature_registry.py",
    "features.py",
    "game_types.py",
    "jax_policy.py",
    "trajectory_shield.py",
)
TRAINING_ONLY_IMPORTS = ("hydra", "omegaconf", "wandb", "optax", "src.train", "src.jax_train")


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
        print(json.dumps({"ok": False, "phase": exc.phase, "error": exc.message}), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - last-resort CLI guard
        print(json.dumps({"ok": False, "phase": "unexpected_failure", "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and validate an Orbit Wars Kaggle submission in Kaggle Docker."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/kaggle_submission"))
    parser.add_argument("--docker-image", default=DOCKER_IMAGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--player-count",
        choices=("2", "4", "both"),
        default="both",
        help="Self-play player counts to validate inside Docker.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1.0)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=500,
        help="Maximum episode steps for Docker self-play validation.",
    )
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def build_submission_package(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise ValidationError("checkpoint_missing", f"Checkpoint does not exist: {checkpoint_path}")

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
        raise ValidationError("checkpoint_schema_failed", "Checkpoint must be a mapping with params")

    config = checkpoint.get("config")
    config_dict = _to_plain_data(config)
    if not isinstance(config_dict, dict):
        raise ValidationError("checkpoint_schema_failed", "Checkpoint config could not be serialized")
    env_cfg = _require_dict(config_dict, "env")
    model_cfg = _require_dict(config_dict, "model")
    ppo_cfg = _require_dict(config_dict, "ppo")
    architecture = str(model_cfg.get("architecture", "")).strip().lower()
    if architecture not in {
        "mlp",
        "attention",
        "transformer",
        "mlp_pointer",
        "transformer_pointer",
        "gnn_pointer",
    }:
        raise ValidationError("unsupported_architecture", f"Unsupported architecture: {architecture!r}")

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
            "env": env_cfg,
            "model": model_cfg,
            "ppo": {
                "enable_gradient_checkpointing": bool(
                    ppo_cfg.get("enable_gradient_checkpointing", False)
                )
            },
        },
        "feature_metadata": _to_plain_data(checkpoint.get("feature_metadata", {})),
        "export_seconds": time.perf_counter() - start,
    }
    return artifact


def write_runtime_package(staging_dir: Path, artifact: dict[str, Any]) -> None:
    (staging_dir / "main.py").write_text(MAIN_TEMPLATE, encoding="utf-8")
    with (staging_dir / "runtime_artifact.pkl").open("wb") as file:
        pickle.dump(artifact, file)
    manifest = {
        "format_version": RUNTIME_FORMAT_VERSION,
        "checkpoint_update": artifact["checkpoint_update"],
        "source_checkpoint_name": artifact["source_checkpoint_name"],
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
        "architecture": artifact["config"]["model"].get("architecture"),
        "feature_metadata": artifact["feature_metadata"],
        "forbidden_runtime_imports": TRAINING_ONLY_IMPORTS,
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    package_dir = staging_dir / "src"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(CONFIG_TEMPLATE, encoding="utf-8")
    (package_dir / "config.py").write_text(CONFIG_TEMPLATE, encoding="utf-8")
    (package_dir / "conf_schema.py").write_text(CONFIG_TEMPLATE, encoding="utf-8")
    repo_src = Path(__file__).resolve().parents[1] / "src"
    for filename in RUNTIME_FILES:
        shutil.copy2(repo_src / filename, package_dir / filename)


def create_tarball(staging_dir: Path, package_path: Path) -> None:
    with tarfile.open(package_path, "w:gz") as archive:
        for path in sorted(staging_dir.rglob("*")):
            archive.add(path, arcname=path.relative_to(staging_dir))


def validate_tarball_layout(package_path: Path) -> None:
    with tarfile.open(package_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        for member in members:
            _validate_tar_member(member, Path("."), phase_error=ValidationError)
        if "main.py" not in names:
            raise ValidationError("package_layout_failed", "submission.tar.gz must contain root main.py")


def run_docker_validation(package_path: Path, args: argparse.Namespace) -> None:
    if shutil.which("docker") is None:
        raise ValidationError("docker_unavailable", "docker executable was not found")
    if str(args.docker_image).startswith("-"):
        raise ValidationError("docker_image_invalid", "Docker image must not start with '-'")
    docker_info = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if docker_info.returncode != 0:
        message = (docker_info.stderr or docker_info.stdout or "docker daemon is not reachable").strip()
        raise ValidationError("docker_unavailable", message)
    replay_output_dir = args.output_dir.resolve() / "replays"
    replay_output_dir.mkdir(parents=True, exist_ok=True)
    replay_output_dir.chmod(0o777)
    with tempfile.TemporaryDirectory(prefix="orbit-wars-kaggle-docker-") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        validator_path = temp_dir / "validate_submission.py"
        validator_path.write_text(IN_CONTAINER_VALIDATOR, encoding="utf-8")
        command = [
            "docker",
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
            "--timeout-seconds",
            str(args.timeout_seconds),
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
            raise ValidationError("docker_validation_failed", f"docker exited {completed.returncode}")


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


def _validate_tar_member(member: tarfile.TarInfo, root: Path, *, phase_error: type[Exception]) -> None:
    path = Path(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise phase_error("package_layout_failed", f"Unsafe archive member: {member.name}")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise phase_error("package_layout_failed", f"Unsafe archive member type: {member.name}")
    if member.size > 512 * 1024 * 1024:
        raise phase_error("package_layout_failed", f"Archive member too large: {member.name}")
    target = (root / path).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise phase_error("package_layout_failed", f"Archive member escapes extraction root: {member.name}")


def _require_dict(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValidationError("checkpoint_schema_failed", f"Checkpoint config missing {key}")
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
        return {key: _to_plain_data(item) for key, item in vars(value).items() if not key.startswith("_")}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CONFIG_TEMPLATE = '''from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EnvConfig:
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_fleets: int = 256
    player_count: int = 2
    max_ships: float = 400.0
    reward_capture_planet: float = 0.0
    reward_ship_delta: float = 0.0
    reward_production_delta: float = 0.0
    reward_terminal_scale: float = 1.0
    early_terminal_reward_shaping_enabled: bool = True
    early_terminal_reward_shaping_horizon: int = 500
    terminal_reward_mode: str = "binary_win"
    feature_history_steps: int = 1
    trajectory_shield_enabled: bool = True
    trajectory_shield_hit_mode: str = "selected_target"
    trajectory_shield_horizon: int = 500
    trajectory_shield_epsilon: float = 1e-6


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "gnn_pointer"
    hidden_size: int = 128
    attention_heads: int = 4
    max_moves_k: int = 3
    gnn_k_neighbors: int = 5
    gnn_message_passing_layers: int = 2
    normalize_observations: bool = True
    obs_norm_clip: float = 10.0


@dataclass(slots=True)
class PPOConfig:
    enable_gradient_checkpointing: bool = False


@dataclass(slots=True)
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def config_from_plain(data: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(**data.get("env", {})),
        model=ModelConfig(**data.get("model", {})),
        ppo=PPOConfig(**data.get("ppo", {})),
    )
'''


MAIN_TEMPLATE = r'''from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from src.config import config_from_plain
from src.constants import MAX_PLANETS
from src.features import FeatureHistoryBuffer, encode_turn
from src.feature_registry import candidate_feature_dim, global_feature_dim, self_feature_dim
from src.jax_policy import build_jax_policy
from src.trajectory_shield import is_trajectory_safe_for_launch, select_runtime_shielded_policy_actions


_ROOT = Path(__file__).resolve().parent
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
    params = artifact["params"]
    policy = build_jax_policy(cfg)
    _warm_policy(policy, params, cfg)
    history = FeatureHistoryBuffer(max_steps=int(cfg.env.feature_history_steps))
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    _STATE = {
        "cfg": cfg,
        "params": params,
        "policy": policy,
        "history": history,
        "manifest": manifest,
        "load_seconds": time.perf_counter() - started,
    }
    return _STATE


def _warm_policy(policy, params, cfg) -> None:
    self_features = jnp.zeros((MAX_PLANETS, self_feature_dim(cfg.env)), dtype=jnp.float32)
    candidate_features = jnp.zeros(
        (MAX_PLANETS, int(cfg.env.candidate_count), candidate_feature_dim(cfg.env)),
        dtype=jnp.float32,
    )
    global_features = jnp.zeros((MAX_PLANETS, global_feature_dim(cfg.env)), dtype=jnp.float32)
    candidate_mask = jnp.ones((MAX_PLANETS, int(cfg.env.candidate_count)), dtype=jnp.bool_)
    outputs = policy.apply(
        {"params": params},
        self_features,
        candidate_features,
        global_features,
        candidate_mask,
    )
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), outputs)


def agent(obs: Any) -> list[list[float | int]]:
    state = _load_state()
    cfg = state["cfg"]
    batch = encode_turn(obs, cfg.env, env_index=0, feature_history=state["history"])
    if batch.self_features.shape[0] == 0:
        return []
    sampled = select_runtime_shielded_policy_actions(
        jax.random.PRNGKey(0),
        state["policy"],
        {"params": state["params"]},
        batch,
        cfg.env,
        deterministic=True,
    )
    target_indices = jax.device_get(sampled.target_index)
    ship_buckets = jax.device_get(sampled.ship_bucket)

    moves: list[list[float | int]] = []
    for row_idx, context in enumerate(batch.contexts):
        remaining_ships = int(context.source_ships)
        for step_idx in range(target_indices.shape[1]):
            if len(moves) >= int(cfg.env.max_fleets):
                break
            target_idx = int(target_indices[row_idx, step_idx])
            bucket_idx = int(ship_buckets[row_idx, step_idx])
            if target_idx == 0 or bucket_idx == 0:
                continue
            if target_idx >= len(context.candidate_ids):
                continue
            if not bool(context.candidate_mask[target_idx]):
                continue
            ships = _ship_count_for_bucket(remaining_ships, bucket_idx, int(cfg.env.ship_bucket_count))
            if ships <= 0:
                continue
            target_id = int(context.candidate_ids[target_idx])
            angle = float(context.target_angles[target_idx])
            if not is_trajectory_safe_for_launch(
                batch.state,
                int(context.source_id),
                target_id,
                angle,
                int(ships),
                cfg.env,
            ):
                continue
            remaining_ships = max(0, remaining_ships - ships)
            moves.append([int(context.source_id), angle, int(ships)])
    state["history"].append(batch.state_snapshot if hasattr(batch, "state_snapshot") else _snapshot_from_batch(batch))
    return moves


def _ensure_sequence(values):
    if values.ndim == 2:
        return values[:, None, :]
    return values


def _ensure_ship_sequence(values):
    if values.ndim == 3:
        return values[:, None, :, :]
    return values


def _padded_policy_inputs(batch):
    row_count = int(batch.self_features.shape[0])
    if row_count > MAX_PLANETS:
        raise ValueError(f"Too many policy rows: {row_count} > {MAX_PLANETS}")
    pad_rows = MAX_PLANETS - row_count
    self_features = jnp.asarray(batch.self_features)
    candidate_features = jnp.asarray(batch.candidate_features)
    global_features = jnp.asarray(batch.global_features)
    candidate_mask = jnp.asarray(batch.candidate_mask).astype(jnp.bool_)
    if pad_rows <= 0:
        return self_features, candidate_features, global_features, candidate_mask
    self_features = jnp.pad(self_features, ((0, pad_rows), (0, 0)))
    candidate_features = jnp.pad(candidate_features, ((0, pad_rows), (0, 0), (0, 0)))
    global_features = jnp.pad(global_features, ((0, pad_rows), (0, 0)))
    candidate_mask = jnp.pad(candidate_mask, ((0, pad_rows), (0, 0)), constant_values=False)
    return self_features, candidate_features, global_features, candidate_mask


def _ship_count_for_bucket(available_ships: int, bucket: int, bucket_count: int) -> int:
    if available_ships <= 0 or bucket <= 0:
        return 0
    fraction = float(bucket) / float(max(bucket_count - 1, 1))
    ships = int(math.ceil(float(available_ships) * fraction))
    return min(available_ships, max(1, ships))


def _snapshot_from_batch(batch):
    from src.features import build_feature_snapshot

    return build_feature_snapshot(batch)
'''


IN_CONTAINER_VALIDATOR = r'''from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def main() -> int:
    args = parse_args()
    try:
        dependency_probe()
        with tempfile.TemporaryDirectory(prefix="submission-") as extract_text:
            extract_dir = Path(extract_text)
            extract_package(args.package, extract_dir)
            module, import_seconds = import_submission(extract_dir)
            artifact_load_seconds = artifact_load_probe(module)
            first_latency = first_action_probe(module, args.timeout_seconds, args.episode_steps)
            results = []
            player_counts = [2, 4] if args.player_count == "both" else [int(args.player_count)]
            for player_count in player_counts:
                modules = [import_submission(extract_dir, module_name=f"submission_main_p{idx}_{player_count}")[0] for idx in range(player_count)]
                results.append(run_episode(
                    modules,
                    player_count,
                    args.seed,
                    args.timeout_seconds,
                    args.episode_steps,
                    args.replay_output_dir,
                    args.checkpoint_update,
                ))
            print(json.dumps({
                "ok": True,
                "phase": "complete",
                "docker_image": "gcr.io/kaggle-images/python-simulations",
                "package": str(args.package),
                "import_seconds": import_seconds,
                "artifact_load_seconds": artifact_load_seconds,
                "first_action_seconds": first_latency,
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
    parser.add_argument("--timeout-seconds", type=float, default=1.0)
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


def artifact_load_probe(module) -> float:
    started = time.perf_counter()
    try:
        load_state = getattr(module, "_load_state")
        load_state()
    except Exception as exc:
        raise PhaseError("artifact_load_failed", str(exc)) from exc
    return time.perf_counter() - started


def first_action_probe(module, timeout_seconds: float, episode_steps: int) -> float:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 123, "episodeSteps": episode_steps}, debug=True)
    env.run([lambda obs: [] for _ in range(2)])
    obs = env.steps[0][0].observation
    started = time.perf_counter()
    try:
        action = module.agent(obs)
    except Exception as exc:
        raise PhaseError("first_action_failed", str(exc)) from exc
    elapsed = time.perf_counter() - started
    validate_action(action)
    if elapsed > timeout_seconds:
        raise PhaseError("timeout_failed", f"first action took {elapsed:.3f}s > {timeout_seconds:.3f}s")
    return elapsed


def run_episode(modules, player_count: int, seed: int, timeout_seconds: float, episode_steps: int, replay_output_dir, checkpoint_update: int):
    from kaggle_environments import make

    latencies = []

    def make_timed_agent(module):
        def timed_agent(obs):
            started = time.perf_counter()
            action = module.agent(obs)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            validate_action(action)
            if elapsed > timeout_seconds:
                raise PhaseError("timeout_failed", f"agent call took {elapsed:.3f}s > {timeout_seconds:.3f}s")
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
        "max_action_seconds": max(latencies) if latencies else 0.0,
        "agent_calls": len(latencies),
        "episode_steps": episode_steps,
        "replay_html_path": str(replay_html_path) if replay_html_path is not None else None,
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
'''


if __name__ == "__main__":
    raise SystemExit(main())