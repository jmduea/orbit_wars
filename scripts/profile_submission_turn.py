"""Profile submission encode vs shielded-policy latency on a live Kaggle obs."""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_kaggle_docker_submission import export_runtime_artifact
from src.config import config_from_plain
from src.features import FeatureExtractor
from src.jax.features import empty_feature_history
from src.jax.policy import build_jax_policy
from src.jax.submission_runtime import (
    apply_feature_metadata_to_model_config,
    compile_batched_feature_encode,
    compile_feature_history_append,
    compile_shielded_policy_act,
    jax_game_from_observation_fast,
)


def _load_obs(episode_steps: int):
    from kaggle_environments import make

    env = make(
        "orbit_wars",
        configuration={"seed": 123, "episodeSteps": episode_steps},
        debug=True,
    )
    env.run([lambda _obs: [] for _ in range(2)])
    return env.steps[0][0].observation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500)
    args = parser.parse_args()

    artifact = export_runtime_artifact(args.checkpoint.resolve())
    cfg = config_from_plain(artifact["config"])
    cfg = apply_feature_metadata_to_model_config(cfg, artifact.get("feature_metadata"))
    params = artifact["params"]
    policy = build_jax_policy(cfg)
    variables = {"params": params}
    obs = _load_obs(args.episode_steps)

    parse_game = partial(
        jax_game_from_observation_fast, max_fleet_slots=int(cfg.task.max_fleets)
    )
    jitted_encode = compile_batched_feature_encode(cfg.task)
    jitted_append = compile_feature_history_append(cfg.task)
    jitted_act = compile_shielded_policy_act(
        policy, variables, cfg, deterministic=True, deterministic_eval=True
    )

    extractor = FeatureExtractor(cfg.task)
    history = empty_feature_history(cfg.task)
    legacy_history = empty_feature_history(cfg.task)

    for _ in range(args.warmup):
        game = parse_game(obs)
        game_batched, batch_batched = jitted_encode(game, history)
        action = jitted_act(game_batched, batch_batched, jax.random.PRNGKey(0))
        jax.block_until_ready(action.valid)
        history = jitted_append(history, game)

    legacy_times: list[float] = []
    jitted_times: list[float] = []
    for _ in range(args.reps):
        started = time.perf_counter()
        extracted = extractor.extract(
            obs, history=legacy_history, max_fleet_slots=int(cfg.task.max_fleets)
        )
        legacy_history = extractor.append_history(
            legacy_history,
            obs,
            max_fleet_slots=int(cfg.task.max_fleets),
        )
        legacy_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        game = parse_game(obs)
        game_batched, batch_batched = jitted_encode(game, history)
        action = jitted_act(game_batched, batch_batched, jax.random.PRNGKey(0))
        jax.block_until_ready(action.valid)
        history = jitted_append(history, game)
        jitted_times.append(time.perf_counter() - started)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print(
        {
            "checkpoint": str(args.checkpoint),
            "architecture": cfg.model.architecture,
            "legacy_extract_append_mean_s": round(_mean(legacy_times), 4),
            "jitted_encode_act_append_mean_s": round(_mean(jitted_times), 4),
            "note": "jitted path matches submission main.py after warmup",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
