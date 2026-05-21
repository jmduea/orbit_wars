from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.checkpoint_compat import validate_checkpoint_feature_compatibility  # noqa: E402
from src.config import TrainConfig, default_train_config_path, load_train_config  # noqa: E402
from src.normalization import ObservationNormalizer  # noqa: E402
from src.opponents import SelfPlayOpponent, SniperOpponent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one rl_template match against sniper and save replay HTML.")
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="artifacts/rl_template/replays/vs_sniper.html")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint_params(
    normalizer: ObservationNormalizer | None, checkpoint_path: str | None, cfg: TrainConfig
) -> dict[str, Any]:
    if checkpoint_path is None:
        raise ValueError("--checkpoint is required for JAX inference.")
    with Path(checkpoint_path).open("rb") as file:
        checkpoint = pickle.load(file)
    validate_checkpoint_feature_compatibility(checkpoint, cfg.env, checkpoint_path=checkpoint_path)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(f"JAX checkpoint must contain params: {checkpoint_path}")
    normalizer_state = checkpoint.get("normalizer")
    if normalizer is not None and normalizer_state is not None:
        normalizer.load_state_dict(normalizer_state)
    return checkpoint["params"]


def extract_observation(state: Any) -> Any:
    return state.get("observation") if isinstance(state, dict) else getattr(state, "observation")


def extract_status(state: Any) -> str:
    return str(state.get("status", "UNKNOWN")) if isinstance(state, dict) else str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    value = state.get("reward", 0.0) if isinstance(state, dict) else getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def run_match(learner: SelfPlayOpponent, *, seed: int) -> tuple[str, float, int]:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    player_obs = extract_observation(states[0])
    opponent_obs = extract_observation(states[1])
    opponent = SniperOpponent()
    done = extract_status(states[0]) != "ACTIVE"
    step_count = 0
    while not done:
        player_action = learner.act(player_obs)
        opponent_action = opponent.act(opponent_obs)
        states = env.step([player_action, opponent_action])
        player_obs = extract_observation(states[0])
        opponent_obs = extract_observation(states[1])
        done = extract_status(states[0]) != "ACTIVE"
        step_count += 1
    return env.render(mode="html"), extract_reward(states[0]), step_count


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(args.seed)
    normalizer = ObservationNormalizer(clip=cfg.model.obs_norm_clip, env_cfg=cfg.env) if cfg.model.normalize_observations else None
    params = load_checkpoint_params(normalizer, args.checkpoint, cfg)
    learner = SelfPlayOpponent(cfg, device=args.device, deterministic=args.deterministic)
    learner.sync_from(params, normalizer)
    html, reward, step_count = run_match(learner, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"saved_html={output_path}")
    print(f"reward={reward:.1f}")
    print(f"steps={step_count}")


if __name__ == "__main__":
    main()
