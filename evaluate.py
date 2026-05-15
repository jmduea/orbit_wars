from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TrainConfig, default_train_config_path, load_train_config
from src.env import extract_observation, extract_reward, extract_status
from src.features import candidate_feature_dim, global_feature_dim, self_feature_dim
from src.normalization import ObservationNormalizer
from src.opponents import KaggleRandomOpponent, SelfPlayOpponent, SniperOpponent
from src.policy import build_policy as create_policy


@dataclass(slots=True)
class GameResult:
    opponent: str
    game_index: int
    seed: int
    reward: float
    result: str
    length: int


@dataclass(slots=True)
class AggregateMetrics:
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_terminal_reward: float
    mean_game_length: float
    win_rate_se: float
    win_rate_ci95_low: float
    win_rate_ci95_high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Orbit Wars policy checkpoint against one or more opponents.")
    parser.add_argument("--config", type=str, default=str(default_train_config_path()), help="Training YAML config path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint to evaluate.")
    parser.add_argument("--games", type=int, default=20, help="Number of games to run against each opponent.")
    parser.add_argument(
        "--opponents",
        type=str,
        default="sniper",
        help="Comma-separated opponents to evaluate: sniper, random, self_play_snapshot.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds for each opponent, either comma-separated (0,1,2) or an inclusive range (0:99). Defaults to --seed..--seed+games-1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Starting seed when --seeds is omitted.")
    parser.add_argument("--device", type=str, default="auto", help="Device name, or auto to use the config value.")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy policy actions instead of sampling.")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval", help="Directory for JSON/CSV evaluation outputs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output filename prefix.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_opponents(value: str) -> list[str]:
    opponents = [part.strip() for part in value.split(",") if part.strip()]
    if not opponents:
        raise ValueError("At least one opponent is required.")
    supported = {"sniper", "random", "self_play_snapshot"}
    unknown = [name for name in opponents if name not in supported]
    if unknown:
        raise ValueError(f"Unknown opponent(s): {', '.join(unknown)}. Supported opponents: {', '.join(sorted(supported))}")
    return opponents


def parse_seed_spec(value: str | None, *, start_seed: int, games: int) -> list[int]:
    if games <= 0:
        raise ValueError("--games must be positive.")
    if value is None:
        return [start_seed + idx for idx in range(games)]
    value = value.strip()
    if ":" in value:
        start_text, end_text = value.split(":", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        seeds = list(range(start, end + step, step))
    else:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds did not contain any seeds.")
    if len(seeds) < games:
        raise ValueError(f"--seeds provided {len(seeds)} seed(s), but --games requires {games}.")
    return seeds[:games]


def build_policy(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    return create_policy(
        architecture=cfg.model.architecture,
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
    ).to(device)


def register_checkpoint_module_aliases() -> None:
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.rl_template", types.ModuleType("src.rl_template"))
    module_candidates = {
        "config": ["src.rl_template.config", "src.config", "config"],
        "features": ["src.rl_template.features", "src.features", "features"],
        "policy": ["src.rl_template.policy", "src.policy", "policy"],
        "ppo": ["src.rl_template.ppo", "src.ppo", "ppo"],
        "game_types": ["src.rl_template.game_types", "src.game_types", "game_types"],
        "opponents": ["src.rl_template.opponents", "src.opponents", "opponents"],
        "env": ["src.rl_template.env", "src.env", "env"],
        "train": ["src.rl_template.train", "src.train", "train"],
        "normalization": ["src.rl_template.normalization", "src.normalization", "normalization"],
    }

    for canonical_name, candidates in module_candidates.items():
        module = None
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
                break
            except ModuleNotFoundError:
                continue
        if module is None:
            continue
        sys.modules[f"src.rl_template.{canonical_name}"] = module
        sys.modules[f"src.{canonical_name}"] = module


def load_checkpoint_if_available(
    policy: torch.nn.Module,
    normalizer: ObservationNormalizer | None,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    register_checkpoint_module_aliases()
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    policy.load_state_dict(state_dict)
    normalizer_state = checkpoint.get("normalizer") if isinstance(checkpoint, dict) else None
    if normalizer is not None and normalizer_state is not None:
        normalizer.load_state_dict(normalizer_state)


def build_opponent(
    name: str,
    cfg: TrainConfig,
    policy: torch.nn.Module,
    normalizer: ObservationNormalizer | None,
    device: torch.device,
    deterministic: bool,
) -> Any:
    if name == "sniper":
        return SniperOpponent()
    if name == "random":
        return KaggleRandomOpponent()
    if name == "self_play_snapshot":
        opponent = SelfPlayOpponent(cfg, device=device, deterministic=deterministic)
        opponent.sync_from(policy, normalizer)
        return opponent
    raise ValueError(f"Unknown opponent: {name}")


def reward_to_label(reward: float) -> str:
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def play_one_game(
    learner: SelfPlayOpponent,
    opponent: Any,
    *,
    seed: int,
) -> tuple[float, int]:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    player_obs = extract_observation(states[0])
    opponent_obs = extract_observation(states[1])
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

    return extract_reward(states[0]), step_count


def aggregate(results: Iterable[GameResult]) -> AggregateMetrics:
    items = list(results)
    games = len(items)
    if games == 0:
        return AggregateMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = sum(1 for result in items if result.result == "win")
    draws = sum(1 for result in items if result.result == "draw")
    losses = sum(1 for result in items if result.result == "loss")
    win_rate = wins / games
    draw_rate = draws / games
    loss_rate = losses / games
    win_rate_se = math.sqrt(win_rate * (1.0 - win_rate) / games) if games > 0 else 0.0
    z = 1.96
    ci_low = max(0.0, win_rate - z * win_rate_se)
    ci_high = min(1.0, win_rate + z * win_rate_se)
    return AggregateMetrics(
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate=win_rate,
        draw_rate=draw_rate,
        loss_rate=loss_rate,
        mean_terminal_reward=float(np.mean([result.reward for result in items])),
        mean_game_length=float(np.mean([result.length for result in items])),
        win_rate_se=win_rate_se,
        win_rate_ci95_low=ci_low,
        win_rate_ci95_high=ci_high,
    )


def write_outputs(output_dir: Path, run_name: str, payload: dict[str, Any], results: list[GameResult]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["opponent", "game_index", "seed", "result", "reward", "length"])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    device_name = cfg.device if args.device == "auto" else args.device
    device = resolve_device(device_name)
    opponents = parse_opponents(args.opponents)
    seeds = parse_seed_spec(args.seeds, start_seed=args.seed, games=args.games)
    seed_everything(seeds[0])

    policy = build_policy(cfg, device)
    normalizer = ObservationNormalizer(clip=cfg.model.obs_norm_clip) if cfg.model.normalize_observations else None
    load_checkpoint_if_available(policy, normalizer, args.checkpoint, device)
    policy.eval()

    learner = SelfPlayOpponent(cfg, device=device, deterministic=args.deterministic)
    learner.sync_from(policy, normalizer)

    all_results: list[GameResult] = []
    for opponent_name in opponents:
        opponent = build_opponent(opponent_name, cfg, policy, normalizer, device, args.deterministic)
        opponent_results: list[GameResult] = []
        for game_idx, game_seed in enumerate(seeds):
            reward, length = play_one_game(learner, opponent, seed=game_seed)
            result = GameResult(
                opponent=opponent_name,
                game_index=game_idx + 1,
                seed=game_seed,
                reward=reward,
                result=reward_to_label(reward),
                length=length,
            )
            opponent_results.append(result)
            all_results.append(result)
            print(
                f"opponent={opponent_name} game={game_idx + 1}/{args.games} seed={game_seed} "
                f"result={result.result} reward={reward:.1f} length={length}"
            )
        metrics = aggregate(opponent_results)
        print(
            f"opponent={opponent_name} win_rate={metrics.win_rate:.4f} "
            f"draw_rate={metrics.draw_rate:.4f} loss_rate={metrics.loss_rate:.4f} "
            f"mean_reward={metrics.mean_terminal_reward:.4f} mean_length={metrics.mean_game_length:.2f} "
            f"win_rate_se={metrics.win_rate_se:.4f} ci95=[{metrics.win_rate_ci95_low:.4f},{metrics.win_rate_ci95_high:.4f}]"
        )

    overall = aggregate(all_results)
    by_opponent = {name: asdict(aggregate(result for result in all_results if result.opponent == name)) for name in opponents}
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    safe_checkpoint = Path(args.checkpoint).stem if args.checkpoint else "uncheckpointed"
    run_name = args.run_name or f"eval_{safe_checkpoint}_{timestamp}"
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "deterministic": bool(args.deterministic),
        "device": str(device),
        "games_per_opponent": args.games,
        "opponents": opponents,
        "seeds": seeds,
        "overall": asdict(overall),
        "by_opponent": by_opponent,
        "games": [asdict(result) for result in all_results],
    }
    json_path, csv_path = write_outputs(Path(args.output_dir), run_name, payload, all_results)
    print(
        f"overall win_rate={overall.win_rate:.4f} draw_rate={overall.draw_rate:.4f} "
        f"loss_rate={overall.loss_rate:.4f} mean_reward={overall.mean_terminal_reward:.4f} "
        f"mean_length={overall.mean_game_length:.2f} win_rate_se={overall.win_rate_se:.4f} "
        f"ci95=[{overall.win_rate_ci95_low:.4f},{overall.win_rate_ci95_high:.4f}]"
    )
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")


if __name__ == "__main__":
    main()
