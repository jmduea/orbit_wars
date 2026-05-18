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

from src.config import TrainConfig, default_train_config_path, load_train_config  # noqa: E402
from src.env import extract_observation, extract_reward, extract_status  # noqa: E402
from src.features import candidate_feature_dim, global_feature_dim, self_feature_dim  # noqa: E402
from src.normalization import ObservationNormalizer  # noqa: E402
from src.opponents import KaggleRandomOpponent, SelfPlayOpponent, SniperOpponent  # noqa: E402
from src.policy import build_policy as create_policy  # noqa: E402


@dataclass(slots=True)
class GameResult:
    format: str
    player_count: int
    opponent: str
    game_index: int
    seed: int
    learner_seat: int
    opponent_slots: int
    reward: float
    result: str
    placement: float
    first_place: bool
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


@dataclass(slots=True)
class FormatMetrics:
    games: int
    player_count: int
    win_rate_2p: float | None
    first_place_rate_4p: float | None
    average_placement_4p: float | None
    mean_terminal_reward: float
    mean_game_length: float
    per_seat: dict[str, dict[str, float | int | None]]


@dataclass(slots=True)
class GameOutcome:
    reward: float
    result: str
    placement: float
    first_place: bool
    length: int


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
        "--player-counts",
        "--formats",
        dest="formats",
        type=str,
        default="2p",
        help="Comma-separated game formats to evaluate, such as 2,4 or 2p,4p.",
    )
    parser.add_argument(
        "--learner-seats",
        type=str,
        default="0",
        help="Comma-separated learner seats to evaluate, or 'all' to rotate through every seat in each format.",
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


def parse_formats(value: str) -> list[int]:
    player_counts: list[int] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.endswith("p"):
            token = token[:-1]
        try:
            player_count = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid format {part!r}; use values like 2,4 or 2p,4p.") from exc
        if player_count not in {2, 4}:
            raise ValueError(f"Unsupported player count {player_count}; supported formats are 2p and 4p.")
        if player_count not in player_counts:
            player_counts.append(player_count)
    if not player_counts:
        raise ValueError("At least one format is required.")
    return player_counts


def format_label(player_count: int) -> str:
    return f"{player_count}p"


def parse_learner_seats(value: str, player_count: int) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(player_count))
    seats = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seats:
        raise ValueError("--learner-seats must be 'all' or contain at least one seat index.")
    invalid = [seat for seat in seats if seat < 0 or seat >= player_count]
    if invalid:
        raise ValueError(
            f"Invalid learner seat(s) for {format_label(player_count)}: {', '.join(map(str, invalid))}. "
            f"Valid seats are 0..{player_count - 1}."
        )
    deduped: list[int] = []
    for seat in seats:
        if seat not in deduped:
            deduped.append(seat)
    return deduped


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


def placement_from_rewards(rewards: list[float], learner_seat: int) -> tuple[float, bool]:
    learner_reward = rewards[learner_seat]
    best_reward = max(rewards) if rewards else 0.0
    rank = 1.0 + sum(reward > learner_reward for reward in rewards)
    ties = sum(reward == learner_reward for reward in rewards)
    placement = rank + (float(ties) - 1.0) * 0.5
    first_place = learner_reward == best_reward and learner_reward > 0.0
    return float(placement), bool(first_place)


def play_one_game(
    learner: SelfPlayOpponent,
    opponents: list[Any],
    *,
    seed: int,
    player_count: int,
    learner_seat: int,
) -> GameOutcome:
    if player_count < 2:
        raise ValueError("player_count must be at least 2.")
    if learner_seat < 0 or learner_seat >= player_count:
        raise ValueError(f"learner_seat must be in 0..{player_count - 1}; got {learner_seat}.")
    if len(opponents) != player_count - 1:
        raise ValueError(
            f"Expected {player_count - 1} opponent slot(s) for {format_label(player_count)}, got {len(opponents)}."
        )

    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
    env.reset(num_agents=player_count)
    states = env.step([[] for _ in range(player_count)])
    observations = [extract_observation(state) for state in states]
    done = extract_status(states[learner_seat]) != "ACTIVE"
    step_count = 0

    while not done:
        joint_actions: list[list[list[float | int]]] = [[] for _ in range(player_count)]
        joint_actions[learner_seat] = learner.act(observations[learner_seat])
        opponent_idx = 0
        for seat in range(player_count):
            if seat == learner_seat:
                continue
            joint_actions[seat] = opponents[opponent_idx].act(observations[seat])
            opponent_idx += 1
        states = env.step(joint_actions)
        observations = [extract_observation(state) for state in states]
        done = extract_status(states[learner_seat]) != "ACTIVE"
        step_count += 1

    rewards = [extract_reward(state) for state in states]
    placement, first_place = placement_from_rewards(rewards, learner_seat)
    reward = rewards[learner_seat]
    return GameOutcome(
        reward=reward,
        result=reward_to_label(reward),
        placement=placement,
        first_place=first_place,
        length=step_count,
    )


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


def aggregate_format(results: Iterable[GameResult], player_count: int) -> FormatMetrics:
    items = list(results)
    games = len(items)
    if games == 0:
        return FormatMetrics(
            games=0,
            player_count=player_count,
            win_rate_2p=None,
            first_place_rate_4p=None,
            average_placement_4p=None,
            mean_terminal_reward=0.0,
            mean_game_length=0.0,
            per_seat={},
        )

    def _seat_metrics(seat_items: list[GameResult]) -> dict[str, float | int | None]:
        if not seat_items:
            return {
                "games": 0,
                "win_rate_2p": None,
                "first_place_rate_4p": None,
                "average_placement_4p": None,
                "mean_terminal_reward": 0.0,
                "mean_game_length": 0.0,
            }
        return {
            "games": len(seat_items),
            "win_rate_2p": (
                sum(1 for result in seat_items if result.result == "win") / len(seat_items)
                if player_count == 2
                else None
            ),
            "first_place_rate_4p": (
                sum(1 for result in seat_items if result.first_place) / len(seat_items)
                if player_count == 4
                else None
            ),
            "average_placement_4p": (
                float(np.mean([result.placement for result in seat_items])) if player_count == 4 else None
            ),
            "mean_terminal_reward": float(np.mean([result.reward for result in seat_items])),
            "mean_game_length": float(np.mean([result.length for result in seat_items])),
        }

    per_seat = {
        str(seat): _seat_metrics([result for result in items if result.learner_seat == seat])
        for seat in sorted({result.learner_seat for result in items})
    }
    return FormatMetrics(
        games=games,
        player_count=player_count,
        win_rate_2p=sum(1 for result in items if result.result == "win") / games if player_count == 2 else None,
        first_place_rate_4p=sum(1 for result in items if result.first_place) / games if player_count == 4 else None,
        average_placement_4p=float(np.mean([result.placement for result in items])) if player_count == 4 else None,
        mean_terminal_reward=float(np.mean([result.reward for result in items])),
        mean_game_length=float(np.mean([result.length for result in items])),
        per_seat=per_seat,
    )


def write_outputs(output_dir: Path, run_name: str, payload: dict[str, Any], results: list[GameResult]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "format",
                "player_count",
                "opponent",
                "game_index",
                "seed",
                "learner_seat",
                "opponent_slots",
                "result",
                "reward",
                "placement",
                "first_place",
                "length",
            ],
        )
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
    player_counts = parse_formats(args.formats)
    learner_seats_by_format = {
        player_count: parse_learner_seats(args.learner_seats, player_count) for player_count in player_counts
    }
    seeds = parse_seed_spec(args.seeds, start_seed=args.seed, games=args.games)
    seed_everything(seeds[0])

    policy = build_policy(cfg, device)
    normalizer = ObservationNormalizer(clip=cfg.model.obs_norm_clip) if cfg.model.normalize_observations else None
    load_checkpoint_if_available(policy, normalizer, args.checkpoint, device)
    policy.eval()

    learner = SelfPlayOpponent(cfg, device=device, deterministic=args.deterministic)
    learner.sync_from(policy, normalizer)

    all_results: list[GameResult] = []
    for player_count in player_counts:
        label = format_label(player_count)
        seats = learner_seats_by_format[player_count]
        for opponent_name in opponents:
            format_opponent_results: list[GameResult] = []
            for seat in seats:
                for game_idx, game_seed in enumerate(seeds):
                    opponent_slots = [
                        build_opponent(opponent_name, cfg, policy, normalizer, device, args.deterministic)
                        for _ in range(player_count - 1)
                    ]
                    outcome = play_one_game(
                        learner,
                        opponent_slots,
                        seed=game_seed,
                        player_count=player_count,
                        learner_seat=seat,
                    )
                    result = GameResult(
                        format=label,
                        player_count=player_count,
                        opponent=opponent_name,
                        game_index=game_idx + 1,
                        seed=game_seed,
                        learner_seat=seat,
                        opponent_slots=len(opponent_slots),
                        reward=outcome.reward,
                        result=outcome.result,
                        placement=outcome.placement,
                        first_place=outcome.first_place,
                        length=outcome.length,
                    )
                    format_opponent_results.append(result)
                    all_results.append(result)
                    print(
                        f"format={label} opponent={opponent_name} seat={seat} "
                        f"game={game_idx + 1}/{args.games} seed={game_seed} result={result.result} "
                        f"reward={outcome.reward:.1f} placement={outcome.placement:.1f} "
                        f"first_place={int(outcome.first_place)} length={outcome.length}"
                    )
            format_metrics = aggregate_format(format_opponent_results, player_count)
            if player_count == 2:
                print(
                    f"format={label} opponent={opponent_name} win_rate_2p={format_metrics.win_rate_2p:.4f} "
                    f"mean_reward={format_metrics.mean_terminal_reward:.4f} "
                    f"mean_length={format_metrics.mean_game_length:.2f} per_seat={format_metrics.per_seat}"
                )
            else:
                print(
                    f"format={label} opponent={opponent_name} "
                    f"first_place_rate_4p={format_metrics.first_place_rate_4p:.4f} "
                    f"average_placement_4p={format_metrics.average_placement_4p:.4f} "
                    f"mean_reward={format_metrics.mean_terminal_reward:.4f} "
                    f"mean_length={format_metrics.mean_game_length:.2f} per_seat={format_metrics.per_seat}"
                )

    overall = aggregate(all_results)
    by_opponent = {name: asdict(aggregate(result for result in all_results if result.opponent == name)) for name in opponents}
    by_format = {
        format_label(player_count): asdict(
            aggregate_format((result for result in all_results if result.player_count == player_count), player_count)
        )
        for player_count in player_counts
    }
    by_format_opponent = {
        format_label(player_count): {
            name: asdict(
                aggregate_format(
                    (
                        result
                        for result in all_results
                        if result.player_count == player_count and result.opponent == name
                    ),
                    player_count,
                )
            )
            for name in opponents
        }
        for player_count in player_counts
    }
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
        "formats": [format_label(player_count) for player_count in player_counts],
        "learner_seats": {format_label(key): value for key, value in learner_seats_by_format.items()},
        "seeds": seeds,
        "overall": asdict(overall),
        "by_opponent": by_opponent,
        "by_format": by_format,
        "by_format_opponent": by_format_opponent,
        "games": [asdict(result) for result in all_results],
    }
    json_path, csv_path = write_outputs(Path(args.output_dir), run_name, payload, all_results)
    print(
        f"overall win_rate={overall.win_rate:.4f} draw_rate={overall.draw_rate:.4f} "
        f"loss_rate={overall.loss_rate:.4f} mean_reward={overall.mean_terminal_reward:.4f} "
        f"mean_length={overall.mean_game_length:.2f} win_rate_se={overall.win_rate_se:.4f} "
        f"ci95=[{overall.win_rate_ci95_low:.4f},{overall.win_rate_ci95_high:.4f}]"
    )
    print(f"by_format={by_format}")
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")


if __name__ == "__main__":
    main()
