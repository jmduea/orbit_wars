from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.constants import MAX_STEPS

from .config import EnvConfig, TrainConfig
from .features import (
    FeatureHistoryBuffer,
    TurnBatch,
    build_feature_snapshot,
    encode_turn,
)
from .game_types import GameState, PlanetState, parse_observation
from .opponents import OpponentPolicy
from .trajectory_shield import filter_moves_with_trajectory_shield


@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


class OrbitWarsEnv:
    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.active_opponent = opponent
        self.active_opponents: dict[int, OpponentPolicy] = {}
        self.active_opponent_metadata: dict[int, dict[str, Any]] = {}
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.last_opponent_obs: dict[int, Any] = {}
        self.previous_player_state: GameState | None = None
        self.previous_opp_state: GameState | None = None
        self.previous_opponent_states: dict[int, GameState] = {}
        self.episode_index = 0
        self.learner_player = 0
        self.feature_history = FeatureHistoryBuffer(
            max(0, self.cfg.env.feature_history_steps - 1)
        )

    def reset(self, seed: int | None = None) -> TurnBatch:
        make_fn = self.make_fn or default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)
        player_count = self.cfg.env.player_count
        if player_count < 1:
            raise ValueError("cfg.env.player_count must be at least 1")
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % player_count
        else:
            self.learner_player = 0
        self.feature_history = FeatureHistoryBuffer(
            max(0, self.cfg.env.feature_history_steps - 1)
        )
        opponent_players = [
            player for player in range(player_count) if player != self.learner_player
        ]
        sampled = self._sample_opponents(len(opponent_players))
        self.active_opponents = {
            player: selection[0]
            for player, selection in zip(opponent_players, sampled, strict=True)
        }
        self.active_opponent_metadata = {
            player: selection[1]
            for player, selection in zip(opponent_players, sampled, strict=True)
        }
        self.active_opponent = (
            self.active_opponents[opponent_players[0]]
            if opponent_players
            else self.opponent
        )
        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=player_count)
        states = self.env.step([[] for _ in range(player_count)])
        learner_state = states[self.learner_player]
        self.last_obs = extract_observation(learner_state)
        self.last_opponent_obs = {
            player: extract_observation(states[player]) for player in opponent_players
        }
        self.last_opp_obs = (
            self.last_opponent_obs.get(opponent_players[0])
            if opponent_players
            else None
        )
        self.previous_player_state = parse_observation(self.last_obs)
        self.previous_opponent_states = {
            player: parse_observation(observation)
            for player, observation in self.last_opponent_obs.items()
        }
        self.previous_opp_state = (
            self.previous_opponent_states.get(opponent_players[0])
            if opponent_players
            else None
        )
        self.episode_index += 1
        batch = encode_turn(
            self.previous_player_state,
            self.cfg.env,
            env_index=self.env_index,
            feature_history=self.feature_history,
        )
        self.feature_history.append(build_feature_snapshot(batch))
        return batch

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        player_count = self.cfg.env.player_count
        joint_action: list[list[list[float | int]]] = [[] for _ in range(player_count)]
        learner_state = self.previous_player_state or parse_observation(self.last_obs)
        joint_action[self.learner_player] = filter_moves_with_trajectory_shield(
            player_action,
            learner_state,
            self.cfg.env,
        )
        for player, opponent in self.active_opponents.items():
            opponent_state = self.previous_opponent_states.get(player) or parse_observation(
                self.last_opponent_obs[player]
            )
            joint_action[player] = filter_moves_with_trajectory_shield(
                opponent.act(self.last_opponent_obs[player]),
                opponent_state,
                self.cfg.env,
            )

        states = self.env.step(joint_action)
        player_state = states[self.learner_player]
        opponent_players = [
            player for player in range(player_count) if player != self.learner_player
        ]
        opponent_states = {player: states[player] for player in opponent_players}
        next_obs = extract_observation(player_state)
        next_opponent_obs = {
            player: extract_observation(state)
            for player, state in opponent_states.items()
        }
        next_player_state = parse_observation(next_obs)
        next_opponent_states = {
            player: parse_observation(observation)
            for player, observation in next_opponent_obs.items()
        }
        done = extract_status(player_state) != "ACTIVE"

        terminal_diagnostics = terminal_reward_diagnostics(
            next_player_state, self.cfg.env
        )
        terminal_component = (
            apply_early_terminal_reward_shaping(
                terminal_diagnostics["terminal_reward_unscaled"],
                next_player_state.step,
                self.cfg.env,
            )
            * self.cfg.env.reward_terminal_scale
            if done
            else 0.0
        )
        shaping_components = shaped_reward_components(
            self.previous_player_state, next_player_state, self.cfg.env
        )
        shaping_component = sum(shaping_components.values())
        reward = terminal_component + shaping_component

        self.last_obs = next_obs
        self.last_opponent_obs = next_opponent_obs
        self.last_opp_obs = (
            next_opponent_obs.get(opponent_players[0]) if opponent_players else None
        )
        self.previous_player_state = next_player_state
        self.previous_opponent_states = next_opponent_states
        self.previous_opp_state = (
            next_opponent_states.get(opponent_players[0]) if opponent_players else None
        )

        batch = encode_turn(
            next_player_state,
            self.cfg.env,
            env_index=self.env_index,
            feature_history=self.feature_history,
        )
        self.feature_history.append(build_feature_snapshot(batch))
        opponent_statuses = {
            player: extract_status(state) for player, state in opponent_states.items()
        }
        info = {
            "learner_player": self.learner_player,
            "opponent_players": opponent_players,
            "player_status": extract_status(player_state),
            "opponent_status": (
                opponent_statuses.get(opponent_players[0]) if opponent_players else None
            ),
            "opponent_statuses": opponent_statuses,
            "opponent_composition": self.active_opponent_metadata,
            "reward": reward,
            "terminal_reward": terminal_component,
            "terminal_rank": terminal_diagnostics["terminal_rank"] if done else 0.0,
            "terminal_placement": (
                terminal_diagnostics["terminal_placement"] if done else 0.0
            ),
            "terminal_is_first": (
                terminal_diagnostics["terminal_is_first"] if done else 0.0
            ),
            "terminal_score_share": (
                terminal_diagnostics["terminal_score_share"] if done else 0.0
            ),
            "terminal_survival_time": (
                terminal_diagnostics["terminal_survival_time"] if done else 0.0
            ),
            "shaping_reward": shaping_component,
            **shaping_components,
        }
        return StepResult(batch=batch, reward=reward, done=done, info=info)

    def _sample_opponents(
        self, count: int
    ) -> list[tuple[OpponentPolicy, dict[str, Any]]]:
        sampler = getattr(self.opponent, "sample_opponents", None)
        if callable(sampler):
            selections = sampler(count)
            return [(selection.policy, selection.metadata) for selection in selections]
        single_sampler = getattr(self.opponent, "sample_opponent", None)
        if callable(single_sampler):
            return [
                (
                    single_sampler(),
                    {"snapshot_id": -1, "update": 0, "source": self.cfg.opponent},
                )
                for _ in range(count)
            ]
        return [
            (
                self.opponent,
                {"snapshot_id": -1, "update": 0, "source": self.cfg.opponent},
            )
            for _ in range(count)
        ]

    def _sample_opponent(self) -> OpponentPolicy:
        return self._sample_opponents(1)[0][0]


def shaped_reward_components(
    previous_state: GameState | None,
    current_state: GameState,
    env_cfg: EnvConfig,
) -> dict[str, float]:
    components = {
        "reward_capture_planet": 0.0,
        "reward_ship_delta": 0.0,
        "reward_production_delta": 0.0,
    }
    if previous_state is None:
        return components

    player = current_state.player
    previous_planets = planets_by_id(previous_state.planets)
    current_planets = planets_by_id(current_state.planets)

    captured = 0
    lost = 0
    for planet_id, current_planet in current_planets.items():
        previous_planet = previous_planets.get(planet_id)
        if previous_planet is None:
            continue
        if previous_planet.owner != player and current_planet.owner == player:
            captured += 1
        elif previous_planet.owner == player and current_planet.owner != player:
            lost += 1

    components["reward_capture_planet"] = env_cfg.reward_capture_planet * float(
        captured - lost
    )

    previous_ship_advantage = ship_advantage(previous_state, player)
    current_ship_advantage = ship_advantage(current_state, player)
    components["reward_ship_delta"] = env_cfg.reward_ship_delta * (
        current_ship_advantage - previous_ship_advantage
    )

    previous_production = controlled_production(previous_state.planets, player)
    current_production = controlled_production(current_state.planets, player)
    components["reward_production_delta"] = env_cfg.reward_production_delta * (
        current_production - previous_production
    )
    return components


def planets_by_id(planets: list[PlanetState]) -> dict[int, PlanetState]:
    return {planet.id: planet for planet in planets}


def ship_advantage(state: GameState, player: int) -> float:
    player_ships = sum(
        planet.ships for planet in state.planets if planet.owner == player
    )
    player_ships += sum(fleet.ships for fleet in state.fleets if fleet.owner == player)
    opponent_ships = sum(
        planet.ships for planet in state.planets if planet.owner not in {-1, player}
    )
    opponent_ships += sum(
        fleet.ships for fleet in state.fleets if fleet.owner not in {-1, player}
    )
    return float(player_ships - opponent_ships)


def controlled_production(planets: list[PlanetState], player: int) -> float:
    return float(sum(planet.production for planet in planets if planet.owner == player))


def default_make_fn() -> Any:
    from kaggle_environments import make

    return make


def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def terminal_reward(
    player_state: Any, opponent_states: Any, env_cfg: EnvConfig | None = None
) -> float:
    player_reward = extract_reward(player_state)
    if env_cfg is None:
        if not isinstance(opponent_states, (list, tuple)):
            opponent_states = [opponent_states]
        opponent_rewards = [extract_reward(opp_state) for opp_state in opponent_states]
        if player_reward > 0.0 and any(
            opponent_reward > 0.0 for opponent_reward in opponent_rewards
        ):
            return 0.0
        return player_reward
    player_count = int(getattr(env_cfg, "player_count", 2))
    rewards = [player_reward]
    if not isinstance(opponent_states, (list, tuple)):
        opponent_states = [opponent_states]
    rewards.extend(extract_reward(opp_state) for opp_state in opponent_states)
    return terminal_reward_from_scores(rewards[:player_count], env_cfg)[
        "terminal_reward_unscaled"
    ]


def terminal_reward_diagnostics(
    state: GameState, env_cfg: EnvConfig
) -> dict[str, float]:
    player_count = int(getattr(env_cfg, "player_count", 2))
    scores = []
    for player in range(player_count):
        score = sum(planet.ships for planet in state.planets if planet.owner == player)
        score += sum(fleet.ships for fleet in state.fleets if fleet.owner == player)
        scores.append(float(score))
    diagnostics = terminal_reward_from_scores(
        scores, env_cfg, learner_index=state.player
    )
    diagnostics["terminal_survival_time"] = min(
        float(state.step + 1), float(MAX_STEPS)
    ) / max(float(MAX_STEPS), 1.0)
    if getattr(env_cfg, "terminal_reward_mode", "binary_win").strip().lower() == (
        "survival_plus_rank"
    ):
        diagnostics["terminal_reward_unscaled"] = (
            0.5 * diagnostics["terminal_ranked_reward"]
            + 0.5 * diagnostics["terminal_survival_time"]
        )
    return diagnostics


def terminal_reward_from_scores(
    scores: list[float], env_cfg: EnvConfig, learner_index: int = 0
) -> dict[str, float]:
    player_count = max(int(getattr(env_cfg, "player_count", len(scores))), 1)
    padded_scores = [0.0 for _ in range(player_count)]
    for index, score in enumerate(scores[:player_count]):
        padded_scores[index] = float(score)
    learner_index = min(max(int(learner_index), 0), player_count - 1)
    learner_score = padded_scores[learner_index]
    best_score = max(padded_scores) if padded_scores else 0.0
    rank = 1.0 + sum(score > learner_score for score in padded_scores)
    ties = sum(score == learner_score for score in padded_scores)
    placement = rank + (float(ties) - 1.0) * 0.5
    is_first = 1.0 if learner_score == best_score and learner_score > 0.0 else 0.0
    total_score = sum(padded_scores)
    score_share = learner_score / total_score if total_score > 0.0 else 0.0
    ranked_reward = (
        1.0 - 2.0 * (placement - 1.0) / (player_count - 1.0)
        if player_count > 1
        else 1.0
    )
    mode = getattr(env_cfg, "terminal_reward_mode", "binary_win").strip().lower()
    if mode == "binary_win":
        reward = 1.0 if is_first > 0.0 else -1.0
    elif mode == "ranked":
        reward = ranked_reward
    elif mode == "score_share":
        reward = score_share
    elif mode == "survival_plus_rank":
        reward = ranked_reward
    else:
        raise ValueError(
            "env.terminal_reward_mode must be one of binary_win, ranked, "
            f"score_share, or survival_plus_rank; got {mode!r}."
        )
    return {
        "terminal_reward_unscaled": float(reward),
        "terminal_rank": float(rank),
        "terminal_placement": float(placement),
        "terminal_is_first": float(is_first),
        "terminal_score_share": float(score_share),
        "terminal_survival_time": 0.0,
        "terminal_ranked_reward": float(ranked_reward),
    }


def apply_early_terminal_reward_shaping(
    terminal_reward_value: float, step_index: int, env_cfg: EnvConfig
) -> float:
    if not getattr(env_cfg, "early_terminal_reward_shaping_enabled", True):
        return float(terminal_reward_value)
    horizon = max(int(getattr(env_cfg, "early_terminal_reward_shaping_horizon", 500)), 1)
    step_number = max(int(step_index) + 1, 1)
    if step_number >= horizon:
        return float(terminal_reward_value)
    bonus_scale = (horizon - step_number) / float(horizon)
    return float(terminal_reward_value) * (1.0 + bonus_scale)
