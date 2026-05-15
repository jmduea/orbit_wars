
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import EnvConfig, TrainConfig
from .features import TurnBatch, encode_turn
from .game_types import GameState, PlanetState, parse_observation
from .opponents import OpponentPolicy


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
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.previous_player_state: GameState | None = None
        self.previous_opp_state: GameState | None = None
        self.episode_index = 0
        self.learner_player = 0

    def reset(self, seed: int | None = None) -> TurnBatch:
        make_fn = self.make_fn or default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=2)
        states = self.env.step([[], []])
        learner_state = states[self.learner_player]
        opponent_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(learner_state)
        self.last_opp_obs = extract_observation(opponent_state)
        self.previous_player_state = parse_observation(self.last_obs)
        self.previous_opp_state = parse_observation(self.last_opp_obs)
        self.episode_index += 1
        return encode_turn(self.previous_player_state, self.cfg.env, env_index=self.env_index)

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        opponent_action = self.opponent.act(self.last_opp_obs)
        if self.learner_player == 0:
            joint_action = [player_action, opponent_action]
        else:
            joint_action = [opponent_action, player_action]
        states = self.env.step(joint_action)
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]
        next_obs = extract_observation(player_state)
        next_opp_obs = extract_observation(opp_state)
        next_player_state = parse_observation(next_obs)
        next_opp_state = parse_observation(next_opp_obs)
        done = extract_status(player_state) != "ACTIVE"

        terminal_component = (
            terminal_reward(player_state, opp_state) * self.cfg.env.reward_terminal_scale if done else 0.0
        )
        shaping_components = shaped_reward_components(self.previous_player_state, next_player_state, self.cfg.env)
        shaping_component = sum(shaping_components.values())
        reward = terminal_component + shaping_component

        self.last_obs = next_obs
        self.last_opp_obs = next_opp_obs
        self.previous_player_state = next_player_state
        self.previous_opp_state = next_opp_state

        batch = encode_turn(next_player_state, self.cfg.env, env_index=self.env_index)
        info = {
            "learner_player": self.learner_player,
            "player_status": extract_status(player_state),
            "opponent_status": extract_status(opp_state),
            "reward": reward,
            "terminal_reward": terminal_component,
            "shaping_reward": shaping_component,
            **shaping_components,
        }
        return StepResult(batch=batch, reward=reward, done=done, info=info)


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

    components["reward_capture_planet"] = env_cfg.reward_capture_planet * float(captured - lost)

    previous_ship_advantage = ship_advantage(previous_state, player)
    current_ship_advantage = ship_advantage(current_state, player)
    components["reward_ship_delta"] = env_cfg.reward_ship_delta * (current_ship_advantage - previous_ship_advantage)

    previous_production = controlled_production(previous_state.planets, player)
    current_production = controlled_production(current_state.planets, player)
    components["reward_production_delta"] = env_cfg.reward_production_delta * (current_production - previous_production)
    return components


def planets_by_id(planets: list[PlanetState]) -> dict[int, PlanetState]:
    return {planet.id: planet for planet in planets}


def ship_advantage(state: GameState, player: int) -> float:
    player_ships = sum(planet.ships for planet in state.planets if planet.owner == player)
    player_ships += sum(fleet.ships for fleet in state.fleets if fleet.owner == player)
    opponent_ships = sum(planet.ships for planet in state.planets if planet.owner not in {-1, player})
    opponent_ships += sum(fleet.ships for fleet in state.fleets if fleet.owner not in {-1, player})
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


def terminal_reward(player_state: Any, opp_state: Any) -> float:
    player_reward = extract_reward(player_state)
    opponent_reward = extract_reward(opp_state)
    if player_reward > 0.0 and opponent_reward > 0.0:
        return 0.0
    return player_reward
