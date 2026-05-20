from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import EnvConfig
from .constants import (
    BASE_CANDIDATE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_DIM,
    BASE_SELF_FEATURE_DIM,
    BOARD_CENTER,
    MAX_OWNER_FEATURE_PLAYERS,
    NO_OP_CANDIDATE_INDEX,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    BOARD_SIZE,
)
from .game_types import GameState, PlanetState, parse_observation


def real_candidate_slots(candidate_count: int) -> int:
    """Return target slots available after reserving index 0 for no-op.

    The policy action space uses ``candidate_count`` total choices. Index 0 is
    always the no-op action, so only ``candidate_count - 1`` slots can point at
    actual target planets.
    """

    return max(0, candidate_count - 1)


@dataclass(slots=True)
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    ship_counts: list[int]
    source_ships: int
    target_angles: list[float]


@dataclass(slots=True)
class TurnBatch:
    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    candidate_mask: np.ndarray
    contexts: list[DecisionContext]
    state: GameState


def feature_history_steps(env_cfg: EnvConfig | None = None) -> int:
    if env_cfg is None:
        return 1
    return max(1, int(getattr(env_cfg, "feature_history_steps", 1)))


def self_feature_dim(env_cfg: EnvConfig | None = None) -> int:
    return BASE_SELF_FEATURE_DIM * feature_history_steps(env_cfg)


def candidate_feature_dim(env_cfg: EnvConfig | None = None) -> int:
    return BASE_CANDIDATE_FEATURE_DIM * feature_history_steps(env_cfg)


def global_feature_dim(env_cfg: EnvConfig | None = None) -> int:
    return BASE_GLOBAL_FEATURE_DIM * feature_history_steps(env_cfg)


@dataclass(slots=True)
class FeatureSnapshot:
    self_by_source: dict[int, np.ndarray]
    candidate_by_source_target: dict[tuple[int, int, int], np.ndarray]
    global_features: np.ndarray


@dataclass(slots=True)
class FeatureHistoryBuffer:
    max_steps: int
    snapshots: deque[FeatureSnapshot] = field(default_factory=deque)

    def append(self, snapshot: FeatureSnapshot) -> None:
        self.snapshots.append(snapshot)
        while len(self.snapshots) > max(0, self.max_steps):
            self.snapshots.popleft()

    def clear(self) -> None:
        self.snapshots.clear()


def encode_turn(
    observation: Any,
    env_cfg: EnvConfig,
    *,
    env_index: int = 0,
    feature_history: FeatureHistoryBuffer | None = None,
) -> TurnBatch:
    state = (
        observation
        if isinstance(observation, GameState)
        else parse_observation(observation)
    )
    my_planets = sorted(
        (planet for planet in state.planets if planet.owner == state.player),
        key=lambda planet: planet.id,
    )
    if not my_planets:
        return TurnBatch(
            self_features=np.zeros((0, self_feature_dim(env_cfg)), dtype=np.float32),
            candidate_features=np.zeros(
                (0, env_cfg.candidate_count, candidate_feature_dim(env_cfg)),
                dtype=np.float32,
            ),
            global_features=np.zeros(
                (0, global_feature_dim(env_cfg)), dtype=np.float32
            ),
            candidate_mask=np.zeros((0, env_cfg.candidate_count), dtype=bool),
            contexts=[],
            state=state,
        )

    global_feat = build_global_features(state, env_cfg, feature_history)
    history_steps = feature_history_steps(env_cfg)
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    contexts: list[DecisionContext] = []

    for src in my_planets:
        candidates = build_candidates(src, state, env_cfg)
        cand_feat, cand_mask, ship_counts, candidate_ids, target_angles = (
            build_candidate_features(
                src,
                candidates,
                state,
                env_cfg,
                feature_history,
                env_index=env_index,
            )
        )
        self_feat = build_self_features(src, state, env_cfg, feature_history)
        self_rows.append(
            _stack_self_history(src.id, self_feat, feature_history, history_steps)
        )
        candidate_rows.append(
            _stack_candidate_history(
                env_index,
                src.id,
                candidate_ids,
                cand_feat,
                feature_history,
                history_steps,
            )
        )
        candidate_masks.append(cand_mask)
        contexts.append(
            DecisionContext(
                env_index=env_index,
                source_id=src.id,
                candidate_ids=candidate_ids,
                candidate_mask=cand_mask,
                ship_counts=ship_counts,
                source_ships=max(0, int(src.ships)),
                target_angles=target_angles,
            )
        )

    return TurnBatch(
        self_features=np.asarray(self_rows, dtype=np.float32),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32),
        global_features=np.repeat(
            _stack_global_history(global_feat, feature_history, history_steps)[None, :],
            len(self_rows),
            axis=0,
        ),
        candidate_mask=np.asarray(candidate_masks, dtype=bool),
        contexts=contexts,
        state=state,
    )


def _stack_self_history(
    source_id: int,
    current: np.ndarray,
    history: FeatureHistoryBuffer | None,
    steps: int,
) -> np.ndarray:
    if steps <= 1:
        return current
    snapshots = list(history.snapshots)[-(steps - 1) :] if history is not None else []
    rows = [
        snapshot.self_by_source.get(
            source_id, np.zeros(BASE_SELF_FEATURE_DIM, dtype=np.float32)
        )
        for snapshot in snapshots
    ]
    while len(rows) < steps - 1:
        rows.insert(0, np.zeros(BASE_SELF_FEATURE_DIM, dtype=np.float32))
    rows.append(current)
    return np.concatenate(rows).astype(np.float32, copy=False)


def _stack_candidate_history(
    env_index: int,
    source_id: int,
    candidate_ids: list[int],
    current: np.ndarray,
    history: FeatureHistoryBuffer | None,
    steps: int,
) -> np.ndarray:
    if steps <= 1:
        return current
    snapshots = list(history.snapshots)[-(steps - 1) :] if history is not None else []
    rows = []
    for candidate_index, candidate_id in enumerate(candidate_ids):
        if candidate_id == -1:
            candidate_history = [
                np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32)
                for _snapshot in snapshots
            ]
        else:
            candidate_history = [
                snapshot.candidate_by_source_target.get(
                    (env_index, source_id, candidate_id),
                    np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32),
                )
                for snapshot in snapshots
            ]
        while len(candidate_history) < steps - 1:
            candidate_history.insert(
                0, np.zeros(BASE_CANDIDATE_FEATURE_DIM, dtype=np.float32)
            )
        candidate_history.append(current[candidate_index])
        rows.append(np.concatenate(candidate_history).astype(np.float32, copy=False))
    return np.asarray(rows, dtype=np.float32)


def _stack_global_history(
    current: np.ndarray, history: FeatureHistoryBuffer | None, steps: int
) -> np.ndarray:
    if steps <= 1:
        return current
    snapshots = list(history.snapshots)[-(steps - 1) :] if history is not None else []
    rows = [snapshot.global_features for snapshot in snapshots]
    while len(rows) < steps - 1:
        rows.insert(0, np.zeros(BASE_GLOBAL_FEATURE_DIM, dtype=np.float32))
    rows.append(current)
    return np.concatenate(rows).astype(np.float32, copy=False)


def build_feature_snapshot(batch: TurnBatch) -> FeatureSnapshot:
    self_by_source: dict[int, np.ndarray] = {}
    candidate_by_source_target: dict[tuple[int, int, int], np.ndarray] = {}
    global_features = np.zeros(BASE_GLOBAL_FEATURE_DIM, dtype=np.float32)
    if batch.global_features.shape[0] > 0:
        global_features = batch.global_features[0, -BASE_GLOBAL_FEATURE_DIM:].astype(
            np.float32, copy=True
        )
    for row_index, context in enumerate(batch.contexts):
        self_by_source[context.source_id] = batch.self_features[
            row_index, -BASE_SELF_FEATURE_DIM:
        ].astype(np.float32, copy=True)
        for candidate_index, candidate_id in enumerate(context.candidate_ids):
            if candidate_id == -1:
                continue
            candidate_by_source_target[
                (context.env_index, context.source_id, candidate_id)
            ] = batch.candidate_features[
                row_index, candidate_index, -BASE_CANDIDATE_FEATURE_DIM:
            ].astype(np.float32, copy=True)
    return FeatureSnapshot(self_by_source, candidate_by_source_target, global_features)


def build_candidates(
    src: PlanetState, state: GameState, env_cfg: EnvConfig
) -> list[PlanetState]:
    """Build real target planets for candidate slots 1..candidate_count-1.

    Candidate index 0 remains reserved for no-op. Real target slots prioritize
    unobstructed shots first, then blocked shots, with a fallback that ensures we
    keep at least one unblocked target when any exists globally.
    """

    real_slots = real_candidate_slots(env_cfg.candidate_count)
    if real_slots <= 0:
        return []

    others = [planet for planet in state.planets if planet.id != src.id]
    ordered = sorted(others, key=lambda planet: (distance(src, planet), planet.id))

    unblocked: list[PlanetState] = []
    blocked: list[PlanetState] = []
    for tgt in ordered:
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        if shot_crosses_sun(src, angle, tgt):
            blocked.append(tgt)
        else:
            unblocked.append(tgt)

    selected = unblocked[:real_slots]
    if len(selected) < real_slots:
        selected.extend(blocked[: real_slots - len(selected)])

    # Fallback pass: if every selected candidate is blocked but there exists any
    # unblocked target globally, force one unblocked target into the candidate set.
    if selected and unblocked and all(tgt in blocked for tgt in selected):
        selected[-1] = unblocked[0]

    return selected


def build_self_features(
    src: PlanetState,
    state: GameState,
    env_cfg: EnvConfig,
    feature_history: FeatureHistoryBuffer | None = None,
) -> np.ndarray:
    my_planets = [planet for planet in state.planets if planet.owner == state.player]
    enemy_planets = [
        planet for planet in state.planets if planet.owner not in {-1, state.player}
    ]
    owner_counts, owner_ships, _owner_fleets, active_mask, player_count_feature = (
        owner_relative_summary(state, env_cfg)
    )
    previous = _latest_snapshot(feature_history)
    previous_self = (
        previous.self_by_source.get(src.id) if previous is not None else None
    )
    history_present = 1.0 if previous_self is not None else 0.0
    previous_ships = (
        float(previous_self[4]) * env_cfg.max_ships
        if previous_self is not None
        else src.ships
    )
    ship_delta = (float(src.ships) - previous_ships) / env_cfg.max_ships
    outgoing_friendly = sum(
        fleet.ships
        for fleet in state.fleets
        if fleet.owner == state.player
        and getattr(fleet, "from_planet_id", None) == src.id
    )
    incoming_friendly, incoming_enemy = incoming_fleet_pressure(src, state, env_cfg)
    return np.asarray(
        [
            1.0,
            src.x / BOARD_SIZE,
            src.y / BOARD_SIZE,
            src.radius / 5.0,
            min(src.ships, env_cfg.max_ships) / env_cfg.max_ships,
            src.production / env_cfg.max_production,
            1.0 if is_rotating_planet(src) else 0.0,
            len(my_planets) / env_cfg.max_planets,
            len(enemy_planets) / env_cfg.max_planets,
            total_ships(my_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            total_ships(enemy_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            *owner_counts.tolist(),
            *owner_ships.tolist(),
            *active_mask.tolist(),
            player_count_feature,
            ship_delta,
            history_present,
            history_present,  # ownership stable: previous self row means same owner slot.
            outgoing_friendly / env_cfg.max_ships,
            incoming_friendly / env_cfg.max_ships,
            incoming_enemy / env_cfg.max_ships,
        ],
        dtype=np.float32,
    )


def build_candidate_features(
    src: PlanetState,
    candidates: list[PlanetState],
    state: GameState,
    env_cfg: EnvConfig,
    feature_history: FeatureHistoryBuffer | None = None,
    *,
    env_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[float]]:
    features = np.zeros(
        (env_cfg.candidate_count, BASE_CANDIDATE_FEATURE_DIM), dtype=np.float32
    )
    candidate_mask = np.zeros((env_cfg.candidate_count,), dtype=bool)
    ship_counts = [0] * env_cfg.candidate_count
    candidate_ids = [-1] * env_cfg.candidate_count
    target_angles = [0.0] * env_cfg.candidate_count
    previous = _latest_snapshot(feature_history)
    if env_cfg.candidate_count > NO_OP_CANDIDATE_INDEX:
        candidate_mask[NO_OP_CANDIDATE_INDEX] = True

    if env_cfg.candidate_count <= 1:
        return features, candidate_mask, ship_counts, candidate_ids, target_angles

    blocked: list[tuple[PlanetState, float]] = []
    unblocked: list[tuple[PlanetState, float]] = []
    for tgt in candidates:
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        if shot_crosses_sun(src, angle, tgt):
            blocked.append((tgt, angle))
        else:
            unblocked.append((tgt, angle))

    ordered: list[tuple[PlanetState, float]] = []
    real_slots = env_cfg.candidate_count - 1
    ordered.extend(unblocked[:real_slots])
    if len(ordered) < real_slots:
        ordered.extend(blocked[: real_slots - len(ordered)])
    if ordered and unblocked and all(shot_crosses_sun(src, angle, tgt) for tgt, angle in ordered):
        ordered[-1] = unblocked[0]

    for idx, (tgt, angle) in enumerate(ordered, start=1):
        if idx >= env_cfg.candidate_count:
            break
        dx = tgt.x - src.x
        dy = tgt.y - src.y
        dist = distance(src, tgt)
        crosses_sun = shot_crosses_sun(src, angle, tgt)
        previous_candidate = (
            previous.candidate_by_source_target.get((env_index, src.id, tgt.id))
            if previous is not None
            else None
        )
        previous_target_ships = (
            float(previous_candidate[9]) * env_cfg.max_ships
            if previous_candidate is not None
            else tgt.ships
        )
        current_owner = target_owner_one_hot(tgt.owner, state, env_cfg)
        previous_owner = (
            previous_candidate[14:18]
            if previous_candidate is not None
            else current_owner
        )
        owner_changed = (
            1.0
            if previous_candidate is not None
            and not np.array_equal(current_owner > 0.5, previous_owner > 0.5)
            else 0.0
        )
        incoming_friendly, incoming_enemy = incoming_fleet_pressure(tgt, state, env_cfg)
        turns_to_arrival = dist / max(float(env_cfg.ship_speed), 1e-6)
        features[idx] = np.asarray(
            [
                1.0,
                1.0 if tgt.owner == -1 else 0.0,
                1.0 if tgt.owner == state.player else 0.0,
                1.0 if tgt.owner not in {-1, state.player} else 0.0,
                tgt.x / BOARD_SIZE,
                tgt.y / BOARD_SIZE,
                dx / BOARD_SIZE,
                dy / BOARD_SIZE,
                dist / BOARD_SIZE,
                min(tgt.ships, env_cfg.max_ships) / env_cfg.max_ships,
                tgt.production / env_cfg.max_production,
                1.0 if is_rotating_planet(tgt) else 0.0,
                1.0 if crosses_sun else 0.0,
                min(src.ships, env_cfg.max_ships) / env_cfg.max_ships,
                *current_owner.tolist(),
                turns_to_arrival / env_cfg.episode_steps,
                incoming_friendly / env_cfg.max_ships,
                incoming_enemy / env_cfg.max_ships,
                (float(tgt.ships) - previous_target_ships) / env_cfg.max_ships,
                owner_changed,
                1.0,
            ],
            dtype=np.float32,
        )
        ship_counts[idx] = max(0, int(src.ships))
        candidate_mask[idx] = not crosses_sun
        candidate_ids[idx] = tgt.id
        target_angles[idx] = angle

    return features, candidate_mask, ship_counts, candidate_ids, target_angles


def build_global_features(
    state: GameState,
    env_cfg: EnvConfig,
    feature_history: FeatureHistoryBuffer | None = None,
) -> np.ndarray:
    my_planets = [planet for planet in state.planets if planet.owner == state.player]
    enemy_planets = [
        planet for planet in state.planets if planet.owner not in {-1, state.player}
    ]
    neutral_planets = [planet for planet in state.planets if planet.owner == -1]
    my_fleets = [fleet for fleet in state.fleets if fleet.owner == state.player]
    enemy_fleets = [fleet for fleet in state.fleets if fleet.owner != state.player]
    owner_counts, owner_ships, owner_fleets, active_mask, player_count_feature = (
        owner_relative_summary(state, env_cfg)
    )
    owner_production = owner_relative_production(state, env_cfg)
    previous = _latest_snapshot(feature_history)
    if previous is None:
        ship_delta = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
        planet_delta = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
        fleet_delta = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
        production_delta = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
    else:
        prior = previous.global_features
        planet_delta = owner_counts - prior[8:12]
        ship_delta = owner_ships - prior[12:16]
        fleet_delta = owner_fleets - prior[16:20]
        production_delta = owner_production - prior[25:29]
    return np.asarray(
        [
            state.step / env_cfg.episode_steps,
            len(my_planets) / env_cfg.max_planets,
            len(enemy_planets) / env_cfg.max_planets,
            len(neutral_planets) / env_cfg.max_planets,
            total_ships(my_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            total_ships(enemy_planets) / (env_cfg.max_planets * env_cfg.max_ships),
            sum(fleet.ships for fleet in my_fleets)
            / (env_cfg.max_planets * env_cfg.max_ships),
            sum(fleet.ships for fleet in enemy_fleets)
            / (env_cfg.max_planets * env_cfg.max_ships),
            *owner_counts.tolist(),
            *owner_ships.tolist(),
            *owner_fleets.tolist(),
            *active_mask.tolist(),
            player_count_feature,
            *owner_production.tolist(),
            *ship_delta.tolist(),
            *planet_delta.tolist(),
            *fleet_delta.tolist(),
            *production_delta.tolist(),
        ],
        dtype=np.float32,
    )


def _latest_snapshot(history: FeatureHistoryBuffer | None) -> FeatureSnapshot | None:
    if history is None or not history.snapshots:
        return None
    return history.snapshots[-1]


def incoming_fleet_pressure(
    planet: PlanetState, state: GameState, env_cfg: EnvConfig
) -> tuple[float, float]:
    friendly = 0.0
    enemy = 0.0
    for fleet in state.fleets:
        if not fleet_aims_at_planet(fleet, planet):
            continue
        if fleet.owner == state.player:
            friendly += float(fleet.ships)
        else:
            enemy += float(fleet.ships)
    return friendly, enemy


def fleet_aims_at_planet(fleet: Any, planet: PlanetState) -> bool:
    if not all(hasattr(fleet, attr) for attr in ("x", "y", "angle")):
        return False
    dx = planet.x - float(fleet.x)
    dy = planet.y - float(fleet.y)
    forward = dx * math.cos(float(fleet.angle)) + dy * math.sin(float(fleet.angle))
    if forward < 0.0:
        return False
    closest_x = float(fleet.x) + math.cos(float(fleet.angle)) * forward
    closest_y = float(fleet.y) + math.sin(float(fleet.angle)) * forward
    return math.hypot(planet.x - closest_x, planet.y - closest_y) <= planet.radius


def owner_relative_production(state: GameState, env_cfg: EnvConfig) -> np.ndarray:
    player_count = clipped_player_count(env_cfg)
    production = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
    for planet in state.planets:
        slot = relative_owner_slot(planet.owner, state.player, player_count)
        if slot is None:
            continue
        production[slot] += float(planet.production)
    return production / (env_cfg.max_planets * env_cfg.max_production)


def owner_relative_summary(
    state: GameState, env_cfg: EnvConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return fixed-size owner-relative count/ship/fleet summaries.

    Slot 0 always represents the current player. Remaining slots represent
    ``(owner - current_player) % player_count`` for up to four players; inactive
    player slots are zero padded so the feature shape is stable for JAX.
    Neutral ownership (``-1``) is intentionally excluded from these summaries.
    """

    player_count = clipped_player_count(env_cfg)
    counts = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
    ships = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
    fleets = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)

    for planet in state.planets:
        slot = relative_owner_slot(planet.owner, state.player, player_count)
        if slot is None:
            continue
        counts[slot] += 1.0
        ships[slot] += float(planet.ships)

    for fleet in state.fleets:
        slot = relative_owner_slot(fleet.owner, state.player, player_count)
        if slot is None:
            continue
        fleets[slot] += float(fleet.ships)

    denom = env_cfg.max_planets * env_cfg.max_ships
    active_mask = np.asarray(
        [idx < player_count for idx in range(MAX_OWNER_FEATURE_PLAYERS)],
        dtype=np.float32,
    )
    return (
        counts / env_cfg.max_planets,
        ships / denom,
        fleets / denom,
        active_mask,
        player_count / MAX_OWNER_FEATURE_PLAYERS,
    )


def target_owner_one_hot(
    owner: int, state: GameState, env_cfg: EnvConfig
) -> np.ndarray:
    """Encode a target owner relative to ``state.player`` in four fixed slots."""

    one_hot = np.zeros((MAX_OWNER_FEATURE_PLAYERS,), dtype=np.float32)
    slot = relative_owner_slot(owner, state.player, clipped_player_count(env_cfg))
    if slot is not None:
        one_hot[slot] = 1.0
    return one_hot


def clipped_player_count(env_cfg: EnvConfig) -> int:
    return max(
        1, min(MAX_OWNER_FEATURE_PLAYERS, int(getattr(env_cfg, "player_count", 2)))
    )


def relative_owner_slot(owner: int, player: int, player_count: int) -> int | None:
    if owner < 0 or owner >= player_count:
        return None
    return (int(owner) - int(player)) % player_count


def ship_bucket_fraction(bucket: int, bucket_count: int) -> float:
    if bucket <= 0 or bucket_count <= 1:
        return 0.0
    return min(1.0, max(0.0, bucket / float(bucket_count - 1)))


def ship_count_for_bucket(
    available_ships: float | int, bucket: int, bucket_count: int
) -> int:
    available = max(0, int(available_ships))
    fraction = ship_bucket_fraction(bucket, bucket_count)
    if available <= 0 or fraction <= 0.0:
        return 0
    return min(available, max(1, math.ceil(available * fraction)))


def distance(a: PlanetState, b: PlanetState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def total_ships(planets: list[PlanetState]) -> float:
    return float(sum(planet.ships for planet in planets))


def is_rotating_planet(planet: PlanetState) -> bool:
    dx = planet.x - BOARD_CENTER[0]
    dy = planet.y - BOARD_CENTER[1]
    orbital_radius = math.hypot(dx, dy)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def shot_crosses_sun(src: PlanetState, angle: float, tgt: PlanetState) -> bool:
    start_x = src.x + math.cos(angle) * (src.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = src.y + math.sin(angle) * (src.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    return (
        point_to_segment_distance(BOARD_CENTER, (start_x, start_y), (tgt.x, tgt.y))
        < SUN_RADIUS
    )


def point_to_segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    segment_len_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if segment_len_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * (end[0] - start[0])
        + (point[1] - start[1]) * (end[1] - start[1])
    ) / segment_len_sq
    projection = max(0.0, min(1.0, projection))
    closest_x = start[0] + projection * (end[0] - start[0])
    closest_y = start[1] + projection * (end[1] - start[1])
    return math.hypot(point[0] - closest_x, point[1] - closest_y)
