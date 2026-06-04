"""Held-out JAX qualifier games on ``eval_seed_set`` only (R18, R25)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jax
from src.artifacts.checkpoint_compat import (
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
    validate_checkpoint_feature_compatibility,
)
from src.config import TrainConfig
from src.game.constants import MAX_STEPS
from src.jax.env import JaxEnvState, reset, step
from src.jax.features import encode_turn
from src.jax.policy import build_jax_policy
from src.jax.qualifier_calibration import legs_for_stage
from src.jax.rollout.types import JaxTrainState
from src.jax.tournament_qualifiers.metrics import (
    final_ship_scores,
    learner_won_from_final_scores,
)
from src.jax.train.state import init_train_state
from src.opponents.jax_actions.builders import (
    build_noop_action_from_edge_batch,
    build_sniper_action_from_edge_batch,
)
from src.opponents.jax_actions.sampling import (
    _shielded_random_edge_action,
    _shielded_scripted_edge_action,
)

LEG_BUILDERS: dict[str, str] = {
    "random": "random",
    "noop": "noop",
    "nearest_sniper": "nearest_sniper",
}


def held_out_eval_seeds(cfg: TrainConfig) -> tuple[int, ...]:
    """Seeds reserved for qualifiers; must not overlap training draws (R25)."""

    reserved = frozenset(int(s) for s in cfg.eval_seed_set)
    if not reserved:
        raise ValueError("eval_seed_set must be non-empty for SSOT qualifier eval")
    training = frozenset(int(s) for s in cfg.training_seed_set)
    overlap = reserved & training
    if overlap:
        raise ValueError(
            "eval_seed_set must be disjoint from training_seed_set for qualifiers; "
            f"overlap={sorted(overlap)}"
        )
    if int(cfg.seed) in reserved:
        raise ValueError(
            f"training.seed={cfg.seed} must not appear in eval_seed_set"
        )
    return tuple(sorted(reserved))


def _load_eval_train_state(
    cfg: TrainConfig,
    checkpoint_path: Path,
    *,
    policy_key: jax.Array,
) -> tuple[object, JaxTrainState]:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"JAX checkpoint must contain a parameter payload: {checkpoint_path}"
        )
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=str(checkpoint_path)
    )
    validate_checkpoint_feature_compatibility(
        checkpoint, cfg.task, checkpoint_path=str(checkpoint_path)
    )
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    params = jax.device_put(checkpoint["params"])
    return policy, replace(train_state, params=params)


def _learner_action(
    key: jax.Array,
    policy: object,
    train_state: JaxTrainState,
    env_state: JaxEnvState,
    batch,
    cfg: TrainConfig,
):
    from src.jax.action_sampling import _sample_shielded_sequence_with_params
    from src.opponents.jax_actions.builders import build_action_from_factored_batch

    sample = _sample_shielded_sequence_with_params(
        key,
        env_state.game,
        batch,
        train_state.params,
        policy,
        cfg,
        deterministic=True,
    )
    return build_action_from_factored_batch(
        env_state.game,
        batch,
        sample.source_index,
        sample.target_slot,
        sample.ship_bucket,
        sample.stop_flag,
        sample.step_mask,
        cfg,
        ship_fraction=sample.ship_fraction,
    )


def _opponent_action_for_leg(
    key: jax.Array,
    game,
    batch,
    cfg: TrainConfig,
    leg: str,
):
    if leg == "noop":
        return build_noop_action_from_edge_batch(game, batch, cfg)
    if leg == "random":
        return _shielded_random_edge_action(key, game, batch, cfg)
    if leg == "nearest_sniper":
        return _shielded_scripted_edge_action(
            game, batch, cfg, build_sniper_action_from_edge_batch
        )
    raise ValueError(f"unknown qualifier leg {leg!r}")


def run_qualifier_game(
    cfg: TrainConfig,
    *,
    policy: object,
    train_state: JaxTrainState,
    eval_seed: int,
    leg: str,
) -> bool:
    """Play one 2p held-out game; return whether learner won on final ship score."""

    if leg not in LEG_BUILDERS:
        raise ValueError(f"unknown qualifier leg {leg!r}")
    if cfg.task.player_count != 2:
        raise ValueError(
            "SSOT qualifier eval supports 2-player tasks only; "
            f"got player_count={cfg.task.player_count}"
        )
    key = jax.random.PRNGKey(int(eval_seed))
    key, reset_key, step_key = jax.random.split(key, 3)
    env_state, batch = reset(reset_key, cfg.task)
    env_state = env_state._replace(learner_player=jnp.array(0, dtype=jnp.int32))

    for step_idx in range(MAX_STEPS):
        learner_key, opp_key, step_key = jax.random.split(
            jax.random.fold_in(step_key, step_idx), 3
        )
        learner_action = _learner_action(
            learner_key, policy, train_state, env_state, batch, cfg
        )
        opp_player = 1 - int(jax.device_get(env_state.learner_player))
        opp_game = env_state.game._replace(
            player=jnp.array(opp_player, dtype=jnp.int32)
        )
        opp_batch = encode_turn(opp_game, cfg.task)
        opponent_action = _opponent_action_for_leg(
            opp_key, opp_game, opp_batch, cfg, leg
        )
        env_state, result = step(
            env_state,
            learner_action,
            opponent_action,
            cfg.task,
            cfg.reward,
        )
        if bool(jax.device_get(result.done.reshape(()))):
            scores = final_ship_scores(
                env_state.game, int(cfg.task.player_count)
            )
            learner_player = int(jax.device_get(env_state.learner_player.reshape(())))
            return learner_won_from_final_scores(scores, learner_player)
        batch = result.batch

    scores = final_ship_scores(env_state.game, int(cfg.task.player_count))
    learner_player = int(jax.device_get(env_state.learner_player.reshape(())))
    return learner_won_from_final_scores(scores, learner_player)


def run_held_out_qualifier_eval(
    cfg: TrainConfig,
    *,
    checkpoint_path: Path,
    stage: int,
) -> dict[str, tuple[int, int]]:
    """Run qualifier legs for the stage on ``eval_seed_set`` with checkpoint policy."""

    ssot = cfg.artifacts.ssot_pipeline
    games_per_seed = int(ssot.qualifier_games_per_seed)
    if games_per_seed <= 0:
        return {}
    eval_seeds = held_out_eval_seeds(cfg)
    legs = legs_for_stage(stage)
    policy_key = jax.random.PRNGKey(int(cfg.seed) + int(stage) * 10_007)
    policy, train_state = _load_eval_train_state(
        cfg, checkpoint_path, policy_key=policy_key
    )
    leg_wins: dict[str, tuple[int, int]] = {}
    for leg in legs:
        wins = 0
        games = 0
        for eval_seed in eval_seeds:
            for game_index in range(games_per_seed):
                game_seed = int(eval_seed) + game_index * 1_000_003
                if run_qualifier_game(
                    cfg,
                    policy=policy,
                    train_state=train_state,
                    eval_seed=game_seed,
                    leg=leg,
                ):
                    wins += 1
                games += 1
        leg_wins[leg] = (wins, games)
    return leg_wins
