from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.game.trajectory_shield import apply_trajectory_shield_to_turn_batch_v2
from src.jax.env import JaxAction
from src.jax.policy_v2 import edge_action_count
from src.opponents.jax_actions.builders_v2 import build_random_action_from_edge_batch
from src.opponents.jax_actions.sampling import _select_env_action


def _four_player_step_action_v2_random(
    player_id: jax.Array,
    *,
    opp_key: jax.Array,
    player_games,
    player_batches,
    learner_action: JaxAction,
    learner_player: jax.Array,
    cfg: TrainConfig,
) -> JaxAction:
    """Build one player's env-batch action row for v2 random-opponent 4p rollout."""

    player_batch = jax.tree.map(
        lambda x: jnp.take(x, player_id, axis=0), player_batches
    )
    player_game = jax.tree.map(lambda x: jnp.take(x, player_id, axis=0), player_games)
    player_key = jax.random.fold_in(opp_key, player_id)
    edge_count = edge_action_count(cfg.task)
    player_shielded = jax.vmap(
        lambda game, turn: apply_trajectory_shield_to_turn_batch_v2(
            game, turn, cfg.task
        )
    )(player_game, player_batch)
    opponent_action = build_random_action_from_edge_batch(
        player_key,
        player_game,
        player_shielded.batch,
        cfg,
        player_shielded.ship_bucket_mask.reshape(
            player_game.step.shape[0], edge_count, cfg.task.ship_bucket_count
        ),
    )
    is_learner_player = learner_player == player_id
    return _select_env_action(is_learner_player, learner_action, opponent_action)
