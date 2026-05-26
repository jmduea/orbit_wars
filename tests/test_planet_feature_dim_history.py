"""Planet feature width stays base-sized when global history stacks."""

import jax

from src.config import TaskConfig
from src.features.registry import global_feature_dim, planet_feature_dim
from src.game.constants import MAX_PLANETS
from src.jax.env import reset
from src.jax.features import encode_turn


def test_planet_dim_ignores_history_steps() -> None:
    cfg = TaskConfig(
        max_fleets=32,
        candidate_count=4,
        player_count=2,
        feature_history_steps=5,
        ship_feature_scale=1000.0,
    )
    state, _ = reset(jax.random.PRNGKey(9), cfg)
    batch = encode_turn(state.game, cfg)
    assert planet_feature_dim(cfg) == 13
    assert batch.planet_features.shape == (MAX_PLANETS, 13)
    assert batch.global_features.shape == (global_feature_dim(cfg),)
