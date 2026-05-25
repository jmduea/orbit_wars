from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp

from src.artifacts import replay
from src.config import TrainConfig
from src.jax.env import JaxAction
from src.jax.features import TurnBatch
from src.jax.policy import JaxPolicyOutput
from src.game.constants import MAX_PLANETS


class _FakePolicy:
    def apply(self, *_args, **_kwargs) -> JaxPolicyOutput:
        target_logits = jnp.full((1, 1, 4), -10.0, dtype=jnp.float32)
        target_logits = target_logits.at[0, 0, 1].set(10.0)
        ship_logits = jnp.full((1, 1, 4, 2), -10.0, dtype=jnp.float32)
        ship_logits = ship_logits.at[0, 0, 1, 1].set(10.0)
        return JaxPolicyOutput(
            target_logits=target_logits,
            ship_logits=ship_logits,
            value=jnp.zeros((1,), dtype=jnp.float32),
            decoded_target_sequence=jnp.full((1, 1), -1, dtype=jnp.int32),
        )


def test_jax_replay_actor_uses_submission_runtime_path(monkeypatch, tmp_path: Path) -> None:
    cfg = TrainConfig()
    cfg.task.candidate_count = 4
    cfg.task.ship_bucket_count = 2
    cfg.task.max_fleets = 8
    cfg.task.trajectory_shield_enabled = False

    checkpoint_path = tmp_path / "jax_ckpt_000100.pkl"
    with checkpoint_path.open("wb") as file:
        pickle.dump({"params": {"fake": "params"}}, file)

    fake_batch = TurnBatch(
        planet_features=jnp.zeros((MAX_PLANETS, 13), dtype=jnp.float32),
        planet_mask=jnp.ones((MAX_PLANETS,), dtype=bool),
        edge_features=jnp.zeros((MAX_PLANETS, 3, 12), dtype=jnp.float32),
        edge_mask=jnp.zeros((MAX_PLANETS, 3), dtype=bool),
        edge_src_ids=jnp.arange(MAX_PLANETS, dtype=jnp.int32),
        edge_tgt_ids=jnp.full((MAX_PLANETS, 3), -1, dtype=jnp.int32),
        global_features=jnp.zeros((46,), dtype=jnp.float32),
        theta_ref=jnp.array(0.0, dtype=jnp.float32),
    )
    fake_action = JaxAction(
        source_id=jnp.array([7], dtype=jnp.int32),
        angle=jnp.array([1.0], dtype=jnp.float32),
        ships=jnp.array([3.0], dtype=jnp.float32),
        valid=jnp.array([True], dtype=bool),
    )

    with patch("src.artifacts.replay.build_jax_policy", return_value=_FakePolicy()), patch(
        "src.artifacts.replay.jax_game_from_observation"
    ) as mock_game, patch(
        "src.artifacts.replay.encode_turn", return_value=fake_batch
    ), patch(
        "src.artifacts.replay.select_runtime_shielded_policy_actions",
        return_value=fake_action,
    ), patch(
        "src.artifacts.replay.moves_from_jax_action", return_value=[[7, 1.0, 3]]
    ):
        act = replay._build_jax_policy_actions(cfg, checkpoint_path)
        moves = act({"player": 0, "planets": []})

    assert moves == [[7, 1.0, 3]]
    mock_game.assert_called_once()
