from __future__ import annotations

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.artifacts import replay
from src.config import TrainConfig
from src.features import DecisionContext, TurnBatch
from src.game.types import GameState
from src.jax.policy import JaxPolicyOutput


class _FakePolicy:
    def apply(self, *_args, **_kwargs) -> JaxPolicyOutput:
        target_logits = jnp.full((1, 2, 12), -10.0, dtype=jnp.float32)
        target_logits = target_logits.at[0, 0, 3].set(10.0)
        target_logits = target_logits.at[0, 1, 5].set(9.0)

        ship_logits = jnp.full((1, 2, 12, 4), -10.0, dtype=jnp.float32)
        ship_logits = ship_logits.at[0, 0, 3, 2].set(10.0)
        ship_logits = ship_logits.at[0, 1, 5, 3].set(10.0)
        return JaxPolicyOutput(
            target_logits=target_logits,
            ship_logits=ship_logits,
            value=jnp.zeros((1,), dtype=jnp.float32),
            decoded_target_sequence=jnp.full((1, 2), -1, dtype=jnp.int32),
        )


def test_jax_replay_actor_handles_sequence_policy_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = TrainConfig()
    cfg.task.candidate_count = 12
    cfg.task.ship_bucket_count = 4
    cfg.task.max_fleets = 8
    cfg.task.trajectory_shield_enabled = False

    checkpoint_path = tmp_path / "jax_ckpt_000100.pkl"
    with checkpoint_path.open("wb") as file:
        pickle.dump({"params": {"fake": "params"}}, file)

    candidate_mask = np.ones((12,), dtype=bool)
    candidate_ids = list(range(12))
    target_angles = [float(index) for index in range(12)]

    batch = TurnBatch(
        self_features=np.zeros((1, 1), dtype=np.float32),
        candidate_features=np.zeros((1, 12, 1), dtype=np.float32),
        global_features=np.zeros((1, 1), dtype=np.float32),
        candidate_mask=candidate_mask[None, :],
        contexts=[
            DecisionContext(
                env_index=0,
                source_id=7,
                candidate_ids=candidate_ids,
                candidate_mask=candidate_mask,
                ship_counts=[0] * 12,
                source_ships=10,
                target_angles=target_angles,
            )
        ],
        state=GameState(player=0, step=0, planets=[], fleets=[]),
    )

    monkeypatch.setattr(replay, "build_jax_policy", lambda cfg: _FakePolicy())
    monkeypatch.setattr(replay, "encode_turn", lambda *_args, **_kwargs: batch)

    act = replay._build_jax_policy_actions(cfg, checkpoint_path)

    assert act({}) == [[7, 3.0, 7], [7, 5.0, 3]]
