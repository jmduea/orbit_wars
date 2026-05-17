import pytest
import torch

from src.policy import build_policy


@pytest.mark.parametrize("architecture", ["mlp", "attention", "transformer"])
def test_policy_returns_per_candidate_ship_logits(architecture: str) -> None:
    policy = build_policy(
        architecture=architecture,
        self_dim=4,
        candidate_dim=5,
        global_dim=3,
        candidate_count=6,
        ship_bucket_count=4,
        hidden_size=8,
        attention_heads=2,
    )
    outputs = policy(
        self_features=torch.randn(2, 4),
        candidate_features=torch.randn(2, 6, 5),
        global_features=torch.randn(2, 3),
        candidate_mask=torch.tensor(
            [
                [True, False, True, True, False, True],
                [True, True, False, False, False, True],
            ]
        ),
    )

    assert outputs.target_logits.shape == (2, 6)
    assert outputs.ship_logits.shape == (2, 6, 4)
    assert outputs.value.shape == (2,)
