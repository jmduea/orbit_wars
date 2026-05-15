import torch
from torch.distributions import Categorical

from src.policy import PolicyOutput
from src.ppo import action_log_prob_and_entropy, gather_target_ship_logits, sample_actions


def test_action_log_prob_and_entropy_gathers_ship_logits_for_selected_target() -> None:
    outputs = PolicyOutput(
        target_logits=torch.tensor([[0.0, 2.0, -1.0], [1.5, -0.5, 0.25]]),
        ship_logits=torch.tensor(
            [
                [[4.0, 0.0], [0.0, 3.0], [1.0, 1.0]],
                [[-1.0, 2.0], [5.0, -5.0], [0.5, 0.25]],
            ]
        ),
        value=torch.zeros(2),
    )
    target_index = torch.tensor([1, 2])
    ship_bucket = torch.tensor([1, 0])

    log_prob, entropy = action_log_prob_and_entropy(outputs, target_index, ship_bucket)

    target_dist = Categorical(logits=outputs.target_logits)
    selected_ship_logits = outputs.ship_logits[torch.arange(2), target_index]
    ship_dist = Categorical(logits=selected_ship_logits)
    expected_log_prob = target_dist.log_prob(target_index) + ship_dist.log_prob(ship_bucket)
    expected_entropy = target_dist.entropy() + ship_dist.entropy()

    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    torch.testing.assert_close(log_prob, expected_log_prob)
    torch.testing.assert_close(entropy, expected_entropy)


def test_sample_actions_conditions_ship_bucket_on_masked_valid_target() -> None:
    outputs = PolicyOutput(
        target_logits=torch.tensor([[0.0, torch.finfo(torch.float32).min, 2.0]]),
        ship_logits=torch.tensor([[[10.0, -10.0, -10.0], [20.0, -20.0, -20.0], [-10.0, -10.0, 10.0]]]),
        value=torch.zeros(1),
    )

    sampled = sample_actions(outputs, deterministic=True)

    assert sampled.target_index.tolist() == [2]
    assert sampled.ship_bucket.tolist() == [2]
    assert sampled.log_prob.shape == (1,)
    assert sampled.entropy.shape == (1,)


def test_gather_target_ship_logits_preserves_legacy_batch_ship_logits() -> None:
    ship_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    selected = gather_target_ship_logits(ship_logits, torch.tensor([0, 1]))

    assert selected is ship_logits
