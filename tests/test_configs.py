import yaml


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_attention_shaped_reward_config_matches_unshaped_ppo_budget() -> None:
    shaped = load_yaml("configs/attention_shaped_reward_training.yaml")
    unshaped = load_yaml("configs/attention_training.yaml")

    assert shaped["model"]["architecture"] == "attention"
    assert shaped["ppo"] == unshaped["ppo"]


def test_attention_shaped_reward_config_uses_conservative_shaping_values() -> None:
    shaped = load_yaml("configs/attention_shaped_reward_training.yaml")

    assert shaped["env"]["reward_capture_planet"] == 0.1
    assert shaped["env"]["reward_ship_delta"] == 0.001
    assert shaped["env"]["reward_production_delta"] == 0.02
    assert shaped["env"]["reward_terminal_scale"] == 1.0
