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


def test_default_config_preserves_torch_kaggle_training_path() -> None:
    default = load_yaml("default_cfg.yaml")

    assert default["env_backend"] == "kaggle"
    assert default["rl_backend"] == "torch"


def test_jax_training_config_selects_jax_backends() -> None:
    jax_cfg = load_yaml("configs/jax_training.yaml")

    assert jax_cfg["env_backend"] == "jax"
    assert jax_cfg["rl_backend"] == "jax"
    assert jax_cfg["opponent"] in {"self", "random"}
