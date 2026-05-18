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


def test_jax_self_play_shaped_reward_config_combines_jax_self_play_and_shaping() -> None:
    shaped = load_yaml("configs/jax_self_play_shaped_reward_training.yaml")
    unshaped = load_yaml("configs/jax_training.yaml")

    assert shaped["env_backend"] == "jax"
    assert shaped["rl_backend"] == "jax"
    assert shaped["opponent"] == "self"
    assert shaped["self_play_enabled"] is True
    assert shaped["model"]["architecture"] == "attention"
    assert shaped["ppo"] == unshaped["ppo"]
    assert shaped["env"]["reward_capture_planet"] == 0.1
    assert shaped["env"]["reward_ship_delta"] == 0.001
    assert shaped["env"]["reward_production_delta"] == 0.02
    assert shaped["env"]["reward_terminal_scale"] == 1.0


def test_train_config_loads_training_format_schedule_and_allocations() -> None:
    from src.config import load_train_config

    cfg = load_train_config("configs/jax_training.yaml")

    assert cfg.env.player_count == 2
    assert [entry["player_count"] for entry in cfg.training_format.format_schedule] == [2, 4]
    assert cfg.ppo.num_envs_2p == 4
    assert cfg.ppo.num_envs_4p == 4
    assert [group["player_count"] for group in cfg.training_format.rollout_groups] == [2, 4]


def test_jax_mixed_training_config_declares_2p_4p_mix() -> None:
    mixed = load_yaml("configs/jax_mixed_2p_4p_training.yaml")

    assert mixed["env_backend"] == "jax"
    assert mixed["rl_backend"] == "jax"
    assert mixed["env"]["player_count"] == 2
    assert [entry["player_count"] for entry in mixed["training_format"]["format_mix"]] == [2, 4]
    assert [entry["weight"] for entry in mixed["training_format"]["format_mix"]] == [0.5, 0.5]
    assert mixed["ppo"]["num_envs_2p"] == 4
    assert mixed["ppo"]["num_envs_4p"] == 4


def test_torch_kaggle_mixed_training_config_declares_2p_4p_mix() -> None:
    mixed = load_yaml("configs/mixed_2p_4p_training.yaml")

    assert mixed["env_backend"] == "kaggle"
    assert mixed["rl_backend"] == "torch"
    assert [entry["player_count"] for entry in mixed["training_format"]["format_mix"]] == [2, 4]
    assert mixed["ppo"]["num_envs_2p"] == 4
    assert mixed["ppo"]["num_envs_4p"] == 4


def test_train_config_loads_jax_mixed_format_mix_and_rollout_groups() -> None:
    from src.config import load_train_config

    cfg = load_train_config("configs/jax_mixed_2p_4p_training.yaml")

    assert cfg.env.player_count == 2
    assert [entry["player_count"] for entry in cfg.training_format.format_mix] == [2, 4]
    assert [entry["weight"] for entry in cfg.training_format.format_mix] == [0.5, 0.5]
    assert [group["player_count"] for group in cfg.training_format.rollout_groups] == [2, 4]
    assert cfg.ppo.num_envs_2p == 4
    assert cfg.ppo.num_envs_4p == 4
