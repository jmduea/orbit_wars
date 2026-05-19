from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import train_config_from_omegaconf


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compose_train_config(*overrides: str) -> dict:
    conf_dir = Path("conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg_raw = compose(config_name="config", overrides=list(overrides))
    cfg = train_config_from_omegaconf(cfg_raw)
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)


def _assert_configs_equivalent(actual: dict, expected: dict) -> None:
    assert actual == expected


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


def test_critical_legacy_defaults_remain_stable() -> None:
    cfg = compose_train_config()

    assert cfg["seed"] == 42
    assert cfg["env_backend"] == "kaggle"
    assert cfg["rl_backend"] == "torch"
    assert cfg["opponent"] == "random"
    assert cfg["env"]["candidate_count"] == 8
    assert cfg["env"]["ship_bucket_count"] == 8
    assert cfg["env"]["feature_history_steps"] == 1
    assert cfg["ppo"]["rollout_steps"] == 32
    assert cfg["ppo"]["num_envs"] == 4
    assert cfg["ppo"]["total_updates"] == 200
    assert cfg["print_resolved_config"] is False


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
    assert shaped["ppo"]["num_envs"] == 32
    assert shaped["ppo"]["num_envs_2p"] == 16
    assert shaped["ppo"]["num_envs_4p"] == 16
    assert shaped["ppo"]["rollout_steps"] == 500
    assert shaped["ppo"]["total_updates"] == unshaped["ppo"]["total_updates"]
    assert shaped["ppo"]["ent_coef"] == unshaped["ppo"]["ent_coef"]
    assert [
        group["num_envs"] for group in shaped["training_format"]["rollout_groups"]
    ] == [16, 16]
    assert shaped["env"]["reward_capture_planet"] == 0.1
    assert shaped["env"]["reward_ship_delta"] == 0.001
    assert shaped["env"]["reward_production_delta"] == 0.02
    assert shaped["env"]["reward_terminal_scale"] == 1.0


def test_composed_jax_experiment_resolves_expected_backend_model_and_opponent() -> None:
    cfg = compose_train_config("experiment=jax_self_play_shaped_reward")

    assert cfg["env_backend"] == "jax"
    assert cfg["rl_backend"] == "jax"
    assert cfg["model"]["architecture"] == "attention"
    assert cfg["opponent"] == "self"
    assert cfg["self_play_enabled"] is True
    assert cfg["ppo"]["num_envs"] == 32
    assert cfg["ppo"]["num_envs_2p"] == 16
    assert cfg["ppo"]["num_envs_4p"] == 16


def test_jax_mixed_training_config_declares_2p_4p_mix() -> None:
    mixed = load_yaml("configs/jax_mixed_2p_4p_training.yaml")

    assert mixed["env_backend"] == "jax"
    assert mixed["rl_backend"] == "jax"
    assert mixed["env"]["player_count"] == 2
    assert [
        entry["player_count"] for entry in mixed["training_format"]["format_mix"]
    ] == [2, 4]
    assert [entry["weight"] for entry in mixed["training_format"]["format_mix"]] == [
        0.5,
        0.5,
    ]
    assert mixed["ppo"]["num_envs_2p"] == 16
    assert mixed["ppo"]["num_envs_4p"] == 16
    assert [
        group["num_envs"] for group in mixed["training_format"]["rollout_groups"]
    ] == [16, 16]


def test_torch_kaggle_mixed_training_config_declares_2p_4p_mix() -> None:
    mixed = load_yaml("configs/mixed_2p_4p_training.yaml")

    assert mixed["env_backend"] == "kaggle"
    assert mixed["rl_backend"] == "torch"
    assert [
        entry["player_count"] for entry in mixed["training_format"]["format_mix"]
    ] == [2, 4]
    assert mixed["ppo"]["num_envs_2p"] == 4
    assert mixed["ppo"]["num_envs_4p"] == 4
    assert [
        group["num_envs"] for group in mixed["training_format"]["rollout_groups"]
    ] == [4, 4]


def test_legacy_yaml_equivalence_for_key_presets() -> None:
    preset_to_yaml = {
        "torch_kaggle": "default_cfg.yaml",
        "attention": "configs/attention_training.yaml",
        "jax": "configs/jax_training.yaml",
        "mixed_2p_4p": "configs/mixed_2p_4p_training.yaml",
    }
    for preset, yaml_path in preset_to_yaml.items():
        composed = compose_train_config(f"preset={preset}")
        legacy = load_yaml(yaml_path)
        _assert_configs_equivalent(composed, legacy)


def test_train_config_loads_replay_overrides() -> None:
    from src.config import train_config_from_dict

    cfg = train_config_from_dict(
        {
            "replay": {
                "enabled": True,
                "every_n_checkpoints": 2,
                "opponent": "sniper",
                "seed_policy": "fixed",
                "max_steps": 300,
                "output_dir": "checkpoint_replays",
            }
        }
    )

    assert cfg.replay.enabled is True
    assert cfg.replay.every_n_checkpoints == 2
    assert cfg.replay.opponent == "sniper"
    assert cfg.replay.seed_policy == "fixed"
    assert cfg.replay.max_steps == 300
    assert cfg.replay.output_dir == "checkpoint_replays"
