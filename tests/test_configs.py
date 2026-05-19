from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from src.config import train_config_from_omegaconf

# Canonical policy: all experiment editing/sweeping happens in conf/.


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compose_train_config(*overrides: str) -> dict:
    conf_dir = Path("conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg_raw = compose(config_name="config", overrides=list(overrides))
    cfg_data = OmegaConf.to_container(cfg_raw, resolve=True)
    assert isinstance(cfg_data, dict)
    cfg_data.pop("experiment", None)
    cfg = train_config_from_omegaconf(OmegaConf.create(cfg_data))
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)


def test_attention_shaped_reward_config_matches_unshaped_ppo_budget() -> None:
    shaped = compose_train_config("+experiment=attention_shaped_reward")
    unshaped = compose_train_config("+experiment=attention_training")
    assert shaped["ppo"] == unshaped["ppo"]


def test_attention_shaped_reward_config_has_positive_terminal_shaping() -> None:
    shaped = compose_train_config("+experiment=attention_shaped_reward")
    assert shaped["env"]["reward_capture_planet"] > 0.0
    assert shaped["env"]["reward_terminal_scale"] == 1.0


def test_default_config_uses_jax_training_path() -> None:
    default = load_yaml("default_cfg.yaml")
    assert default["env_backend"] == "jax"
    assert default["rl_backend"] == "jax"


def test_critical_defaults_remain_stable() -> None:
    cfg = compose_train_config()
    assert cfg["seed"] == 42
    assert cfg["env_backend"] == "jax"
    assert cfg["rl_backend"] == "jax"


def test_preset_override_is_not_a_supported_public_selector() -> None:
    conf_dir = Path("conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        with pytest.raises(ConfigCompositionException, match="Could not override 'preset'"):
            compose(config_name="config", overrides=["preset=jax"])


def test_train_config_rejects_conflicting_rollout_group_locations() -> None:
    raw = OmegaConf.create(
        {
            "training_format": {"rollout_groups": [{"player_count": 2, "num_envs": 4}]},
            "ppo": {"rollout_groups": [{"player_count": 4, "num_envs": 4}]},
        }
    )
    with pytest.raises(ValueError, match="training_format.rollout_groups"):
        train_config_from_omegaconf(raw)


def test_train_config_rejects_conflicting_phase_locations() -> None:
    raw = OmegaConf.create(
        {
            "training_format": {"phases": [{"start_update": 0, "format_mix": {"2": 1.0}}]},
            "ppo": {"phases": [{"start_update": 100}]},
        }
    )
    with pytest.raises(ValueError, match="training_format.phases"):
        train_config_from_omegaconf(raw)


def test_train_config_rejects_legacy_num_env_format_knobs() -> None:
    raw = OmegaConf.create({"ppo": {"num_envs_2p": 4, "num_envs_4p": 4}})
    with pytest.raises(ValueError, match="training_format.rollout_groups"):
        train_config_from_omegaconf(raw)
