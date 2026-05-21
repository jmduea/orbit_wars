from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.compare_attention_candidates import DEFAULT_CONFIGS
from src.config import compose_hydra_train_config


def test_root_config_composes_from_responsibility_groups() -> None:
    cfg = compose_hydra_train_config()

    assert cfg.env.candidate_count == 8
    assert cfg.ppo.total_updates == 500
    assert cfg.training_format.rollout_groups
    assert cfg.curriculum.enabled is True
    assert cfg.self_play_enabled is True
    assert cfg.curriculum.snapshot.pool_size == 5
    assert cfg.artifact_pipeline.enabled is True


def test_new_responsibility_overrides_normalize_to_runtime_config() -> None:
    cfg = compose_hydra_train_config(
        [
            "training.total_updates=2",
            "task.candidate_count=12",
            "reward.reward_production_delta=0.01",
            "format=mix_2p_4p_16env",
            "telemetry.wandb.group=capacity",
        ]
    )

    assert cfg.ppo.total_updates == 2
    assert cfg.env.candidate_count == 12
    assert cfg.env.reward_production_delta == 0.01
    assert cfg.training_format.rollout_groups[0]["num_envs"] == 16
    assert cfg.wandb.group == "capacity"


def test_legacy_nested_overrides_still_parse_during_migration() -> None:
    cfg = compose_hydra_train_config(
        [
            "ppo.total_updates=3",
            "env.candidate_count=16",
            "wandb.group=legacy_override",
        ]
    )

    assert cfg.ppo.total_updates == 3
    assert cfg.env.candidate_count == 16
    assert cfg.wandb.group == "legacy_override"


def test_legacy_top_level_self_play_overrides_still_parse_during_migration() -> None:
    cfg = compose_hydra_train_config(
        [
            "curriculum=latest_only",
            "self_play_enabled=false",
            "self_play_pool_size=0",
            "self_play_snapshot_interval=0",
        ]
    )

    assert cfg.self_play_enabled is False
    assert cfg.self_play_pool_size == 0
    assert cfg.self_play_snapshot_interval == 0


def test_different_owner_new_and_legacy_overrides_can_mix() -> None:
    cfg = compose_hydra_train_config(
        ["task.candidate_count=16", "env.reward_production_delta=0.01"]
    )

    assert cfg.env.candidate_count == 16
    assert cfg.env.reward_production_delta == 0.01


@pytest.mark.parametrize(
    ("new_override", "legacy_override"),
    [
        ("training.total_updates=2", "ppo.total_updates=3"),
        ("task.candidate_count=12", "env.candidate_count=16"),
        ("format=mix_2p_4p_16env", "training_format.rollout_groups=[]"),
        ("opponents=latest_only", "self_play_enabled=false"),
        ("telemetry.wandb.group=new", "wandb.group=old"),
        ("artifacts.save_dir=artifacts/new", "save_dir=artifacts/old"),
    ],
)
def test_old_new_owner_override_conflicts_are_rejected(
    new_override: str, legacy_override: str
) -> None:
    with pytest.raises(ValueError, match="Conflicting config overrides"):
        compose_hydra_train_config([new_override, legacy_override])


def test_compare_script_default_configs_compose() -> None:
    for overrides in DEFAULT_CONFIGS.values():
        cfg = compose_hydra_train_config(overrides)
        assert cfg.model.architecture == "attention"
        assert cfg.env.candidate_count in {8, 16, 24}


def test_wandb_sweep_campaign_samples_compose() -> None:
    sweep_dir = Path("conf/sweeps/wandb")
    for path in sorted(sweep_dir.glob("*.yaml")):
        sweep = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        parameters = sweep["parameters"]
        keys = []
        value_sets = []
        for key, spec in parameters.items():
            if "value" in spec:
                values = [spec["value"]]
            else:
                values = list(spec["values"])
            keys.append(key)
            value_sets.append(values)

        for values in product(*value_sets):
            overrides = [
                f"{key}={_hydra_value(value)}"
                for key, value in zip(keys, values, strict=True)
            ]
            cfg = compose_hydra_train_config(overrides)
            assert cfg.wandb.group
            assert cfg.wandb.tags


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)
