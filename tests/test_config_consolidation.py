from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import compose_hydra_train_config

SWEEP_COMPOSE_RECIPES = ("budget", "2p_only_throughput", "4p_only_throughput")

LAUNCH_RECIPES: dict[str, list[str]] = {
    "smoke": [
        "training=smoke",
        "format=2p_16env",
        "curriculum=off",
        "opponents=noop_only",
        "telemetry=throughput_only",
        "artifacts=disabled",
    ],
    "shield_cheap": ["task=shield_cheap", "telemetry=default"],
}


def test_root_config_composes_from_responsibility_groups() -> None:
    cfg = compose_hydra_train_config()

    assert cfg.task.candidate_count == 6
    assert cfg.training.total_updates == 100
    assert cfg.format.rollout_groups
    assert cfg.curriculum.enabled is True
    assert len(cfg.curriculum.stages) == 1
    assert cfg.curriculum.stages[0]["id"] == "sp_2p"
    assert cfg.opponents.self_play.enabled is True
    assert cfg.opponents.snapshot.pool_size == 2
    assert cfg.artifacts.artifact_pipeline.enabled is True
    assert cfg.output.root == "outputs"
    assert cfg.output.campaign == "default"
    assert cfg.artifacts.artifact_pipeline.queue_dir == "queue/optional_jobs"
    assert cfg.artifacts.artifact_pipeline.result_dir == "evaluations"
    assert not hasattr(cfg, "env")
    assert not hasattr(cfg, "ppo")
    assert not hasattr(cfg, "save_dir")


def test_new_responsibility_overrides_compose_to_canonical_runtime_config() -> None:
    cfg = compose_hydra_train_config(
        [
            "training.total_updates=2",
            "task.candidate_count=12",
            "reward.reward_production_delta=0.01",
            "format=2p_4p_16env",
            "telemetry.wandb.group=capacity",
        ]
    )

    assert cfg.training.total_updates == 2
    assert cfg.task.candidate_count == 12
    assert cfg.reward.reward_production_delta == 0.01
    assert cfg.format.rollout_groups[0]["num_envs"] == 16
    assert cfg.telemetry.wandb.group == "capacity"


@pytest.mark.parametrize(
    "legacy_override",
    [
        "ppo.total_updates=3",
        "env.candidate_count=16",
        "wandb.group=legacy_override",
        "training_format.rollout_groups=[]",
        "self_play_enabled=false",
        "self_play_pool_size=0",
        "self_play_snapshot_interval=0",
        "save_dir=artifacts/old",
    ],
)
def test_legacy_overrides_are_rejected(legacy_override: str) -> None:
    with pytest.raises(Exception):
        compose_hydra_train_config([legacy_override])


@pytest.mark.parametrize("name,overrides", LAUNCH_RECIPES.items())
def test_launch_recipe_composes(name: str, overrides: list[str]) -> None:
    del name
    cfg = compose_hydra_train_config(overrides)
    assert cfg.model.architecture == "planet_graph_transformer"


def test_output_campaign_slug_is_validated() -> None:
    with pytest.raises(ValueError, match="output.campaign"):
        compose_hydra_train_config(["output.campaign='bad campaign'"])


def test_output_paths_must_be_relative() -> None:
    with pytest.raises(ValueError, match="output.wandb_dir"):
        compose_hydra_train_config(["output.wandb_dir=/tmp/wandb"])

    
    
@pytest.mark.parametrize(
    "override",
    [
        "output.run_id=../escape",
        "output.root=../outputs",
        "output.wandb_dir=../wandb",
        "artifacts.artifact_pipeline.queue_dir=../jobs",
        "artifacts.artifact_pipeline.result_dir=../evals",
    ],
)
def test_output_paths_reject_traversal(override: str) -> None:
    with pytest.raises(ValueError, match="\.\.|run_id"):
        compose_hydra_train_config([override])

def test_wandb_sweep_yaml_smoke_compose() -> None:
    for overrides in _iter_sweep_compose_cases(full_grid=False):
        cfg = compose_hydra_train_config(overrides)
        assert cfg.telemetry.wandb.group
        assert cfg.telemetry.wandb.tags


@pytest.mark.slow
def test_wandb_sweep_campaign_samples_compose_full() -> None:
    for overrides in _iter_sweep_compose_cases(full_grid=True):
        cfg = compose_hydra_train_config(overrides)
        assert cfg.telemetry.wandb.group
        assert cfg.telemetry.wandb.tags


def _iter_sweep_compose_cases(*, full_grid: bool):
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    for recipe in SWEEP_COMPOSE_RECIPES:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            sweep = OmegaConf.to_container(
                compose(
                    config_name="sweep_gen",
                    overrides=[f"wandb_sweep={recipe}"],
                ),
                resolve=True,
            )
        GlobalHydra.instance().clear()
        parameters = sweep["parameters"]
        keys = []
        value_sets = []
        for key, spec in parameters.items():
            if "value" in spec:
                values = [spec["value"]]
            elif "values" in spec:
                values = list(spec["values"])
            elif "distribution" in spec:
                # W&B bayes/uniform sweeps are not grid-enumerable; smoke with min.
                values = [spec["min"]]
            else:
                raise KeyError(
                    f"Unsupported sweep parameter spec for {key!r}: {spec!r}"
                )
            keys.append(key)
            value_sets.append(values)

        if full_grid:
            value_products = product(*value_sets)
        else:
            value_products = [tuple(values[0] for values in value_sets)]

        for values in value_products:
            yield [
                f"{key}={_hydra_value(value)}"
                for key, value in zip(keys, values, strict=True)
            ]


def test_baseline_sweep_scaffolding_is_discoverable() -> None:
    fixed_path = Path("conf/wandb_sweep/fixed/kaggle_runner_mvp.yaml")
    assert fixed_path.is_file()


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)
