import yaml

from scripts.generate_default_cfg import render_default_cfg
from src.config import TrainConfig


def test_default_cfg_yaml_matches_train_config_defaults() -> None:
    committed = open("default_cfg.yaml", encoding="utf-8").read()
    assert committed == render_default_cfg()


def test_generated_template_contains_all_top_level_train_config_fields() -> None:
    generated = yaml.safe_load(render_default_cfg())
    assert set(generated) == set(TrainConfig.__dataclass_fields__)
