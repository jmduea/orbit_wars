from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import yaml

from src.config import TrainConfig


def _expected_schema(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _expected_schema(getattr(value, f.name)) for f in fields(value)}
    return None


def _assert_schema_keys(actual: Any, expected: Any, path: str = "") -> None:
    if not isinstance(expected, dict):
        return
    assert isinstance(actual, dict), f"Expected mapping at '{path or '<root>'}'"

    actual_keys = set(actual)
    expected_keys = set(expected)
    assert actual_keys == expected_keys, (
        f"Key mismatch at '{path or '<root>'}': "
        f"missing={sorted(expected_keys - actual_keys)} "
        f"extra={sorted(actual_keys - expected_keys)}"
    )

    for key in expected_keys:
        child_path = f"{path}.{key}" if path else key
        _assert_schema_keys(actual[key], expected[key], child_path)


def test_default_cfg_yaml_contains_full_train_config_schema() -> None:
    loaded = yaml.safe_load(open("default_cfg.yaml", encoding="utf-8"))
    expected = _expected_schema(TrainConfig())
    _assert_schema_keys(loaded, expected)
