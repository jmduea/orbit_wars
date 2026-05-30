from __future__ import annotations

import pytest

from src.config import compose_hydra_train_config, config_from_plain
from src.config.rollout_allocation import (
    allocate_split,
    normalize_format_weights,
    resolve_rollout_group_specs,
    run_name_env_count,
)


def test_allocate_split_even_weights() -> None:
    weights = normalize_format_weights({2: 0.5, 4: 0.5})
    assert allocate_split(32, weights) == {2: 16, 4: 16}


def test_allocate_split_remainder_tie_breaks_to_lower_player_count() -> None:
    weights = normalize_format_weights({2: 0.5, 4: 0.5})
    assert allocate_split(33, weights) == {2: 17, 4: 16}


def test_per_group_mode_when_rotating() -> None:
    cfg = compose_hydra_train_config(["training=2p4p_16_rotate"])
    specs = resolve_rollout_group_specs(cfg)
    assert len(specs) == 2
    assert all(spec.num_envs == 16 for spec in specs)
    assert run_name_env_count(cfg) == 16


def test_split_mode_total_env_count() -> None:
    cfg = compose_hydra_train_config(["training=2p4p_32_split"])
    specs = resolve_rollout_group_specs(cfg)
    assert sum(spec.num_envs for spec in specs) == 32
    assert run_name_env_count(cfg) == 32


def test_single_format_infers_from_task_when_weights_empty() -> None:
    cfg = config_from_plain(
        {
            "task": {"player_count": 2},
            "training": {"num_envs": 8, "format_weights": {}},
        }
    )
    specs = resolve_rollout_group_specs(cfg)
    assert len(specs) == 1
    assert specs[0].player_count == 2
    assert specs[0].num_envs == 8


def test_split_mode_rejects_too_few_envs_for_mixed_formats() -> None:
    with pytest.raises(ValueError, match="too small for split mode"):
        compose_hydra_train_config(["training=2p4p_16_split", "training.num_envs=1"])


def test_legacy_format_override_is_rejected() -> None:
    with pytest.raises(Exception):
        compose_hydra_train_config(["format=2p_16env"])
