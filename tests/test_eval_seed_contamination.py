"""SSOT train/eval seed partition guards (AE6)."""

from __future__ import annotations

import pytest

from src.config import compose_hydra_train_config


def test_eval_seed_overlap_with_training_seed_set_fails() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        compose_hydra_train_config(
            [
                "training_seed_set=[10,20]",
                "eval_seed_set=[20,30]",
            ]
        )


def test_heldout_eval_seed_set_rejected() -> None:
    with pytest.raises(ValueError, match="heldout_eval_seed_set is removed"):
        compose_hydra_train_config(["heldout_eval_seed_set=[1,2,3]"])


def test_training_seed_must_not_equal_eval_seed() -> None:
    with pytest.raises(ValueError, match="training.seed must not appear"):
        compose_hydra_train_config(["seed=486"])


def test_default_training_and_eval_seed_sets_are_disjoint() -> None:
    cfg = compose_hydra_train_config([])
    assert cfg.training_seed_set not in cfg.eval_seed_set
