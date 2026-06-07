from __future__ import annotations

import jax.numpy as jnp
import pytest

from src.config import TrainConfig
from src.jax.rollout.metrics import rollout_metrics
from src.jax.train.metrics import finalize_cross_chunk_rate_metrics


def _minimal_finalize_metrics(**overrides: jnp.ndarray) -> dict[str, jax.Array]:
    base = {
        "episodes_2p": jnp.array(0.0),
        "episodes_4p": jnp.array(0.0),
        "wins_2p": jnp.array(0.0),
        "first_places_4p": jnp.array(0.0),
        "placement_4p_sum": jnp.array(0.0),
        "episode_done": jnp.array(0.0),
        "survival_time_sum": jnp.array(0.0),
        "score_share_sum": jnp.array(0.0),
    }
    base.update(overrides)
    return finalize_cross_chunk_rate_metrics(base)


def test_finalize_mean_ships_per_launch_from_launch_sums() -> None:
    metrics = _minimal_finalize_metrics(
        launch_ship_count_sum=jnp.array(30.0),
        active_launch_count=jnp.array(10.0),
    )
    assert float(metrics["mean_ships_per_launch"]) == pytest.approx(3.0)


def test_finalize_mean_ships_per_launch_is_zero_without_launches() -> None:
    metrics = _minimal_finalize_metrics(
        launch_ship_count_sum=jnp.array(0.0),
        active_launch_count=jnp.array(0.0),
    )
    assert float(metrics["mean_ships_per_launch"]) == 0.0


def test_rollout_launch_sizing_from_factorized_sequence() -> None:
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.debug = True
    cfg.task.ship_action_mode = "continuous_fraction"

    data = {
        "reward": jnp.zeros((2, 1)),
        "done": jnp.zeros((2, 1), dtype=bool),
        "value": jnp.zeros((2, 1)),
        "terminal_is_first": jnp.zeros((2, 1), dtype=bool),
        "terminal_placement": jnp.zeros((2, 1)),
        "terminal_survival_time": jnp.zeros((2, 1)),
        "terminal_score_share": jnp.zeros((2, 1)),
        "terminal_ship_differential": jnp.zeros((2, 1)),
        "target_index": jnp.zeros((2, 1, 1), dtype=jnp.int32),
        "stop_flag": jnp.array([[[0.0]], [[0.0]]], dtype=jnp.float32),
        "step_mask": jnp.ones((2, 1, 1), dtype=jnp.float32),
        "ship_bucket": jnp.ones((2, 1, 1), dtype=jnp.int32),
        "source_index": jnp.zeros((2, 1, 1), dtype=jnp.int32),
        "initial_planet_ships": jnp.array(
            [[[20.0] + [0.0] * 63], [[20.0] + [0.0] * 63]], dtype=jnp.float32
        ),
        "ship_fraction": jnp.array([[[0.5]], [[0.25]]], dtype=jnp.float32),
    }

    metrics = rollout_metrics(data=data, cfg=cfg, env_count=1)
    finalized = finalize_cross_chunk_rate_metrics(metrics)

    assert float(finalized["active_launch_count"]) == pytest.approx(2.0)
    assert float(finalized["launch_ship_count_sum"]) == pytest.approx(15.0)
    assert float(finalized["mean_ships_per_launch"]) == pytest.approx(7.5)


def test_rollout_launch_sizing_omitted_without_debug_group() -> None:
    cfg = TrainConfig()
    cfg.telemetry.metric_groups.debug = False

    data = {
        "reward": jnp.zeros((1, 1)),
        "done": jnp.zeros((1, 1), dtype=bool),
        "value": jnp.zeros((1, 1)),
        "terminal_is_first": jnp.zeros((1, 1), dtype=bool),
        "terminal_placement": jnp.zeros((1, 1)),
        "terminal_survival_time": jnp.zeros((1, 1)),
        "terminal_score_share": jnp.zeros((1, 1)),
        "terminal_ship_differential": jnp.zeros((1, 1)),
        "target_index": jnp.zeros((1, 1, 1), dtype=jnp.int32),
    }

    metrics = rollout_metrics(data=data, cfg=cfg, env_count=1)

    assert "launch_ship_count_sum" not in metrics
    assert "mean_ships_per_launch" not in metrics
