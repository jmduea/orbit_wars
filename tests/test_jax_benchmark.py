from __future__ import annotations

from src.config import compose_hydra_train_config
from src.jax.benchmark import (
    ProductionBenchmarkResult,
    production_benchmark_payload,
    rollout_group_summary,
)


def test_rollout_group_summary_uses_training_derived_env_counts() -> None:
    cfg = compose_hydra_train_config(["training=2p4p_32_split"])
    summary = rollout_group_summary(cfg)

    assert len(summary) == 2
    assert all(int(group["num_envs"]) == 16 for group in summary)
    assert {int(group["player_count"]) for group in summary} == {2, 4}


def test_production_benchmark_payload_includes_group_metadata() -> None:
    cfg = compose_hydra_train_config(["training=2p4p_32_split"])
    groups = rollout_group_summary(cfg)
    payload = production_benchmark_payload(
        ProductionBenchmarkResult(
            seconds=1.0,
            env_steps=100,
            samples=200,
            updates=3,
            warmup=1,
            rollout_steps=128,
            rollout_microbatch_envs=8,
            total_envs=32,
            rollout_groups=groups,
        )
    )

    assert payload["backend"] == "jax_rl_production"
    assert payload["total_envs"] == 32
    assert payload["rollout_groups"] == [dict(group) for group in groups]
    assert payload["samples_per_sec"] == 200.0
