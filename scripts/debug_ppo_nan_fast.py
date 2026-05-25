#!/usr/bin/env python3
"""Fast single-update NaN diagnostic mirroring M1 factorized ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

import jax
from src.config import compose_hydra_train_config
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import ppo_update_jax
from src.jax.train import (
    _active_group_indices,
    _init_historical_snapshot_pool,
    init_rollout_groups,
)
from src.jax.train_state import init_train_state
from src.training.curriculum import CurriculumController


def main() -> None:
    pin = json.loads((REPO_ROOT / "artifacts/m1/baseline_pin.json").read_text())
    overrides = [
        "model=planet_graph_transformer_factorized",
        "seed=101",
        "run_name=debug-nan-fast",
        *pin["shared_overrides"],
        "training.total_updates=1",
        "training.log_every=1",
    ]
    cfg = compose_hydra_train_config(overrides)
    print(
        json.dumps(
            {
                "num_envs": cfg.training.num_envs,
                "minibatch_size": cfg.training.minibatch_size,
                "rollout_steps": cfg.training.rollout_steps,
                "lean_rollout_metrics": cfg.training.lean_rollout_metrics,
            }
        ),
        flush=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    _, rollout_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    _, rollout_groups = init_rollout_groups(jax.random.fold_in(key, 1), cfg, policy)
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    pool = _init_historical_snapshot_pool(
        train_state.params, pool_size=cfg.opponents.snapshot.pool_size
    )
    stage_view = curriculum.stage_view(
        1,
        snapshot_ids=pool.snapshot_ids,
        snapshot_valid_mask=pool.valid_mask,
        snapshot_updates=pool.snapshot_updates,
    )
    active = _active_group_indices(
        rollout_groups,
        curriculum.current_format_weights(),
        update=1,
        rotate_format_rollouts=cfg.training.rotate_format_rollouts,
    )
    group = rollout_groups[active[0]]
    print(
        json.dumps(
            {
                "active_indices": active,
                "group_num_envs": group.cfg.training.num_envs,
                "global_num_envs": cfg.training.num_envs,
            }
        ),
        flush=True,
    )

    _, env_state, turn_batch, transitions, rollout_metrics = group.collect_fn(
        rollout_key,
        group.env_state,
        group.turn_batch,
        train_state,
        stage_view,
        pool.params,
        jnp.asarray(1, dtype=jnp.int32),
    )
    tr_shape = transitions.source_index.shape
    log_prob_finite = bool(jax.device_get(jnp.isfinite(transitions.log_prob).all()))
    print(
        json.dumps(
            {
                "transition_shape": list(tr_shape),
                "log_prob_finite": log_prob_finite,
                "env_steps": float(jax.device_get(rollout_metrics["env_steps"])),
            }
        ),
        flush=True,
    )

    update_fn = jax.jit(
        lambda ts, tr: ppo_update_jax(ts, policy, tr, cfg),
    )
    train_state, metrics = update_fn(train_state, transitions)
    metrics_host = jax.device_get(metrics)
    print(
        json.dumps(
            {
                k: float(metrics_host[k])
                for k in (
                    "total_loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "entropy_stop",
                    "entropy_move",
                    "approx_kl",
                )
                if k in metrics_host
            }
        ),
        flush=True,
    )
    finite = bool(
        jax.device_get(
            jnp.isfinite(
                jnp.array(
                    [
                        metrics["total_loss"],
                        metrics["policy_loss"],
                        metrics["entropy"],
                    ]
                )
            )
        ).all()
    )
    print(
        f"total_loss={metrics_host['total_loss']:.6f} "
        f"entropy={metrics_host['entropy']:.6f} finite={finite}",
        flush=True,
    )
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
