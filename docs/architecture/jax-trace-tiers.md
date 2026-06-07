# JAX trace tiers

Orbit Wars compiles training rollouts and PPO updates with `jax.jit`. Code on the traced path must not use Python host semantics (I/O, logging, `.item()`, mutable globals) unless explicitly bridged.

## Tiers

| Tier | Role | Examples |
|------|------|----------|
| **A — must trace** | Called from `jax.jit` / `jax.vmap` / `jax.lax.scan` in production | `src/jax/env.py`, `src/jax/features.py`, `src/jax/rollout/collect.py`, `src/jax/ppo_update.py`, `src/jax/action_sampling.py`, `src/jax/factored_sequence_scan.py`, `src/jax/planet_flow.py`, `src/jax/action_codec.py` |
| **B — jit wrappers** | Builds or wraps tier-A functions | `src/jax/train/rollout_groups.py` (`collect_fn`), `src/jax/train/loop.py` (`update_fn`), `src/jax/submission_runtime.py` |
| **C — host orchestration** | Filesystem, W&B, checkpoints, Hydra shell | `src/jax/train/loop.py` (Python loop body), `src/jax/train/checkpoint.py`, artifact/tournament helpers |

Tier **A** must not import `src.telemetry` or `src.artifacts` except frozen debt documented in `tests/test_jax_trace_hygiene.py`. Tier **A** must not use `jax.pure_callback`, Kaggle reference generators, or `env_parity_mode` (see [jax-no-kaggle-callbacks](../solutions/conventions/jax-no-kaggle-callbacks.md)).

## Production entrypoints

- Rollout: `JaxRolloutGroup.collect_fn` — `jax.jit` in `src/jax/train/rollout_groups.py`
- PPO: `update_fn` — `jax.jit(lambda ts, tr: ppo_update_jax(...))` in `src/jax/train/loop.py`
- Env: `batched_reset` / `batched_step` — `jax.vmap` in `src/jax/env.py`

## Verify locally

```bash
make test-jax-trace-hygiene
```

Static gate only:

```bash
./scripts/jax_trace_hygiene.sh
```

## Debugging tracer leaks

```bash
JAX_CHECK_TRACER_LEAKS=1 uv run --group dev pytest tests/test_jax_trace_hygiene.py -m jax -v
```

Use `jax.debug.print` inside `jit`, not `print`. Agent rules: `.cursor/rules/jax-flax-linen.mdc`.
