---
title: JAX training path must not use Kaggle callbacks or reference helpers
date: 2026-06-04
category: conventions
module: jax-env
tags:
  - jax-env
  - kaggle-parity
  - pure-callback
  - training-loop
problem_type: convention
---

# JAX training path: no Kaggle callbacks

## Rule

Inside `src/jax/` (especially `src/jax/env.py` and rollout collect):

- **No** `jax.pure_callback`, `io_callback`, or host Python generators in `reset` / `step` / `vmap` / `jit` rollouts.
- **No** `_reference_*` helpers that call `src/game/planet_generation.py` or `src/game/comet_generation.py`.
- **No** `env_parity_mode`, `task=kaggle_parity`, or parallel train/kaggle env paths.
- **No** `task.env_parity_mode` or Hydra presets that re-enable callbacks.

Kaggle alignment is **pure JAX** in `src/jax/planet_generation.py` and `src/jax/comet_generation.py`, validated in tests by comparing to `src/game/*` **from test code only**, never from the hot path.

## Wrong vs right

| Wrong | Right |
|-------|--------|
| `_reference_planet_tables` + `pure_callback` in `reset` | JAX planet layout in `src/jax/planet_generation.py` |
| `_reference_comet_paths` + `pure_callback` in spawn | `src/jax/comet_generation.py` |
| `env_parity_mode=kaggle` for “real” training | One env; parity = JAX matches reference in CI |

## Verify

```bash
rg 'pure_callback|_reference_|env_parity_mode|from src\.game\.(planet|comet)_generation' src/jax/
```

Expect no matches in env/rollout hot path.

## Trace tiers (broader hygiene)

Tier definitions, jit contract tests, and CI: `docs/architecture/jax-trace-tiers.md` — run `make test-jax-trace-hygiene`.

## Related

- Plan: `docs/plans/2026-06-04-006-feat-pure-jax-env-parity-plan.md`
- Plan: `docs/plans/2026-06-04-008-feat-jax-trace-hygiene-plan.md`
- Supersedes callback guidance in `docs/solutions/architecture-patterns/jax-comet-kaggle-parity-ci-gate.md` (update when JAX ports land)
