# Residual review findings — optimize-exp/opponent-rollout-throughput/exp-001

Source: ce-code-review after H3+H4 implementation (2026-06-06).

## Deferred (out of scope for this PR)

- **P2** `src/opponents/jax_actions/sampling.py` — Historical snapshot pool still `vmap`s full `pool_size` neural samples per pick (H1 stripped intentionally).
- **P2** `src/opponents/jax_actions/sampling.py` — Historical branch runs throwaway `latest_branch` before pool gather (H2 stripped).
- **P2** `src/jax/action_sampling.py` — JIT `collect_fn` may duplicate K-step scan bodies for learner vs `inference_only` opponent.
- **P2** `src/jax/shield/trajectory.py` — Pointwise cheap still builds full sun-anchor tensors per K-step; hoist opportunity remains.
- **P2** `src/jax/action_sampling.py` — Non-factorized decoder path ignores `inference_only`.
- **P3** `src/jax/action_sampling.py` — `inference_only` kwarg exposed on learner entrypoints; consider restricting to opponent wrappers only.

## Throughput note

Quick ladder profiles showed ~2–3% opponent-fraction drop vs baseline; below 10% keep bar. Full-geometry confirmation and `factorized_decode_step` port are follow-ups.
