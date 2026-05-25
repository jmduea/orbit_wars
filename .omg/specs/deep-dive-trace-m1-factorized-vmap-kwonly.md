# Deep Dive Trace: M1 factorized rollout vmap crash

**Slug:** `m1-factorized-vmap-kwonly`  
**Status:** converged  
**Trigger:** M1 Phase 4 ablation — `run_m1_ablation.py` factorized arm crashes on first rollout collect

## Problem

```
TypeError: _sample_factored_step_from_logits() takes 0 positional arguments but 9 were given
```

Stack: `collect_rollout_jax` → `_sample_shielded_factored_sequence_with_params` → `sequence_scan_body` → `jax.vmap(_sample_factored_step_from_logits, ...)`.

## Lane hypotheses

| Lane | Hypothesis | Verdict |
|------|------------|---------|
| **Code-path** | `_sample_factored_step_from_logits` is keyword-only (`*,`) but passed positional args via `jax.vmap` | **Confirmed** |
| **Config/env** | Transformer + factorized combo triggers a different code path than GNN tests | Partial — path is factorized decoder, not encoder-specific |
| **Measurement** | Fast tests never vmapped the rollout sampler; only policy forward / action-build tested | **Confirmed** — gap in `test_factored_action_builders.py` |

## Evidence

1. `builders.py:363` defines `_sample_factored_step_from_logits(*, key, ...)` — Python keyword-only.
2. `builders.py:551-564` calls `jax.vmap(_sample_factored_step_from_logits, in_axes=...)(...)` with 9 positional tensors.
3. Joint-flat path calls `_sample_step_from_logits(...)` with **keyword** args at line ~800 — no vmap, so kw-only there is fine.
4. `test_jax_policy_factorized_decoder.py` exercises policy forward only; no shielded sequence sampling.

## Root cause

Two bugs in `_sample_factored_step_from_logits`, exposed only when the rollout path vmapped the helper over envs (M1 Phase 4 ablation):

1. **Keyword-only signature:** `jax.vmap` passes positional args; `*,` rejected all 9 → `TypeError: takes 0 positional arguments but 9 were given`.
2. **Per-env indexing:** Helper was written for a batched `(batch, P, …)` layout but vmap supplies `(P, …)` slices. Leftover `batch_indices = arange(P)` and `source[:, None]` indexing broke; `jnp.where(row_bucket_mask, ship_logits[target_slot])` broadcast `(k, buckets)` with `(buckets,)` → `ship_lp` shape `(k,)` instead of scalar.

Joint-flat rollout calls `_sample_step_from_logits(..., key=…)` with keywords (no vmap) — fast tests never hit the factored vmap path.

## Fix

1. Remove `*` from `_sample_factored_step_from_logits` (positional args for vmap).
2. Use per-env indexing: `ship_bucket_mask[source]`, `jnp.take` for log-probs, `row_bucket_mask[target_slot]` before ship `where`.
3. Scalar `has_launch`: `.any()` without planet axis.
4. Add `tests/test_factored_step_vmap.py` regression.

## Discriminating probe (post-fix)

```bash
uv run --group dev pytest tests/test_factored_step_vmap.py -m jax
# Then resume ablation:
uv run python scripts/run_m1_ablation.py --skip-existing --arms factorized
```

## Interview injection (Phase 4 bridge)

- **What:** Unblock M1 Phase 4 GPU ablation; no spec change to decode semantics.
- **Non-goals:** Refactoring joint-flat sampler; slow-tier full rollout smoke in CI (optional pre-merge).
- **Open:** Consider one lightweight `@pytest.mark.jax` collect smoke for factorized decoder before merge.
