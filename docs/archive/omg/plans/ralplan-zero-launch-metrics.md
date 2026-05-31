# Ralplan: Fix Zero Launch Metrics (Shield Off + Continuous Fraction)

**Spec:** `.omg/specs/deep-dive-zero-launch-metrics.md`  
**Trace:** `.omg/specs/deep-dive-trace-zero-launch-metrics.md`  
**Iter:** 1 (planner + architect + critic synthesized)

## RALPLAN-DR Summary

### Principles
1. Shield-off means **no trajectory simulation**, not "no launches allowed."
2. Rollout sampling and PPO replay must share identical legality semantics.
3. Continuous ship mode must never fall through to discrete bucket sampling via shape broadcast.
4. Minimal diff — fix legality masks and metric counting, don't refactor decoder.

### Decision Drivers
1. **Training is broken now** — zero env launches with shield off blocks mixed_train sweep.
2. **Dual ship modes** — fix must handle `continuous_fraction` and discrete buckets.
3. **Regression safety** — shield-on path must remain unchanged.

### Viable Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A (chosen)** | Shield-off passthrough masks + continuous broadcast guard | Small surface; reuses existing mask plumbing | Requires careful mask semantics for both ship modes |
| B | Bypass shield helpers entirely when disabled; sampler uses raw `edge_mask` | Conceptually clean | Larger diff; divergent code paths |
| C | Config workaround: force shield on in sweep | Zero code | Violates user intent (no trajectory shield) |

### ADR

**Decision:** Option A — fix shield-off bucket/source masks and guard continuous ship logit masking.

**Drivers:** User confirmed shield-off = trajectory filter only; local repro pinpoints mask + broadcast bug.

**Alternatives rejected:**
- Option B: broader refactor, higher regression risk for marginal clarity gain.
- Option C: contradicts interview answer.

**Consequences:**
- Shield-off training will produce launches and non-zero utilization metrics.
- Must add tests for shield-off continuous and shield-on parity.

---

## Implementation Plan

### Phase 0 — Legality masks (shield off)

**File:** `src/game/trajectory_shield.py`

In `apply_trajectory_shield_factorized_topk` disabled branch:
- Replace noop-only `default_bucket_mask[..., 0] = True` with full legality:
  - **Discrete buckets:** allow buckets `1..bucket_count-1` wherever `batch.edge_mask` has legal edges (or all non-zero buckets when shield off — match edge legality per slot).
  - **Continuous fraction:** ship bucket mask is irrelevant to launch validity; keep mask permissive but fix `factorized_source_mask_from_shield` to not require `buckets > 0` when `ship_action_mode=continuous_fraction`.

Update `factorized_source_mask_from_shield`:
- Add optional `env_cfg` or explicit `continuous_fraction` flag.
- When continuous: `(planet_ships > 0) & shielded_edge_mask.any(axis=-1)` per source row.
- When discrete + shield off: require non-noop bucket availability as today.

### Phase 1 — Sampler broadcast guard

**File:** `src/opponents/jax_actions/builders.py` — `_sample_factored_step_from_logits`

When `selected_ship_logits.shape[-1] == 1` (continuous):
- Do **not** apply per-bucket `jnp.where` that broadcasts to `(bucket_count,)`.
- Gate continuous ship logit with scalar legality, e.g. `target_mask.any()` or `selected_bucket_mask.any()`.

When discrete: keep existing per-bucket masking.

### Phase 2 — PPO replay parity

**File:** `src/jax/action_codec.py` — `_factored_step_log_prob_entropy`

Mirror Phase 1 continuous guard so replay matches rollout math.

Pass `ship_fraction` through replay for continuous log-prob if not already on shield-off path (verify existing path).

### Phase 3 — Launch metrics

**File:** `src/jax/rollout/metrics.py` — `_apply_factorized_metrics`

Count launch steps as:
```python
non_stop = active * (1.0 - stop_flag)
if continuous:  # detect via data shape or cfg passed in
    launch_sum = (non_stop * (ship_fraction > 0)).sum()
else:
    launch_sum = (non_stop * (ship_bucket > 0)).sum()
```

Thread `cfg` into `_apply_factorized_metrics` if needed for mode detection (prefer checking `ship_fraction is not None` in data).

### Phase 4 — Tests

1. **`tests/test_trajectory_shield_factorized.py`**
   - Update `test_factorized_shield_disabled_returns_all_legal` to assert non-noop buckets or continuous-compatible source mask.
   - Replace/adjust `test_factorized_source_mask_requires_ships_and_buckets` — with ships + edges, source mask should be True when shield off.

2. **New test** (e.g. `tests/test_factorized_launch_metrics.py`, CPU-only if possible):
   - Shield off + continuous_fraction rollout → `mean_active_launches_per_turn > 0` over a few seeds OR assert `ship_fraction > 0` on some non-stop steps.
   - Mark `@pytest.mark.jax` if JIT required; keep lightweight (small env count, 1-2 rollout steps).

3. **Broadcast unit test** (pure numpy/jax, no JIT):
   - Assert continuous ship logit `(1,)` + bucket mask `(8,)` does not expand to discrete path after fix.

### Phase 5 — Verification

- `make test-fast` (must pass)
- Optional local repro script: shield-off rollout shows non-zero launches
- Confirm shield-on tests in `test_trajectory_shield_factorized.py` still pass

---

## Risk / Pre-mortem

| Risk | Mitigation |
|------|------------|
| Shield-on regression | Keep disabled branch isolated; run existing shield factorized tests |
| Continuous/discrete divergence | Shared helper for "is continuous ship head" check in sampler + replay |
| Metric false positives | Require `non_stop` AND fraction/bucket > 0 |

---

## Critic Checklist (iter-1)

- [x] Scope bounded to identified root cause
- [x] Acceptance criteria testable
- [x] Rollout + replay parity explicit
- [x] No sweep config change required
- [ ] Execution evidence pending

## Execution Recommendation

Proceed via **omg-autopilot** after approval: implement phases 0-4, run `make test-fast`.
