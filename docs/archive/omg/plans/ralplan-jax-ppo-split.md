# Ralplan: JAX PPO Module Split

Spec: `.omg/specs/deep-interview-jax-ppo-split.md`
Status: planned
Iterations: 1 (consensus on first pass)

## RALPLAN-DR

### Principles

1. **Dependency-first extraction** — leaf modules (types, train_state) before dependents (builders → sampling → collect).
2. **Behavior preservation** — no semantic changes; identical JIT boundaries and metric keys.
3. **Ownership over convenience** — delete `src/jax/ppo.py`; update import sites explicitly.
4. **Metrics dedup without contract break** — shared schema builder; full vs lean entrypoints preserved.
5. **Verify incrementally** — targeted pytest after module creation, full suite + benchmark at end.

### Decision Drivers

1. Avoid import cycles between `src/opponents/jax_actions/` and `src/jax/rollout/collect.py`.
2. Keep `_sample_shielded_sequence_with_params` callable from collect without circular imports.
3. Preserve all metric keys consumed by `src/jax/train.py` and telemetry.

### Viable Options

| Option | Pros | Cons |
|--------|------|------|
| **A: Bottom-up extraction (chosen)** | Matches spec order; testable after each layer | Longer initial setup |
| **B: Copy-then-trim monolith** | Faster start | Harder to validate boundaries; import mess |
| **C: Temporary re-export shim** | Safer incremental migration | Spec rejects facade |

**Chosen: A** — bottom-up extraction per spec implementation order.

## ADR

**Decision:** Split `src/jax/ppo.py` into `train_state`, `rollout/{types,collect,metrics}`, `ppo_update`, and `opponents/jax_actions/{builders,sampling}`; delete monolith.

**Drivers:** 2,428-line file mixes 5 concerns; approved spec mandates clean import break and metrics unification.

**Alternatives rejected:** Facade shim (spec revision); keeping opponents in jax/ (ownership).

**Consequences:** Import updates in train/tests/scripts/docs; `flatten_batch` lives in `ppo_update` and is imported by builders (one-way dependency).

## Architect Review

**Approved with notes:**
- `flatten_batch` in `ppo_update.py` imported by `builders.py` is acceptable (ppo_update does not import opponents).
- `ShieldedSequenceSample` in `types.py`; builders import from rollout.types.
- Metrics: introduce `_rollout_metric_keys()` + `_base_episode_metrics()` shared helpers; full/lean wrappers call shared builder.

**Risk:** Circular import if `collect.py` re-exported through opponents. Mitigation: opponents never import collect.

## Critic Review

**Approved.** Test plan:

1. `uv run --group dev pytest tests/test_jax_ppo.py tests/test_curriculum.py -q`
2. `uv run --group dev pytest -q`
3. `uv run python scripts/benchmark_jax_rl.py` (smoke / no regression)

Critic checklist:
- [ ] All import sites updated, `ppo.py` deleted
- [ ] Metric key sets identical (full + lean)
- [ ] No import cycles (`python -c` import chain smoke)
- [ ] Targeted + full pytest green

## Implementation Phases

| Phase | Deliverable | Verify |
|-------|-------------|--------|
| 0 | Plan + manifest `executing` | — |
| 1 | `rollout/types.py`, `train_state.py` | import smoke |
| 2 | `opponents/jax_actions/builders.py` | import smoke |
| 3 | `opponents/jax_actions/sampling.py` | import smoke |
| 4 | `rollout/metrics.py` (unified) | unit logic review |
| 5 | `rollout/collect.py` | import smoke |
| 6 | `ppo_update.py` | import smoke |
| 7 | Update imports; delete `ppo.py` | pytest jax_ppo |
| 8 | Docs + full pytest + benchmark | acceptance criteria |
