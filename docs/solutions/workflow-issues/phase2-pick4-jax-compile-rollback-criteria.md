---
title: "Phase 2 pick #4 — JAX compile cliff rollback and mechanical parity gates"
date: 2026-06-06
category: workflow-issues
module: jax-env
problem_type: workflow_issue
component: development_workflow
severity: high
applies_when:
  - "Greenfield pure JAX planet/comet generation on orbit_wars-integration (pick #4)"
  - "Mechanical fidelity fixes pass make test-kaggle-parity but smoke or benchmark compile exceeds operator tolerance"
  - "Evaluating whether to roll back integration worktree after parity-green but compile-costly env changes"
  - "Phase 2 env-parity targets mechanical fidelity, not bit-exact seed replay against kaggle_environments"
symptoms:
  - "10m+ JAX compile on smoke or light benchmark after mechanical parity fix commit 0eb349e"
  - "make test-kaggle-parity and trace hygiene pass while integration loop is blocked by compile cliff"
  - "Pick #4 greenfield 75a7cf2 admitted on fast gates but adversarial review found validity defects before fix attempt"
  - "Integration HEAD rolled back from 0eb349e/75a7cf2 to 9db50f5 (picks #1-2 + 3b only)"
root_cause: config_error
resolution_type: workflow_improvement
tags:
  - phase2-cherry-pick
  - pick-4
  - jax-compile
  - compile-time-regression
  - mechanical-parity
  - integration-worktree
  - rollback-criteria
  - planet-generation
  - comet-generation
related_components:
  - src/jax/planet_generation.py
  - src/jax/comet_generation.py
  - src/jax/env.py
  - docs/benchmarks/cherry-pick-manifest.json
  - docs/plans/2026-06-06-001-fix-pick4-jax-parity-plan.md
  - docs/solutions/conventions/jax-no-kaggle-callbacks.md
  - docs/solutions/workflow-issues/phase2-env-parity-cherry-pick-integration-admission.md
---

# Phase 2 pick #4 — JAX compile cliff rollback and mechanical parity gates

## Context

Phase 2 on `orbit_wars-integration` (`throughput-baseline-integration`) admitted picks **#1**, **#2**, and **3b** at integration HEAD **`9db50f5`**. Pick **#4** landed as greenfield pure JAX planet and comet generation at **`75a7cf2`** — no callbacks, no `src/game/*` imports on the hot path ([jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md)).

Adversarial review found **mechanical validity defects** at `75a7cf2` (too few planet groups, `initial_planets` desync after comet expire, comet ID allocation, shared px/py RNG key, inlined spawn subgraph). User adopted **mechanical fidelity** framing: JAX must obey Orbit Wars rules and emit only valid states — maps and comet paths **may differ** from `kaggle_environments` for the same seed. Plan: `docs/plans/2026-06-06-001-fix-pick4-jax-parity-plan.md`.

A fix slice at **`0eb349e`** passed fast gates (51 parity tests, tier-A trace hygiene) but caused **10m+ JAX compile** on smoke/light benchmarks. Operator **rejected** the fix and rolled the integration worktree back:

```bash
cd /home/jmduea/projects/orbit_wars-integration
git reset --hard 9db50f59f6f0d42d74b37cb1dbee373fc3ed6827
```

Manifest records `compile_time_regression` as reject reason; pick #4 greenfield is **off the worktree** until re-applied with an acceptable compile path.

## Guidance

### Independent reject tracks (parity ≠ throughput ≠ compile)

Phase 2 uses **separate manifest verdicts** — same pattern as pick #3 throughput reject:

| Track | Gate | Pick #4 example |
| --- | --- | --- |
| Correctness | `make test-kaggle-parity` + `make test-jax-trace-hygiene` | Green at `0eb349e` |
| Rollout throughput | Unified `admission` gate (operator) | Not re-run after rollback |
| **Compile cost** | Smoke/benchmark first-compile wall time (operator/agent) | **REJECT** at `0eb349e` — 10m+ |

Green parity does **not** admit a pick when compile cost blocks the integration loop. Record `compile_time_regression` in manifest `candidates[]` alongside `throughput_regression` and `parity_fail`.

### Rollback criteria for integration picks

Roll back integration HEAD when **any** of:

1. **Compile cliff** — first-compile on representative smoke/benchmark exceeds operator tolerance (session: 10m+ after `0eb349e`).
2. **Throughput regression** — admission extract outside learning-first baseline band (pick #3: ~18× rollout slowdown).
3. **Trace hygiene violation** — tier-A `rg` or `test_jax_trace_hygiene` fails (callbacks, `src/game/*` in hot path).
4. **Parity fail** — `make test-kaggle-parity` red.

**Rollback action:** `git reset --hard <last_good_sha>` on integration worktree; update manifest `integration_state.integration_head_sha` and candidate verdict. Preserve rejected SHAs in manifest notes for forensics (`75a7cf2`, `0eb349e`).

**Last good SHA (2026-06-06):** `9db50f5` — picks #1, #2, 3b only.

### Mechanical parity scope (not seed replay)

**In scope for pick #4 re-attempt:**

- Validity invariants: 5–10 planet groups, four-planet symmetry, bounds, no collisions, orbiting group present
- `planets.active == initial_planets.active` through comet spawn and post-move expire (step 200+)
- Comet IDs from reserved tail slots (not `max(active_id)+1`)
- JAX-only hot path (no `pure_callback`, no sequential `lax.scan` in `_launch_fleets`)

**Out of scope / deprioritized:**

- Bit-exact coordinate goldens vs reference on fixed seeds
- Kaggle `Random(f"orbit_wars-comet-…")` string RNG bridge
- Home-group PRNG stream coupling to Python `random.Random`

Reference libs in `src/game/planet_generation.py` and `src/game/comet_generation.py` anchor **rules** for tests only.

### Compile cliff likely causes at `0eb349e`

Plan unit **KTD7** hoisted comet spawn into `@jax.jit _jit_spawn_comet_group` behind `lax.cond` — correct for trace hygiene but can still expand compile when spawn subgraph is large or re-traced per env batch. Planet two-phase `while_loop` fixes also increase static trace size.

**Before re-admitting pick #4 fixes**, measure compile on the **same smoke path** the operator uses (short train smoke or `ow benchmark training` with `--updates 3 --warmup 1`), not only unit/parity tests. Parity pytest does not bound training-loop compile.

### Re-attempt strategies (open decisions)

| Approach | Tradeoff |
| --- | --- |
| Precompute planet/comet tables at `reset` only; keep `step` spawn-free | Smaller per-step trace; may shift when compile happens |
| Deferred spawn `jit` with explicit cache / static arg bounds | Needs proof compile stays under tolerance on 32-env vmap |
| Re-apply greenfield `75a7cf2` then fix defects incrementally with compile check per hunk | Avoids big-bang fix commit; slower but bisectable |
| Pick #5 mechanics hunks deferred | Do not bundle with planet/comet until compile path is green |

Pick #3 full bundle remains **rejected** — no sequential `lax.scan` fleet launch (~18× throughput).

### Integration worktree workflow

| Role | Path | Branch | HEAD (2026-06-06) |
| --- | --- | --- | --- |
| Gate harness | `/home/jmduea/projects/orbit_wars` | `main` | Admission YAML + manifest |
| Phase 1 anchor | `/home/jmduea/projects/orbit_wars-throughput-anchor` | `throughput-baseline` | Anchor admission passed |
| Phase 2 integration | `/home/jmduea/projects/orbit_wars-integration` | `throughput-baseline-integration` | **`9db50f5`** |

Per-pick agent gates (from main harness for trace hygiene):

```bash
cd /home/jmduea/projects/orbit_wars-integration
make test-kaggle-parity

cd /home/jmduea/projects/orbit_wars
make test-jax-trace-hygiene
```

Operator milestone only: `make gate-admission REPO_ROOT=/home/jmduea/projects/orbit_wars-integration`

Dry-run locked recipe before long GPU jobs:

```bash
uv run ow benchmark gate run admission --dry-run --verbose \
  --repo-root /home/jmduea/projects/orbit_wars-integration \
  --output-root /home/jmduea/projects/orbit_wars-integration/outputs
```

## Why This Matters

Pick #4 showed that **mechanical fidelity fixes can pass every fast correctness gate** while still failing the integration program on **compile economics**. Without an explicit compile-time reject track, agents risk shipping `0eb349e`-class commits that block smoke, benchmark, and operator iteration for 10+ minutes per cold trace.

Rollback to `9db50f5` preserves the verified substrate (picks #1–2 + 3b + prior admission on picks #1–2) while forcing pick #4 to re-enter as **compile-bounded** greenfield work — not a parity-only merge.

## When to Apply

- Before re-applying pick #4 greenfield (`75a7cf2`) or any mechanical fidelity fix on integration.
- When parity is green but smoke/benchmark "hangs" on first update — treat as compile gate, not training bug ([benchmark-subprocess-training-observability.md](../developer-experience/benchmark-subprocess-training-observability.md)).
- When documenting manifest verdicts — add `compile_time_regression` distinct from `throughput_regression`.
- When scoping Phase 2 proof — mechanical validity invariants, not coordinate goldens ([jax-comet-kaggle-parity-ci-gate.md](../architecture-patterns/jax-comet-kaggle-parity-ci-gate.md) defers refresh: pure_callback guidance is legacy for forward path).

## Examples

**Wrong — admit on parity alone after compile smoke fails:**

```bash
# 51/51 parity + trace green, but 10m+ compile on benchmark
# → still REJECT; do not leave integration HEAD at 0eb349e
```

**Wrong — conflate mechanical fidelity with seed replay:**

```bash
# Assert JAX comet paths == kaggle_environments coordinates for seed 0
# → out of scope for Phase 2; blocks pure JAX ports
```

**Right — rollback + manifest update:**

```bash
cd /home/jmduea/projects/orbit_wars-integration
git reset --hard 9db50f59f6f0d42d74b37cb1dbee373fc3ed6827
# Update docs/benchmarks/cherry-pick-manifest.json:
#   integration_state.integration_head_sha = 9db50f5
#   mechanical_fidelity_fix_verdict = rejected
#   mechanical_fidelity_fix_reject_reason = compile_time_regression
```

**Right — per-pick compile check before manifest admit:**

```bash
cd /home/jmduea/projects/orbit_wars-integration
# After pick #4 hunk: parity + trace, then:
cd /home/jmduea/projects/orbit_wars
uv run ow benchmark training --preset primary --updates 3 --warmup 1 --label pick4-compile-check
# Operator judges wall time to update 1 vs pre-pick baseline
```

## Related

- Pick #3 throughput + admission recipe: [phase2-env-parity-cherry-pick-integration-admission.md](phase2-env-parity-cherry-pick-integration-admission.md)
- JAX-only hot path: [jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md)
- Mechanical fidelity plan (fix rolled back): `docs/plans/2026-06-06-001-fix-pick4-jax-parity-plan.md`
- Session handoff: `docs/session-handoff/2026-06-06-phase2-env-parity-rollback-continued.md`
- Manifest pick state: `docs/benchmarks/cherry-pick-manifest.json` (`pick_4_attempt_2026_06_06`, `integration_state`)
- Legacy comet callback pattern (refresh candidate): [jax-comet-kaggle-parity-ci-gate.md](../architecture-patterns/jax-comet-kaggle-parity-ci-gate.md)
