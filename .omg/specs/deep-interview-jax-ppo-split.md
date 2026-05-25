# JAX PPO Module Split Spec

Generated: 2026-05-24
Workflow: deep-interview
Final ambiguity: 12%

## Goal

Split the 2,428-line `src/jax/ppo.py` into bounded, single-responsibility modules so future changes to opponent sampling, rollout collection, diagnostics, or PPO updates do not require editing one monolithic file.

This is a **behavior-preserving refactor** with two intentional improvements:
1. Move JAX opponent action builders and sampling into `src/opponents/`.
2. Unify full and lean rollout diagnostics behind one shared metrics schema (reducing duplication while preserving existing metric key contracts consumed by training/telemetry).

## Target Module Layout

```
src/jax/
  ppo.py                  # Thin re-export facade (stable public imports)
  ppo_update.py           # PPO loss/update path
  rollout/
    __init__.py
    types.py              # JaxTransitionBatch, JaxTrainState, ShieldedSequenceSample
    collect.py            # collect_rollout_jax + scan_step body
    metrics.py            # Unified rollout diagnostics (full + lean)

src/opponents/
  jax_actions/
    __init__.py
    builders.py           # build_*_action_from_batch, ship_count_for_bucket_jax, shield helpers
    sampling.py           # _sample_*_family_*, mixed 2p/4p, historical, policy sampling
```

### Symbol Assignment

| Symbol(s) | Destination |
|-----------|-------------|
| `JaxTransitionBatch`, `JaxTrainState`, `ShieldedSequenceSample` | `src/jax/rollout/types.py` |
| `collect_rollout_jax` (+ nested `scan_step`) | `src/jax/rollout/collect.py` |
| `_rollout_diagnostics`, `_rollout_diagnostics_lean` | `src/jax/rollout/metrics.py` (unified schema) |
| `build_*_action_from_batch`, `_noop_bucket_mask`, `_ensure_bucket_mask_has_choice`, `_sample_step_from_logits`, `_sample_shielded_sequence_with_params`, `_sample_policy_action*` | `src/opponents/jax_actions/` (builders vs sampling split) |
| `_sample_historical_action`, `_sample_single_family_*`, `_sample_mixed_*`, `_four_player_step_action`, `_opponent_*` helpers | `src/opponents/jax_actions/sampling.py` |
| `ppo_update_jax`, `discounted_returns`, `_reshape_minibatches`, `masked_mean` | `src/jax/ppo_update.py` |
| `init_train_state`, `validate_policy_param_shapes`, `flatten_batch`, `concatenate_transition_batches` | `src/jax/ppo_update.py` |
| Public imports (`collect_rollout_jax`, `ppo_update_jax`, etc.) | Re-exported from `src/jax/ppo.py` |

### Line Budget (≤600 lines/file)

Current section sizes from `ppo.py`:

| Section | Lines | Fits budget? |
|---------|------:|--------------|
| types + init helpers | ~272 | Yes (split across types + ppo_update) |
| opponent actions | ~1,003 | **No** — requires `builders.py` + `sampling.py` sub-split |
| collect_rollout | ~410 | Yes |
| diagnostics (full + lean) | ~470 combined | Yes after dedup into unified schema |
| ppo_update + batch utils | ~273 | Yes |

## Constraints

- **No behavior changes** to rollout collection, opponent sampling semantics, PPO math, or JIT compilation boundaries unless required by the move.
- **Keep `src/jax/ppo.py` as a thin re-export facade** so existing import sites (`src/jax/train.py`, tests, scripts) continue to work unchanged.
- **Cross-package move is in scope**: opponent JAX logic moves to `src/opponents/jax_actions/`. This follows the existing pattern where `src/opponents/runtime.py` already imports `src/jax.policy`.
- **Metrics schema unification is in scope for this PR**: extract shared metric key definitions and reduction helpers; preserve full vs lean entrypoints gated by `cfg.training.lean_rollout_metrics`. Do not break metric keys consumed by `src/jax/train.py` without updating consumers in the same PR.
- **Not part of the broader `src/` domain reorg** (`.omg/specs/deep-interview-src-reorganization.md`) — this is a focused JAX PPO split only.
- **Defer telemetry registry/group config** to `.omg/specs/deep-interview-telemetry-metrics-cleanup.md` — this split prepares metrics for that work but does not implement the central telemetry registry.

## Non-Goals

- Redesigning PPO algorithm, reward semantics, or curriculum behavior.
- Changing Hydra config responsibility groups.
- Checkpoint compatibility changes.
- Moving non-JAX opponent code (`src/opponents/runtime.py`).
- Implementing the full telemetry metric registry or group boolean toggles.
- Removing the full diagnostics path or changing default `lean_rollout_metrics` behavior.

## Acceptance Criteria

1. **Module boundaries**: Each new file ≤600 lines; no file mixes opponent sampling + collect scan + PPO update + diagnostics.
2. **Import stability**: `from src.jax.ppo import collect_rollout_jax, init_train_state, ppo_update_jax, ...` continues to work via facade re-exports.
3. **Tests pass**:
   - `uv run --group dev pytest tests/test_jax_ppo.py tests/test_curriculum.py`
   - Full suite: `uv run --group dev pytest`
4. **Benchmark regression check**: `scripts/benchmark_jax_rl.py` shows no meaningful rollout/update throughput regression (run before/after or confirm existing benchmark still passes).
5. **Metrics contract preserved**: Full and lean paths emit the same keys as today for keys that downstream (`jax/train.py`, telemetry logger) already consumes; shared reductions live in one place in `rollout/metrics.py`.
6. **No import cycles**: `src/opponents/jax_actions/` may import `src/jax/env`, `src/jax/features`, `src/jax/policy`; `src/jax/rollout/collect.py` imports from opponents but opponents must not import from `rollout/collect.py`.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| Pure file split vs behavior change | Split + metrics dedup + opponents move; no algorithm changes |
| Opponent code stays in jax/ | Move to `src/opponents/jax_actions/` |
| Public import break acceptable | No — keep `src/jax/ppo.py` facade |
| Line count matters | Hard ≤600 lines per file; opponents section needs sub-split |
| Performance regression acceptable | No — benchmark must not regress |
| Metrics unification depth | Shared schema/reductions in this PR; telemetry registry deferred |
| Part of full src reorg | No — standalone focused split |

## Ontology

| Entity | Role |
|--------|------|
| **Opponent action builder** | Deterministic JAX function mapping turn batch → `JaxAction` (random, sniper, turtle, etc.) |
| **Opponent sampler** | Stochastic JAX function selecting opponent type and dispatching to builder or policy params |
| **Rollout collector** | JIT-scanned loop: learner action → opponent actions → env step → transition dict |
| **Rollout metrics** | Post-scan reductions over transition data; full vs lean gated by config |
| **PPO update** | Advantage computation, minibatch reshape, clipped surrogate loss, optimizer step |
| **Transition batch** | Named tuple of feature/action/log-prob/return tensors for PPO consumption |
| **Facade module** | `src/jax/ppo.py` re-exporting stable public API |

## Interview Transcript

1. **Scope boundary** (B+C): Split + unify full/lean diagnostics schema; move opponent action builders to `src/opponents/`.
2. **Success criteria** (C+D): ≤600 lines per file; all tests pass; `benchmark_jax_rl.py` no regression.
3. **Symbol placement**: All opponent action + sampling (`_sample_mixed_*`, `_four_player_step_action`) → `src/opponents/jax_actions/`; `rollout/collect.py` orchestrates only.
4. **Imports + metrics** (facade + metrics refactor): Keep `src/jax/ppo.py` re-export facade; unify diagnostics schema in this PR.

## Suggested Implementation Order

1. Create `src/jax/rollout/types.py` — extract data containers (no logic dependencies).
2. Create `src/opponents/jax_actions/builders.py` — extract deterministic builders + shield helpers.
3. Create `src/opponents/jax_actions/sampling.py` — extract sampling orchestration.
4. Create `src/jax/rollout/metrics.py` — extract and unify diagnostics.
5. Create `src/jax/rollout/collect.py` — extract collector (imports from opponents + types + metrics).
6. Create `src/jax/ppo_update.py` — extract update path + train state init + batch utils.
7. Replace `src/jax/ppo.py` body with re-exports.
8. Run targeted tests, then full pytest, then benchmark.
