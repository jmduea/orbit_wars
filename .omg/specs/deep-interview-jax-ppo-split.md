# JAX PPO Module Split Spec

Generated: 2026-05-24 (revised 2026-05-25)
Workflow: deep-interview (revision pass)
Final ambiguity: 14%

## Goal

Split the 2,428-line `src/jax/ppo.py` into bounded, single-responsibility modules so future changes to opponent sampling, rollout collection, diagnostics, or PPO updates do not require editing one monolithic file.

This is a **behavior-preserving refactor** with two intentional improvements:
1. Move JAX opponent action builders and sampling into `src/opponents/`.
2. Unify full and lean rollout diagnostics behind one shared metrics schema (reducing duplication while preserving existing metric key contracts consumed by training/telemetry).

## Target Module Layout

```
src/jax/
  train_state.py          # init_train_state, validate_policy_param_shapes
  ppo_update.py           # PPO loss/update path + batch utils
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

**Remove `src/jax/ppo.py`** after migration. Call sites import from the owning modules directly (no compatibility facade).

### Symbol Assignment

| Symbol(s) | Destination |
|-----------|-------------|
| `JaxTransitionBatch`, `JaxTrainState`, `ShieldedSequenceSample` | `src/jax/rollout/types.py` |
| `collect_rollout_jax` (+ nested `scan_step`) | `src/jax/rollout/collect.py` |
| `_rollout_diagnostics`, `_rollout_diagnostics_lean` | `src/jax/rollout/metrics.py` (unified schema) |
| `build_*_action_from_batch`, `_noop_bucket_mask`, `_ensure_bucket_mask_has_choice`, `_sample_step_from_logits`, `_sample_shielded_sequence_with_params`, `_sample_policy_action*` | `src/opponents/jax_actions/` (builders vs sampling split) |
| `_sample_historical_action`, `_sample_single_family_*`, `_sample_mixed_*`, `_four_player_step_action`, `_opponent_*` helpers | `src/opponents/jax_actions/sampling.py` |
| `init_train_state`, `validate_policy_param_shapes` | `src/jax/train_state.py` |
| `ppo_update_jax`, `discounted_returns`, `_reshape_minibatches`, `masked_mean`, `flatten_batch`, `concatenate_transition_batches` | `src/jax/ppo_update.py` |

### Module Size Policy

**No line-count budget.** Files may exceed arbitrary line thresholds when they represent a single clear ownership boundary. Split further only when a file mixes concerns (e.g., opponent sampling + collect scan + PPO update + diagnostics in one module).

The opponent section (~1,003 lines today) should still be split into `builders.py` + `sampling.py` because it mixes deterministic builders with stochastic sampling orchestration — not because of a numeric cap.

## Constraints

- **No behavior changes** to rollout collection, opponent sampling semantics, PPO math, or JIT compilation boundaries unless required by the move.
- **Clean import break is acceptable**: remove `src/jax/ppo.py`; update all import sites in `src/`, `tests/`, `scripts/`, and docs in the same PR.
- **Cross-package move is in scope**: opponent JAX logic moves to `src/opponents/jax_actions/`. This follows the existing pattern where `src/opponents/runtime.py` already imports `src/jax.policy`.
- **Metrics schema unification is in scope for this PR**: extract shared metric key definitions and reduction helpers; preserve full vs lean entrypoints gated by `cfg.training.lean_rollout_metrics`. Do not break metric keys consumed by `src/jax/train.py` without updating consumers in the same PR.
- **Train state is its own module**: `init_train_state` and `validate_policy_param_shapes` live in `src/jax/train_state.py`, not in `ppo_update.py`.
- **Not part of the broader `src/` domain reorg** — this is a focused JAX PPO split only.
- **Defer telemetry registry/group config** to `.omg/specs/deep-interview-telemetry-metrics-cleanup.md`.

## Non-Goals

- Redesigning PPO algorithm, reward semantics, or curriculum behavior.
- Changing Hydra config responsibility groups.
- Checkpoint compatibility changes.
- Moving non-JAX opponent code (`src/opponents/runtime.py`).
- Implementing the full telemetry metric registry or group boolean toggles.
- Removing the full diagnostics path or changing default `lean_rollout_metrics` behavior.
- Maintaining a `src/jax/ppo.py` re-export shim.

## Acceptance Criteria

1. **Module boundaries**: No file mixes opponent sampling + collect scan + PPO update + diagnostics. Opponent builders and sampling are separate files under `src/opponents/jax_actions/`.
2. **Import migration**: All former `src.jax.ppo` imports updated to owning modules:
   - `src/jax/train.py`
   - `tests/test_jax_ppo.py`, `tests/test_curriculum.py`
   - `scripts/benchmark_jax_rl.py`
   - Docs referencing `src/jax/ppo.py` (e.g. `docs/ONBOARDING.md`)
   - `src/jax/ppo.py` deleted (not left as empty shim).
3. **Tests pass**:
   - `uv run --group dev pytest tests/test_jax_ppo.py tests/test_curriculum.py`
   - Full suite: `uv run --group dev pytest`
4. **Benchmark regression check**: `scripts/benchmark_jax_rl.py` shows no meaningful rollout/update throughput regression.
5. **Metrics contract preserved**: Full and lean paths emit the same keys as today for keys that downstream (`jax/train.py`, telemetry logger) already consumes; shared reductions live in one place in `rollout/metrics.py`.
6. **No import cycles**: `src/opponents/jax_actions/` may import `src/jax/env`, `src/jax/features`, `src/jax/policy`; `src/jax/rollout/collect.py` imports from opponents but opponents must not import from `rollout/collect.py`.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| Pure file split vs behavior change | Split + metrics dedup + opponents move; no algorithm changes |
| Opponent code stays in jax/ | Move to `src/opponents/jax_actions/` |
| Public import break acceptable | **Yes** — remove `ppo.py`, update all import sites |
| Line count matters | **No** — ownership boundaries only; split when concerns mix |
| Train state placement | **Dedicated `src/jax/train_state.py`** |
| Performance regression acceptable | No — benchmark must not regress |
| Metrics unification depth | Shared schema/reductions in this PR; telemetry registry deferred |
| Opponents move + metrics unification | **Both remain in scope** |

## Ontology

| Entity | Role |
|--------|------|
| **Opponent action builder** | Deterministic JAX function mapping turn batch → `JaxAction` (random, sniper, turtle, etc.) |
| **Opponent sampler** | Stochastic JAX function selecting opponent type and dispatching to builder or policy params |
| **Rollout collector** | JIT-scanned loop: learner action → opponent actions → env step → transition dict |
| **Rollout metrics** | Post-scan reductions over transition data; full vs lean gated by config |
| **PPO update** | Advantage computation, minibatch reshape, clipped surrogate loss, optimizer step |
| **Train state** | Policy parameter initialization and shape validation against env feature dims |
| **Transition batch** | Named tuple of feature/action/log-prob/return tensors for PPO consumption |

## Interview Transcript

### Initial pass (2026-05-24)

1. **Scope boundary** (B+C): Split + unify full/lean diagnostics schema; move opponent action builders to `src/opponents/`.
2. **Success criteria** (C+D): ≤600 lines per file; all tests pass; `benchmark_jax_rl.py` no regression.
3. **Symbol placement**: All opponent action + sampling → `src/opponents/jax_actions/`; `rollout/collect.py` orchestrates only.
4. **Imports + metrics** (facade + metrics refactor): Keep `src/jax/ppo.py` re-export facade; unify diagnostics schema in this PR.

### Revision pass (2026-05-25)

1. **Revision targets**: Line budget, import strategy, train state placement.
2. **Line budget**: Remove numeric cap — ownership boundaries only.
3. **Import strategy**: Drop `src/jax/ppo.py` facade; update all import sites to owning modules.
4. **Train state**: Add `src/jax/train_state.py` for `init_train_state` + `validate_policy_param_shapes`.
5. **Unchanged scope**: Keep opponents move + metrics unification.

## Suggested Implementation Order

1. Create `src/jax/rollout/types.py` — extract data containers (no logic dependencies).
2. Create `src/jax/train_state.py` — extract init + param validation.
3. Create `src/opponents/jax_actions/builders.py` — extract deterministic builders + shield helpers.
4. Create `src/opponents/jax_actions/sampling.py` — extract sampling orchestration.
5. Create `src/jax/rollout/metrics.py` — extract and unify diagnostics.
6. Create `src/jax/rollout/collect.py` — extract collector (imports from opponents + types + metrics).
7. Create `src/jax/ppo_update.py` — extract update path + batch utils.
8. Update import sites in `train.py`, tests, scripts, docs.
9. Delete `src/jax/ppo.py`.
10. Run targeted tests, then full pytest, then benchmark.
