---
title: Phase 2 env-parity cherry-pick — integration fork, admission recipe, pick throughput
date: 2026-06-06
category: workflow-issues
module: jax-env
problem_type: workflow_issue
component: development_workflow
severity: high
applies_when:
  - "Granular env-parity hunks onto throughput-baseline-integration after Phase 1 anchor admission"
  - "Integration worktree admission fails throughput while make test-kaggle-parity stays green"
  - "Evaluating full pick #3 env mechanics bundle vs partial cheap hunks (pick 3b)"
  - "Integration branch was forked at baseline_sha 79162a2 without anchor Phase-1 commits"
symptoms:
  - "Integration admission used default 500 rollout_steps and 6 candidates while anchor pass used 256/3/wandb"
  - "Full pick #3 dropped env_steps_per_sec from ~5696 to ~373 (~18x rollout slowdown)"
  - "Picks 1-2 pass parity and trace hygiene but fail admission at wrong recipe geometry"
  - "Green make test-kaggle-parity does not catch sequential intra-player launch edge cases"
root_cause: config_error
resolution_type: workflow_improvement
tags:
  - phase2-cherry-pick
  - env-parity
  - admission-gate
  - throughput-regression
  - sequential-launch
  - pick-3b
  - integration-worktree
  - jax-only-hot-path
related_components:
  - conf/benchmark/gates/admission.yaml
  - src/cli/benchmark_gates.py
  - src/jax/env.py
  - docs/benchmarks/cherry-pick-manifest.json
  - docs/solutions/conventions/jax-no-kaggle-callbacks.md
---

# Phase 2 env-parity cherry-pick — integration fork, admission recipe, pick throughput

## Context

Phase 2 of the nuclear cherry-pick manifest layers **granular env-parity hunks** from `main` onto `throughput-baseline-integration` after Phase 1 anchor admission passes on the throughput-anchor worktree. Correctness (`make test-kaggle-parity`) and performance (unified `admission` gate at the operator-locked recipe) are independent tracks — green parity does not imply admission throughput or tournament fidelity.

This session integrated picks #1–2 (game reference libs + `encode_learner_turn` call sites), attempted full pick #3 (env mechanics bundle), rejected it for throughput, admitted partial **pick 3b** (cheap mechanics hunks only), and verified unified admission on the integration head at the locked recipe.

## Guidance

### Integration fork: sync anchor commits before Phase 2

The integration worktree was initially forked at pre-hygiene `79162a2` **without** the anchor's seven Phase-1 commits (admission gate wiring, PPO source-of-truth fixes, resolved-config display). Integration admission runs against stale harness semantics until the branch fast-forwards to anchor HEAD `52dfdb0`.

**Before Phase 2 picks:** ensure `throughput-baseline-integration` includes anchor Phase-1 commits (`git merge` or `ff-merge` anchor → integration). Record `branch_base_sha` in `docs/benchmarks/cherry-pick-manifest.json` `integration_state`.

### Admission gate must use operator-locked recipe (not defaults)

Failed integration admission runs used **default geometry** (500 `rollout_steps`, 6 `candidate_count`, wandb off) while the anchor pass used the locked recipe (256 steps, 3 candidates, wandb on). A failed run at wrong geometry is **not evidence against picks #1–2**.

**Fix (shipped on main harness):** wire manifest `gate_train_overrides` into `conf/benchmark/gates/admission.yaml` and merge them in `src/cli/benchmark_gates.py`:

```yaml
# conf/benchmark/gates/admission.yaml (excerpt)
train_overrides:
  - training=2p4p_32_split
  - training.rollout_steps=256
  - task.candidate_count=3
  - telemetry.wandb.enabled=true
  - telemetry.wandb.group=preflight
  - artifacts.replay.enabled=false
```

```python
# src/cli/benchmark_gates.py — recipe overrides precede CLI --train-overrides
extra_train_overrides=(*recipe_train_overrides, *train_overrides),
```

**Run admission on integration** (harness on main, training conf in worktree):

```bash
make gate-admission \
  REPO_ROOT=~/projects/orbit_wars-integration \
  ADMISSION_OUT=~/projects/orbit_wars-integration/outputs/benchmarks/admission/gate.json
```

Dry-run to confirm resolved geometry before a 200-update GPU job:

```bash
uv run ow benchmark gate run admission --dry-run --verbose \
  --repo-root ~/projects/orbit_wars-integration \
  --output-root ~/projects/orbit_wars-integration/outputs
```

See [cherry-pick-admission-gate-unified-learn-throughput.md](cherry-pick-admission-gate-unified-learn-throughput.md) for PPO source-of-truth and `--repo-root` boundaries.

### Per-pick gates: parity + trace hygiene; admission at milestones

| Pick | Scope | Fast gates | Throughput gate |
| --- | --- | --- | --- |
| #1 | `src/game/{planet,comet}_generation.py`, constants | `make test-kaggle-parity` | — |
| #2 | `src/jax/features.py` + env learner encode call sites | parity + `make test-jax-trace-hygiene` | — |
| #3 full | Sequential `_launch_fleets` via `lax.scan`, `step_multi_player` | parity PASS | **REJECT** (~18× rollout slowdown) |
| **3b** | Rotation index, `ship_speed`, first-hit combat, `planet_id` launch guard only | parity + trace PASS | Defer unless operator requests |

Do **not** run tier-2 e2e per hunk; reserve unified admission for integration-head milestones after a coherent pick set.

### Pick #3 throughput regression (rejected)

Full pick #3 replaced parallel fleet launch with **sequential `lax.scan` over fleet slots** (`_launch_fleets` + `_concat_fleets`/`_compact_fleets` per slot). Measured on integration at locked recipe:

| Stage | `env_steps_per_sec` (rollout, u3–20 mean) |
| --- | --- |
| Pre pick #3 | ~5696 |
| Post full pick #3 | ~373 |

**Action:** revert pick #3 hunks A–B (sequential launch + `step_multi_player`); keep picks #1–2.

### Sequential launch nuance (defer expensive path)

Kaggle `process_moves` runs a player's action slots **sequentially within that player's list** — not parallel across players. The integration branch already launches per-player via `jax.lax.fori_loop` over `player_count`; sequential `lax.scan` per slot is only required for **intra-player multi-slot oversubscription** edge cases.

**Default train path:** keep parallel slot launch until tournament or targeted tests prove oversubscription parity requires sequential scan. Defer pick #3 hunk A; do not pay ~18× rollout cost on every step.

### JAX-only hot path (user constraint)

Rejected deferred paths that reintroduce callbacks or split env modes:

- No `env_parity_mode` / `task=kaggle_parity` with Python reference in `reset`/`step`
- No `pure_callback` / `io_callback` in rollout hot path
- Forward path for planet/comet parity: **pure JAX** ports in `src/jax/planet_generation.py` and `src/jax/comet_generation.py` (planned picks #4–#5)

See [jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md).

### Pick 3b: admitted cheap mechanics hunks

Partial pick **3b** (`src/jax/env.py` hunks C–F only):

- Planet rotation index matches Kaggle `obs.step` (step+1 indexing)
- `cfg.ship_speed` for fleet speed cap
- First-hit combat uses minimum planet index
- Launch guard validates `action.source_id` against `planets.id` at clipped index

Excluded: sequential `_launch_fleets` (hunk A), `step()` → `step_multi_player` (hunk B), comet/callback/`env_parity_mode` paths.

Parity and trace hygiene green; admission **not** re-run after 3b (operator discretion).

### Verified admission outcome (2026-06-06)

Integration @ picks #1–2 on base `52dfdb0`, run `20260606T060248Z`:

- Learning: `win_rate_delta` 0.173 (VERIFIED)
- Throughput: `env_steps_per_sec` ~5419, `seconds_per_update` ~1.512 (within ±10% of `launch-hygiene-e2e-baseline-learning-first.json`)
- Integration head after pick 3b: `9db50f5`

Inspect:

```bash
jq '{admission_passed, verdict, throughput_verdict}' \
  ~/projects/orbit_wars-integration/outputs/benchmarks/admission/gate.json
```

## Why This Matters

Phase 2 fails silently when (1) integration lacks anchor harness commits, (2) admission runs at default geometry instead of the locked recipe, or (3) a parity-green mechanics bundle ships a sequential launch that destroys rollout throughput. Wiring `train_overrides` into admission YAML makes the locked recipe the default; granular hunk admission separates correctness substrate from performance-critical launch semantics.

Fast parity tests validate JAX vs reference snapshots — they do **not** cover sequential launch oversubscription or full tournament ladders. Treat parity green as necessary but not sufficient for submit-valid or admission throughput proof.

## When to Apply

- After Phase 1 anchor `admission_passed: true`, before layering env-parity hunks on `throughput-baseline-integration`.
- When integration admission fails but `make test-kaggle-parity` passes — check recipe geometry and `integration_state.branch_base_sha` first.
- When evaluating env mechanics commits — split hunks like pick 3b; reject bundled sequential launch without throughput measurement.
- Before planned JAX-only planet/comet generation picks (#4–#5).

## Examples

**Wrong — admission without locked recipe (false reject picks #1–2):**

```bash
# Implicit defaults: rollout_steps=500, candidate_count=6 — not comparable to anchor
uv run ow benchmark gate run admission --repo-root ~/projects/orbit_wars-integration
```

**Wrong — full pick #3 replay after throughput reject:**

```bash
git cherry-pick <env-mechanics-commit>  # bundles sequential lax.scan launch
# → ~18× rollout regression even when parity tests pass
```

**Right — Phase 2 pick order with manifest update:**

1. `ff-merge` anchor `52dfdb0` → integration
2. Apply pick #1 → `make test-kaggle-parity`
3. Apply pick #2 → parity + `make test-jax-trace-hygiene`
4. `make gate-admission REPO_ROOT=~/projects/orbit_wars-integration` (locked recipe from YAML)
5. Apply pick 3b hunks only → parity + trace; re-run admission only if operator requests

## Related

- Phase 1/3 manifest strategy: [nuclear-cherry-pick-manifest-baseline-integration.md](nuclear-cherry-pick-manifest-baseline-integration.md)
- Unified admission recipe and PPO pins: [cherry-pick-admission-gate-unified-learn-throughput.md](cherry-pick-admission-gate-unified-learn-throughput.md)
- JAX hot-path convention: [jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md)
- Validation-preset bisect (orthogonal to tier-2 admission): [jax-validation-throughput-benchmark-and-bisect.md](jax-validation-throughput-benchmark-and-bisect.md)
- Committed pick state: `docs/benchmarks/cherry-pick-manifest.json` (`integration_state`, `candidates[]`, `decision`)
