# Session handoff — Phase 2 env-parity cherry-picks (continued)

**Date:** 2026-06-06  
**Audience:** Fresh agent continuing Phase 2 env-parity cherry-picks onto the integration worktree.  
**Supersedes:** [2026-06-06-phase2-env-parity-picks.md](./2026-06-06-phase2-env-parity-picks.md) for *current* state — keep the prior file as historical Phase 2 kickoff context.

---

## Start here

You are continuing the **nuclear cherry-pick manifest** program. Phase 1 (anchor admission) is done. Phase 2 has landed picks **#1**, **#2**, and partial **3b** on the integration worktree; full pick **#3** was rejected for throughput. **Next work is pick #4** (pure JAX planet/comet generation ports).

**Read first (in order):**

1. [phase2-env-parity-cherry-pick-integration-admission.md](../solutions/workflow-issues/phase2-env-parity-cherry-pick-integration-admission.md) — **ce-compound session doc** (2026-06-06): fork sync, admission recipe fix, pick #3 reject / 3b admit, JAX-only constraint
2. [cherry-pick-manifest.json](../benchmarks/cherry-pick-manifest.json) — committed pick state, `candidates[]`, `integration_state`, `planned_picks`
3. [jax-no-kaggle-callbacks.md](../solutions/conventions/jax-no-kaggle-callbacks.md) — hard user constraint for all forward picks
4. [cherry-pick-admission-gate-unified-learn-throughput.md](../solutions/workflow-issues/cherry-pick-admission-gate-unified-learn-throughput.md) — unified learning + throughput gate, `--repo-root`
5. [AGENTS.md](../../AGENTS.md) — operator commands, test tiers, Phase 2 notes

**Operator runs admission gates themselves.** Agents run fast parity/trace gates per pick; defer `make gate-admission` to milestones unless the operator asks.

---

## Worktrees (as of session end)

| Role | Path | Branch | HEAD / notes |
|------|------|--------|----------------|
| Gate harness | `/home/jmduea/projects/orbit_wars` | `main` | Admission gate recipe fix committed (`conf/benchmark/gates/admission.yaml` `train_overrides`); manifest updated |
| Phase 1 anchor | `/home/jmduea/projects/orbit_wars-throughput-anchor` | `throughput-baseline` | Pre-hygiene anchor + Phase 1 fixes; admission passed on locked recipe |
| Phase 2 integration | `/home/jmduea/projects/orbit_wars-integration` | `throughput-baseline-integration` | **`9db50f5`** — picks #1, #2, **3b** committed |
| Pre-hygiene reference | `/home/jmduea/projects/orbit_wars-pre-hygiene` | detached @ baseline | Manifest `baseline_sha` reference only |

Integration was created at **`../orbit_wars-integration`** (not `../orbit_wars-throughput-baseline-integration` from the original handoff). Use the path above consistently.

Integration branch base is anchor HEAD **`52dfdb0`** (seven Phase 1 commits past manifest `baseline_sha` `79162a2`). Do not fork integration from pre-hygiene again.

---

## What landed (Phase 2 picks)

### Pick #1 — ADMIT

Offline game reference libs from the JAX comet Kaggle parity feature (#188): `src/game/comet_generation.py`, `planet_generation.py`, `constants.py` only. No JAX or test hunks from the same commit.

- Fast gates: parity PASS, trace hygiene PASS
- Manifest: `candidates[0]`, verdict `admit`

### Pick #2 — ADMIT

`encode_learner_turn` / fused learner feature path: `src/jax/features.py` plus learner encode call sites in `src/jax/env.py` (`reset`, `assign_learner_players`, `_finish_step`). Excluded comet state and other env hunks.

- Fast gates: parity PASS, trace hygiene PASS
- Manifest: `candidates[1]`, verdict `admit`

### Pick #3 (full bundle) — REJECT

Env mechanics hunks including **sequential `_launch_fleets` via `lax.scan`** and **`step()` → `step_multi_player`** indirection.

- Parity and trace hygiene: PASS
- Throughput: **~18× rollout regression** (`env_steps_per_sec` ~5696 → ~373 post-pick)
- Action taken: reverted hunks A–B; kept picks #1–2
- Manifest: `candidates[2]`, verdict `reject`, reason `throughput_regression`

**Do not replay full pick #3** without a vectorized slot-order alternative or a proven Kaggle-only test path that does not touch the training hot path.

### Pick 3b (partial) — ADMIT

Cheap mechanics hunks only in `src/jax/env.py` (commit message: partial pick #3 cheap mechanics hunks C–F):

- Planet rotation index parity (Kaggle `obs.step` / step+1 indexing)
- `cfg.ship_speed` for fleet speed cap
- First-hit combat (minimum planet index)
- Launch guard: `action.source_id` vs `planets.id` at clipped index

**Excluded:** sequential launch (hunk A), `step_multi_player` (hunk B), comet/callback/`env_parity_mode` paths.

- Fast gates: parity PASS, trace hygiene PASS
- Admission: **not re-run** after 3b (operator discretion)
- Manifest: `candidates[3]` (`pick_3_partial`), verdict `admit`

---

## Admission state

### Locked recipe (operator)

Mixed 2p/4p, 32 envs, **256 rollout steps**, **3 planet candidates**, wandb on (group `preflight`), replay off. Source: manifest `admission_profile.operator_locked_overrides` and `conf/benchmark/gates/admission.yaml` `train_overrides`.

### Integration VERIFIED (picks #1–2 @ `3f88f1b`, before 3b commit)

Run: `20260606T060248Z-s42-36cbf9d4`

- Learning: VERIFIED (`win_rate_delta` 0.173)
- Throughput: VERIFIED (`env_steps_per_sec` ~5419, `seconds_per_update` ~1.512, within ±10% of learning-first baseline JSON)

### Failed run `053609Z` — not picks fault

Used **wrong gate geometry** (default 500 steps / 6 candidates, no locked overrides). Root cause: admission YAML lacked operator overrides until main harness fix. **Not evidence against picks #1–2.**

### Gate fix (on main)

`conf/benchmark/gates/admission.yaml` now wires manifest `gate_train_overrides` so `make gate-admission` / `ow benchmark gate run admission` default to the locked recipe. Dry-run before long GPU jobs:

```bash
uv run ow benchmark gate run admission --dry-run --verbose \
  --repo-root ~/projects/orbit_wars-integration \
  --output-root ~/projects/orbit_wars-integration/outputs
```

---

## Hard constraints (user — do not violate)

1. **JAX-only hot path** — no `pure_callback`, `io_callback`, `_reference_*` helpers, or `src/game/*` imports in `reset` / `step` / rollout / `jit` paths.
2. **No `env_parity_mode` train/Kaggle split with callbacks** — forward path is pure JAX ports in `src/jax/planet_generation.py` and `src/jax/comet_generation.py`; reference libs are test-only.
3. **Defer sequential `lax.scan` launch** unless vectorized alternative or proven Kaggle-only test path without hot-path cost.
4. **User runs admission gates** — agents document and run fast gates; milestone admission is operator-driven unless explicitly requested.

See [jax-no-kaggle-callbacks.md](../solutions/conventions/jax-no-kaggle-callbacks.md) and manifest `user_constraint`.

---

## Next picks (from manifest `planned_picks`)

| Order | Subject | Status | Notes |
|-------|---------|--------|-------|
| **#4** | Pure JAX `planet_generation.py` + `comet_generation.py` wired into env | **pending — start here** | Wire into `_reset_train` and comet spawn **without callbacks**. Exclude `_reset_kaggle_reference`, `_reference_*`, `env_parity_mode`, `task=kaggle_parity`. |
| **#5** | Remaining mechanics hunks | pending | One hunk per pick if needed; prefer vectorized slot-order launch. Pick 3b already shipped rotation, ship_speed, first-hit, planet_id. Exclude bundled pick #3 replay and `step_multi_player` unless proven neutral. |
| **#6** | Main-branch callback teardown | pending | On `main` eventually — remove dead callback paths so tier-A static gate is clean; single JAX env for train and parity tests. |

After each pick: `make test-kaggle-parity` + trace hygiene (tier-A static `rg` + `make test-jax-trace-hygiene` from main harness against integration `src/jax`). Update manifest `candidates[]` and `integration_state`.

**Milestone only:** `make gate-admission REPO_ROOT=../orbit_wars-integration` when the operator wants full learn+throughput proof on the integration head.

---

## Reject criteria (record in manifest)

- `parity_fail` — `make test-kaggle-parity` red
- `trace_hygiene_fail` — JAX trace tier gate fails
- `throughput_regression` — admission env_steps or seconds/update outside learning-first floor/ceiling
- `wrong_pick_granularity` — whole commit when split required (e.g. comet without pure JAX port)
- Violating **JAX-only hot path** — reject and document under `user_constraint`

---

## Phase 3 (later)

Topological **learning** cherry-picks onto integration head after env-parity stack is green — full gate sequence per manifest R8. Do not start until picks #4–#6 (or operator-defined parity milestone) are stable.

---

## First commands for new session

```bash
cd /home/jmduea/projects/orbit_wars
make agent-context
git worktree list

# Confirm anchor still green (fast)
cd /home/jmduea/projects/orbit_wars-throughput-anchor
make test-kaggle-parity

# Integration head and pick history
cd /home/jmduea/projects/orbit_wars-integration
git log -5 --oneline
git status

# Fast gates on integration (should be green at session end)
make test-kaggle-parity

cd /home/jmduea/projects/orbit_wars
make test-jax-trace-hygiene   # main harness; integration lacks Makefile target

# Survey env-parity history for pick #4 (pure JAX planet/comet)
git merge-base throughput-baseline main
git log --oneline --no-merges $(git merge-base throughput-baseline main)..main -- \
  src/jax/planet_generation.py src/jax/comet_generation.py src/jax/env.py \
  tests/test_jax_env_parity.py

# Confirm admission dry-run uses locked recipe (256/3/wandb) — operator runs full gate
uv run ow benchmark gate run admission --dry-run --verbose \
  --repo-root /home/jmduea/projects/orbit_wars-integration \
  --output-root /home/jmduea/projects/orbit_wars-integration/outputs

# After pick #4 hunks on integration:
cd /home/jmduea/projects/orbit_wars-integration
make test-kaggle-parity
cd /home/jmduea/projects/orbit_wars && make test-jax-trace-hygiene

# Milestone admission (operator only):
cd /home/jmduea/projects/orbit_wars
make gate-admission REPO_ROOT=/home/jmduea/projects/orbit_wars-integration
jq '{admission_passed, verdict, throughput_verdict}' \
  /home/jmduea/projects/orbit_wars-integration/outputs/benchmarks/admission/gate.json

# Update manifest after each admit/reject:
# docs/benchmarks/cherry-pick-manifest.json — candidates[], integration_state
```

---

## Escalation

- GPU contention: check terminals / `make agent-context` before long trains
- Do not pipe long `ow` commands through `tail`/`head` — use `--out`, `ow runs watch`, or `tail -f` on run logs
- Human merges integration → `main`; agents propose `ordered_shas` and conflict notes only
- If admission fails but parity is green: check recipe geometry and `integration_state.branch_base_sha` before reverting picks — see [phase2-env-parity-cherry-pick-integration-admission.md](../solutions/workflow-issues/phase2-env-parity-cherry-pick-integration-admission.md)
