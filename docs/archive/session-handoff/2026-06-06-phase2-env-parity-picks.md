# Session handoff — Phase 2 env-parity cherry-picks

**Date:** 2026-06-06  
**Audience:** Operator starting a fresh agent session on Orbit Wars cherry-pick integration.

---

## Start here

You are continuing the **nuclear cherry-pick manifest** program after **Phase 1** (anchor admission) is complete.

**Read first (in order):**

1. [cherry-pick-admission-gate-unified-learn-throughput.md](../solutions/workflow-issues/cherry-pick-admission-gate-unified-learn-throughput.md) — unified learning + throughput gate, PPO config source of truth, `--repo-root`
2. [nuclear-cherry-pick-manifest-baseline-integration.md](../solutions/workflow-issues/nuclear-cherry-pick-manifest-baseline-integration.md) — worktree roles, Phase 2 granularity, manifest updates
3. [cherry-pick-manifest.json](../benchmarks/cherry-pick-manifest.json) — baseline gates, empty `candidates[]` ready for Phase 2 entries

**Your goal (Phase 2):** Cherry-pick **env-parity substrate** from merge-base → `main` onto **`throughput-baseline-integration`** at **file or hunk** granularity (not whole commits). Preserve anchor throughput; do not run 30-minute tier-2 e2e after every hunk.

**Worktrees:**

| Role | Path / branch | Notes |
|------|----------------|-------|
| Gate harness | `main` @ recent commits (unified admission gate, worktree `--repo-root`) | Training conf resolves in target worktree; harness CLI stays on main |
| Phase 1 anchor (admission passed) | `../orbit_wars-throughput-anchor` — branch `throughput-baseline` | `admission_passed: true` on unified recipe; `make test-kaggle-parity` green (15 passed) |
| Phase 2 integration | Create/use `throughput-baseline-integration` worktree | Fork from anchor: `git branch throughput-baseline-integration throughput-baseline` then `git worktree add ../orbit_wars-throughput-baseline-integration throughput-baseline-integration` if not already present |
| Pre-hygiene reference | `../orbit_wars-pre-hygiene` @ pre-hygiene integration point | Baseline SHA anchor for manifest |

**Pick order (human-readable themes — use `git log` / `git diff` merge-base..main to find SHAs, then apply hunks):**

1. Offline game reference libs — comet/planet generation under `src/game/`
2. `encode_learner_turn` / feature path in `src/jax/features.py` if needed for parity
3. Env mechanics hunks in `src/jax/env.py` (and related) — **no** comet mega-path or Kaggle callbacks yet
4. Comet + Kaggle paths **with** train vs Kaggle env split — **reject** landing a single “comet mega-commit” without the split
5. Legacy comet mode hunks needed to **preserve throughput** on the anchor recipe
6. Parity tests (`tests/test_jax_env_parity.py` and related)

**Fast gates per pick (not tier-2 every time):**

- `make test-kaggle-parity` (~20s)
- `make test-jax-trace-hygiene`
- Optional: `uv run ow benchmark env-parity-ab` (if investigating A/B)
- Short smoke: 5-update validation preset train (not full admission)
- **`make gate-admission REPO_ROOT=<integration-worktree>`** only at milestones (stack green, before Phase 3 learning picks)

**Reject criteria — record in manifest `candidates[]`:**

- `parity_fail` — `make test-kaggle-parity` red after pick
- `trace_hygiene_fail` — JAX trace tier gate fails
- `throughput_regression` — milestone admission or extract shows env_steps below learning-first floor (see manifest `baseline_gates.throughput_e2e`)
- `hygiene_change_without_ablation` — within-turn launch dedup masks touched without learner ablation (see launch-hygiene learner ablation gate doc)
- `wrong_pick_granularity` — whole commit landed when split required (e.g. comet without train/kaggle split)

Each candidate entry should include: `sha` or `subject` (human name), `phase: env_parity`, `cherry_pick_order`, gate artifacts under `outputs/`, `verdict` (`admit` | `reject` | `pending`), `reject_reasons[]`.

**Phase 1 done (do not redo unless regression):**

- Unified admission on anchor: `admission_passed: true` (learning VERIFIED + throughput within ±10% of learning-first baseline JSON)
- Main includes: learning-first admission throughput, worktree `--repo-root` for gates, unified admission gate feature
- Compound docs for admission gate + refreshed nuclear baseline-integration doc
- Manifest: parity PASS on anchor; `learn_proof` beat_noop-only marked NOT_VERIFIED (superseded by unified admission)

**Phase 3 (later):** Topological **learning** cherry-picks onto integration head after parity stack is green — full gate sequence per manifest R8.

---

## First commands for new session

```bash
cd /home/jmduea/projects/orbit_wars
make agent-context
git worktree list

# Confirm anchor still green (fast)
cd /home/jmduea/projects/orbit_wars-throughput-anchor
make test-kaggle-parity

# Ensure integration line exists
cd /home/jmduea/projects/orbit_wars
git show-ref throughput-baseline-integration || git branch throughput-baseline-integration throughput-baseline
test -d ../orbit_wars-throughput-baseline-integration || \
  git worktree add ../orbit_wars-throughput-baseline-integration throughput-baseline-integration

# Survey env-parity-related history (adjust merge-base if manifest updates)
cd /home/jmduea/projects/orbit_wars
git merge-base throughput-baseline main
git log --oneline --no-merges $(git merge-base throughput-baseline main)..main -- src/game/ src/jax/env.py src/jax/planet_generation.py src/jax/comet_generation.py tests/test_jax_env_parity.py

# After each hunk on integration worktree:
cd /home/jmduea/projects/orbit_wars-throughput-baseline-integration
make test-kaggle-parity
make test-jax-trace-hygiene

# Milestone only (from main, training in integration tree):
cd /home/jmduea/projects/orbit_wars
make gate-admission REPO_ROOT=/home/jmduea/projects/orbit_wars-throughput-baseline-integration
jq '{admission_passed, verdict, throughput_verdict}' outputs/benchmarks/admission/gate.json
```

Update `docs/benchmarks/cherry-pick-manifest.json` `candidates[]` and `integration_state` as picks land; run `make test-fast` on main if you edit the manifest schema fields.

---

## Escalation

- GPU contention: check terminals / `make agent-context` before long trains
- Do not pipe long `ow` commands through `tail`/`head` — use `--out`, `ow runs watch`, or `tail -f` on run logs
- Human merges integration → `main`; agents propose `ordered_shas` and conflict notes only
