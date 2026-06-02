---
title: "ops: Multi-branch merge orchestration"
type: ops
status: active
date: 2026-06-02
origin: docs/brainstorms/2026-06-02-merge-orchestration-requirements.md
---

# ops: Multi-Branch Merge Orchestration

## Summary

Execute the merge playbook from `docs/brainstorms/2026-06-02-merge-orchestration-requirements.md`: stabilize both feature branches with commits + `make test-fast`, hard-reset integration branch `merge-sim/planet-flow-preflight` from `main`, merge preflight then planet-flow, pass `make test-premerge`, land on `main` via one PR (default). Target **CPU-complete** by R19; GPU proof is **proof-complete** (R19b) and deferred.

## Problem

Parallel agent sessions left:

- `feat/preflight-training-profiles` — 4 commits + dirty docs/calibration
- `feat/planet-flow-policy` — worktree at `.worktrees/feat/planet-flow-policy` with uncommitted code/docs
- Stale sim at `merge-sim/planet-flow-preflight` (`/tmp/merge-sim-planet-flow`)

Nine+ paths conflict on feature merge (not only five vs `main`).

## Key decisions

**KTD1 — Integration sandbox is mandatory.** All conflict resolution happens on hard-reset `merge-sim/planet-flow-preflight`; never parallel-merge both features straight to `main`.

**KTD2 — Preflight branch wins profile/calibration semantics.** Planet Flow CLI/metrics layers merge in without dropping preflight profile wiring (conflict playbook in requirements R8).

**KTD3 — One PR default.** Split to two PRs only if review size demands it after green integration (R14).

**KTD4 — Calibration regenerated on integration if ambiguous.** Do not hand-merge stale threshold JSON from both dirty checkouts.

## Implementation units

### U1. Inventory and backup (R1–R3)

**Goal:** Known SHAs, recoverable snapshots, merge-tree conflict list.

**Steps:**

1. From repo root (`feat/preflight-training-profiles` checkout):

```bash
git worktree list
git status -sb
git log -1 --oneline feat/preflight-training-profiles feat/planet-flow-policy main
```

2. Capture merge-tree at current tips (refine after U2 commits):

```bash
git merge-tree $(git merge-base feat/preflight-training-profiles feat/planet-flow-policy) \
  feat/preflight-training-profiles feat/planet-flow-policy | rg 'changed in both' || true
```

3. Backup both checkouts (requirements R2 stash SHA pattern).

**Verification:** Backup ref exists under `refs/backup/pre-merge-*`; conflict path list saved in PR notes.

---

### U2. Commit and smoke each feature branch (R4–R6)

**Goal:** Clean commits on both branches; `make test-fast` green before merge.

**Preflight checkout** (primary repo):

```bash
git status --short
# Commit AGENTS.md, docs, calibration ONLY if aligned with preflight-profiles.json
make test-fast
git log -1 --oneline   # record SHA for run log
```

**Planet-flow worktree:**

```bash
cd .worktrees/feat/planet-flow-policy
git status --short   # include all modified src/tests/conf
make test-fast
git log -1 --oneline   # record SHA
```

**Verification (AE1):** Both SHAs recorded; both branches pass `make test-fast`.

---

### U3. Hard-reset integration branch and merge (R7–R11)

**Goal:** Single green integration tip with both streams merged.

**Checkout:** `/tmp/merge-sim-planet-flow` (existing) or new `.worktrees/merge-integration`.

```bash
cd /tmp/merge-sim-planet-flow   # or chosen worktree
git fetch origin
git checkout merge-sim/planet-flow-preflight
git reset --hard origin/main    # or local main tip

git merge --no-ff feat/preflight-training-profiles -m "merge: preflight training profiles"
# Resolve per conflict playbook (requirements R8 table)
git merge --no-ff feat/planet-flow-policy -m "merge: planet flow policy"
# Resolve remaining conflicts

make preflight-calibrate   # if calibration JSON conflict ambiguous
make test-premerge
```

**High-touch files:** `src/jax/preflight.py`, `src/jax/preflight_calibration.py`, `src/cli/benchmark.py`, `src/jax/rollout/metrics.py`, `src/jax/train/metrics.py`, `AGENTS.md`, `docs/benchmarks/preflight-calibration.json`, `docs/benchmarks/preflight-profiles.json`.

**Verification (AE2):** `make test-premerge` exit 0 on integration branch; calibration JSON committed with provenance note in merge commit or PR body.

---

### U4. Land on main (R12–R15)

**Goal:** `main` contains integration tip; feature branches merged or explicitly abandoned.

**Default (one PR):**

```bash
git push -u origin merge-sim/planet-flow-preflight
gh pr create --base main --head merge-sim/planet-flow-preflight \
  --title "merge: preflight profiles + planet flow policy" \
  --body "$(cat <<'EOF'
## Summary
- feat/preflight-training-profiles: per-model PPO profiles, win-rate fixes
- feat/planet-flow-policy: Planet Flow policy + proof pipeline

## Integration SHAs
- preflight: <SHA from U2>
- planet-flow: <SHA from U2>
- integration: <SHA after U3>

## Test plan
- [x] make test-premerge on integration branch
EOF
)"
```

Merge PR when CI green. Optionally delete/archive feature branches after operator confirms.

**Verification (AE3):** `main` includes both streams; `git status` clean on `main`; backup refs retained until confirmed.

---

### U5. Proof-complete follow-on (R19b — optional, post-merge)

**Goal:** GPU go/no-go evidence when needed; does not block U4.

Sequential (one GPU):

```bash
# From main after merge
uv run ow benchmark learn-proof --model transformer_factorized_small --through beat_random \
  --out outputs/preflight/factorized_post_hygiene_learn_proof.json

# From planet-flow-capable checkout
uv run ow benchmark learn-proof --model planet_flow_target_heatmap --through beat_random \
  --out outputs/preflight/planet_flow_learn_proof.json
```

**Verification (AE4):** JSON artifacts exist locally; not required for CPU-complete closure.

## Out of scope

- `issue/*` audit branches
- Untracked brainstorm docs (land separately per R16)
- compare-runs / scorecard implementation
- Deleting `/tmp/merge-sim-planet-flow` until merge confirmed

## Risks

| Risk | Mitigation |
|------|------------|
| Stale sim merge resolutions | U3 hard-reset to `main` before merges |
| Bad calibration baked in commits | U2 omit or regenerate calibration; U3 `make preflight-calibrate` |
| `test-premerge` parallel with other pytest | Check terminals folder before U3 test run |
| R14 two-PR split reintroduces conflicts | Default one PR; only split if review requires |

## Requirements traceability

| Req | Unit |
|-----|------|
| R1–R3 | U1 |
| R4–R6 | U2 |
| R7–R11 | U3 |
| R12–R15 | U4 |
| R19b | U5 |
