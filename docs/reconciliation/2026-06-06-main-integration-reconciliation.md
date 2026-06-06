# Main ↔ Integration reconciliation (2026-06-06)

Operator map for cherry-pick readiness after Phase 2 env-parity and opponent-rollout work diverged across worktrees.

## Worktree roles (locked)

| Worktree | Branch | HEAD | Role |
|----------|--------|------|------|
| `orbit_wars` | `main` (primary checkout) | `1f8766a` | Cherry-pick manifest SSOT, admission gate harness, docs, eventual controlled merges |
| `orbit_wars-integration` | `optimize/opponent-rollout-throughput` | `87e3915` | Phase 2 env-parity + opponent optimizations landing zone |
| `orbit_wars-pre-hygiene` | detached | `79162a2` | Learning-first throughput baseline capture anchor |
| `orbit_wars-throughput-anchor` | `throughput-baseline` | `52dfdb0` | Post-hygiene throughput anchor (integration branch base) |

## Divergence summary

Merge-base between `main` and `optimize/opponent-rollout-throughput` is **`79162a2`** (pre-hygiene), not current `main`. Integration was built forward from throughput-baseline (`52dfdb0`), not from today's `main`.

`feat/family-batched-mixed-sampling` (PR #221) is based on **`main`** (`1f8766a` + 9 commits). It does **not** share history with integration after `79162a2`.

### Area matrix

| Area | Main | Integration | Action |
|------|------|-------------|--------|
| Cherry-pick manifest + admission gate | Full manifest, `gate run admission`, `--repo-root` | Stale 73-line stub manifest | Cherry-pick manifest infra **from main** when integration is ready to re-sync docs |
| Env-parity picks #1–2, 3b | Not landed | `3f88f1b`, `9db50f5` | **Keep on integration**; cherry-pick to main only after per-pick gates |
| Map pool (pick #4 strategy) | Docs/plans only | `85576f0` + `data/jax_map_pool/default_v1.npz` | **Keep on integration**; re-run compile smoke before admit |
| Family-batched mixed sampling | PR #221 (`9bd91c7`…) | `a46c37e` + `894beb0` gather fix | **Close PR #221** — superseded by integration stack |
| Opponent encode dedupe | — | `fd5f48a` (merged #220 on integration base) | Integration-only until cherry-picked |
| CE-optimize ladder infra | — | `2718ba3` | Integration-only |
| H3/H4 inference-only shield | — | `f979c2d` | Integration-only; exp-001 closed |
| Noop encode-skip JIT-safe | `65877e2` on feat/#221 | Present via `should_skip_opponent_batch_refresh_2p` | Verify on integration admission re-gate; may need feat dispatch refactor (`7bad4e6`) later |
| Benchmark package split | Both (different SHAs) | `fe359fd` | Aligned in spirit; main has encode-turn microbench (`1f8766a`) integration lacks |
| Rollout phase profile CLI | `e5480d0` | Offline path via integration map-pool commit | Cherry-pick from main when benchmarking integration |
| SSOT / qualifier / trace-hygiene CI | On main | Absent (integration is anchor-era + picks) | **Do not** merge main wholesale into integration |

### PR #221 vs integration (opponent sampling)

PR #221 and integration implement the same *intent* (family-batched mixed sampling) on different bases:

- **Integration** adds encode/sample sub-meters, pool gather fix, H3/H4, ladder harness, map pool, env-parity substrate.
- **PR #221** adds a later dispatch refactor (`7bad4e6`), noop JIT-safe collect wiring (`65877e2`), and docs consolidation (`1499f8c`) on top of **main**.

Overlapping files (both touch vs `main`): `src/opponents/jax_actions/sampling.py`, `src/jax/rollout/collect.py`, `tests/test_opponent_mixed_sampling.py`, benchmark package paths.

**Decision:** Close PR #221 without merge. Track any unique #221 commits as optional follow-up cherry-picks onto integration after admission re-gate.

## Integration-only commits (must not lose)

```
87e3915 chore: remove legacy tracked .omg wiki and tmp scripts
894beb0 fix(opponents): correct pool gather indices after family-batched reorder
f979c2d feat(opponent-rollout): inference-only K-step shield path (H3+H4)
a46c37e feat(opponents): family-batched mixed sampling and encode/sample meters
2718ba3 feat(ce-optimize): opponent rollout ladder infra and measurement harness
85576f0 feat(map-pool): offline pool, training reset, and rollout phase profiling
9db50f5 fix(env): partial pick #3 cheap mechanics hunks (C–F)
3f88f1b feat(env-parity): admit Phase 2 picks #1-2 onto throughput baseline
52dfdb0 … ac4b5b8  (throughput-baseline anchor fixes)
```

## Main-only commits (cherry-pick candidates for integration or future main←integration)

High-signal groups on `main` not in integration (post-`79162a2`):

1. **Manifest / admission** — `e2011e6`, `e566abf`, `c70bf4e`, `620128e`, manifest doc commits
2. **Benchmark operator tools** — `21aea59`, `e5480d0`, `1f8766a` (encode-turn microbench)
3. **JAX env cleanup on main** — `b11b9b0` (callback teardown, pick #6 precursor), trace-hygiene CI
4. **SSOT qualifier stack** — `1e832f7` … (defer; not integration scope today)
5. **Feat/#221 unique** — `7bad4e6` dispatch refactor, `65877e2` noop JIT-safe (if not already equivalent)

## Manifest pick status (authoritative: `main` `docs/benchmarks/cherry-pick-manifest.json`)

| Pick | Status | Where landed |
|------|--------|--------------|
| #1 game reference libs | admit | integration @ picks substrate |
| #2 encode_learner_turn | admit | integration |
| #3 full mechanics bundle | **reject** (throughput) | reverted |
| #3b cheap mechanics | admit | integration `9db50f5` |
| #4 greenfield JAX gen | admit (superseded) | rolled back → map pool strategy |
| #4 map pool gather reset | **reject** (compile 306s > 300s) | integration `85576f0` committed — re-verify compile |
| #5 mechanics hunks | **pending** | integration-first, one hunk per pick |
| #6 callback teardown | **pending** | main-first when integration env stable |

**Admission:** VERIFIED on integration @ `3f88f1b` (picks #1–2 only, locked recipe). **Not re-run** after 3b, map pool, family-batched, or H3/H4. Prior exp-001 run: learning PASS, throughput FAIL (~20% below floor) — treat post-optimization stack as **blocked** until `make gate-admission REPO_ROOT=integration` passes.

## Verification (2026-06-06)

```bash
cd orbit_wars-integration
uv run pytest tests/test_opponent_mixed_sampling.py tests/test_opponent_inference_only.py -q
# 10 passed
```

Per-pick gates not re-run this session: `make test-kaggle-parity`, compile smoke, full admission.

## Operator checklist

### Safe to cherry-pick from main → integration (docs/tooling only)

1. Ensure integration branch is clean and admission-relevant picks are committed.
2. Cherry-pick manifest + gate harness commits (`e2011e6` … `620128e`) **or** copy `docs/benchmarks/cherry-pick-manifest.json` from main and resolve conflicts.
3. Run `make test-kaggle-parity` + `make test-jax-trace-hygiene` from main harness with `REPO_ROOT=integration`.
4. Re-run `make gate-admission REPO_ROOT=/home/jmduea/projects/orbit_wars-integration`.

### Safe to cherry-pick from integration → main (later, controlled)

Only after integration admission + compile gates pass:

1. Create `feat/phase2-env-parity-picks` from `main`.
2. Cherry-pick in manifest order: `3f88f1b`, `9db50f5`, `85576f0` (map pool), then opponent stack (`2718ba3` … `87e3915`).
3. Per commit: `make test-kaggle-parity`, trace hygiene, admission gate.
4. **Do not** merge integration wholesale; do not force-push `main`.

### PR hygiene

- **PR #221:** Closed 2026-06-06 — superseded by `optimize/opponent-rollout-throughput`.
- **PR #222 (exp-001):** Already closed; branch deleted.

### Primary checkout

`orbit_wars` should track `main` for manifest edits and `gate-admission` harness. Use `--repo-root` pointing at integration for candidate gates.

## Blockers

1. **Admission throughput** — post-H3/H4 / map-pool stack not re-gated on locked recipe; exp-001 showed ~20% below floor.
2. **Map pool compile** — prior smoke 306s vs 300s ceiling; must pass before pick #4 admit.
3. **Learning thresholds** — calibrated on 2p-only geometry; mixed 2p/4p results provisional.
4. **Branch history** — integration not descended from `main`; cherry-picks require explicit commit lists, not `git merge main`.
5. **Manifest drift** — integration carries obsolete stub manifest; SSOT is on `main`.

## Next steps

1. Re-run `make gate-admission REPO_ROOT=integration` on current `87e3915`.
2. If throughput fails, profile with `ow benchmark rollout-phase-profile --repo-root integration`.
3. Sync manifest stub on integration from main (or delete stub and document pointer to main SSOT).
4. Pick #5 first hunk on integration with parity + throughput smoke per manifest.
