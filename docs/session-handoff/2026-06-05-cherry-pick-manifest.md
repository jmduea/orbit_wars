# Session handoff: nuclear cherry-pick manifest (2026-06-05)

Handoff for starting a fresh session on Orbit Wars cherry-pick manifest / throughput-anchor work.

## Current git state

| Location | Branch | SHA | Notes |
|----------|--------|-----|-------|
| **main** (`/home/jmduea/projects/orbit_wars`) | `main` | `46d4812b975edb89907ce5c471c2fb2e1f7d2013` | Recent: manifest updates, `artifact_pipeline.enabled` kill switch, gate metrics fixes, compound solution doc |
| **throughput anchor worktree** (`/home/jmduea/projects/orbit_wars-throughput-anchor`) | `throughput-baseline` | `52dfdb02ffa9a45e7fcfef41aa065509111c4076` | Tracks `origin/throughput-baseline`; learn-proof metrics fix |
| **integration branch** (main repo) | `throughput-baseline-integration` | `79162a2088160b8ed05c3e3a050e064c7f6c9556` | Pre-hygiene anchor (PR #163 parent) |

Recent commits on `main`:

- `46d4812` — fix(artifacts): honor `pipeline.enabled` kill switch; sync 2p anchor gate
- `8d84ebc` — fix(benchmark): resolve gate metrics from multi-repeat aggregate
- `b9499b8` — docs(benchmarks): clarify tier-2 gate metrics and format coverage
- `f05ae76` — docs(solutions): compound cherry-pick manifest baseline-first workflow
- `9cdc8fe` — docs: record U1 anchor throughput gate pass in cherry-pick manifest

## Completed this session

- **Ideation & docs on main:** nomenclature RFC, PR #219 docs consolidation
- **Cherry-pick manifest:** requirements doc + implementation plan (`2026-06-05-002`)
- **Integration scaffolding:** `throughput-baseline-integration` branch, `throughput-baseline` anchor worktree (`orbit_wars-throughput-anchor`), harness copy for gate capture
- **U1 anchor work (worktree @ 52dfdb0):**
  - 2p throughput captured — manifest now **FAIL** (~7755 `env_steps_per_sec` vs ~8799 floor); earlier PASS (~9628) recorded at different measured commit
  - Learn-proof metrics fixed on anchor (`52dfdb0`)
- **`artifact_pipeline.enabled`:** global kill switch (`46d4812` on main)
- **Gate contract clarified:**
  - Blocking metrics: `env_steps_per_sec`, `seconds_per_update_mean` only
  - `samples_per_sec` recorded but **not gated**
  - `repeats=3` in gate CLI vs Makefile `repeats=1` — documented mismatch
- **Benchmark fixes on main:**
  - Repeats aggregate gate (multi-repeat JSON resolution)
  - Flat `log_prob` on anchor worktree
  - Single-group win rate finalize

## Open decisions (PRIORITY for next session)

### Agreed

**One admission profile must gate BOTH throughput and minimal learn-proof.** If throughput and learn-proof use different Hydra presets/overrides, the dual gate is meaningless — a commit could pass one and fail the other under incomparable training paths.

### Not resolved (user ≠ assistant)

**What Hydra preset + overrides constitute the default admission profile?**

- Assistant leaned toward: `primary` preset → `shield_cheap` + `transformer_factorized` + self-play-only 2p (`training_format: 2p_only`)
- User has a **different vision** — **do not assume** the assistant's lean. Next session must brainstorm and decide explicitly.

## Agreed direction (not yet implemented)

- Align `make preflight-learn-proof` / `PREFLIGHT_TRAIN_BASE` with the **chosen** admission profile (not `transformer_factorized_small` ceiling unless explicitly diagnostic)
- Manifest should record an `admission_profile` field (preset + overrides) on every gate capture
- **Advisory only:** 4p-only and 2p+4p characterization runs stay non-blocking until baselines exist

## Blockers before Phase 2 env-parity cherry-picks

1. **Resolve admission profile definition** (see open decision above)
2. **Re-run dual gate** on anchor @ `52dfdb0` under the unified admission profile
3. **2p throughput FAIL vs earlier PASS** — may need fair re-benchmark after profile is locked (measured commits differ; gate contract / repeats fixes landed since first PASS)
4. **U1 parity + learn_proof manifest fields** still `pending` — require user runs on anchor worktree

## Key paths

| Artifact | Path |
|----------|------|
| Manifest | `docs/benchmarks/cherry-pick-manifest.json` |
| Plan | `docs/plans/2026-06-05-002-feat-nuclear-cherry-pick-manifest-plan.md` |
| Requirements | `docs/brainstorms/2026-06-05-nuclear-cherry-pick-manifest-requirements.md` |
| Compound solution | `docs/solutions/workflow-issues/nuclear-cherry-pick-manifest-baseline-integration.md` |
| Anchor 2p throughput JSON | `orbit_wars-throughput-anchor/outputs/benchmarks/cherry-pick/anchor_throughput_2p.json` |
| E2E baseline | `docs/benchmarks/launch-hygiene-e2e-baseline.json` |
| Nomenclature RFC | `docs/nomenclature-rfc.md` |

## Worktrees / branches quick reference

```bash
# Main repo
cd /home/jmduea/projects/orbit_wars

# Anchor worktree (throughput-baseline @ 52dfdb0)
cd /home/jmduea/projects/orbit_wars-throughput-anchor

# Integration branch (from main repo)
git checkout throughput-baseline-integration  # @ 79162a2
```

## Next session starter prompt

Copy the block below verbatim into a new chat:

```
Orbit Wars cherry-pick manifest session — continue from docs/session-handoff/2026-06-05-cherry-pick-manifest.md. PRIORITY: brainstorm and lock the single admission profile (preset + Hydra overrides) used for BOTH tier-2 throughput gate AND minimal learn-proof on throughput-baseline anchor @ 52dfdb0 (worktree orbit_wars-throughput-anchor). User and assistant previously disagreed on what that profile should be — do not assume shield_cheap + transformer_factorized + 2p_only without explicit agreement. After profile is chosen: re-run dual gate under unified profile, update manifest admission_profile field, reconcile U1 2p FAIL (~7755) vs earlier PASS, then capture pending learn_proof + parity on anchor before Phase 2 env-parity cherry-picks onto throughput-baseline-integration @ 79162a2.
```
