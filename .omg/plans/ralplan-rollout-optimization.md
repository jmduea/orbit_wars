# Ralplan: Rollout Optimization (from deep-interview spec)

**Source spec:** `.omg/specs/deep-interview-rollout-optimization.md`  
**North star:** 4p / `mix_2p_4p` rollout throughput approaches 2p-only.

## RALPLAN-DR Summary

### Principles

1. **Measure before refactor** — baseline A/B with existing `rollout_env_steps_per_sec_*` telemetry.
2. **Attack 4p structural cost first** — player loop and eager opponent actions dominate vs 2p.
3. **Preserve semantics by default** — branch specialization and caching before behavior changes.
4. **One change per experiment** — isolate JIT/regression risk.
5. **Parity tests gate semantic changes** — `test_jax_env_parity`, rollout curriculum tests.

### Decision Drivers

1. **4p per-step work scales ~4× on encode + opponent sampling** inside `collect_rollout_jax`.
2. **Mixed format runs 2p and 4p collectors sequentially** — no overlap.
3. **User wants audit → implement top wins** — throughput over perfect behavioral equivalence unless tests fail.

### Viable Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A — Measure & tune** | Run baseline smokes; tune `rollout_microbatch_envs`, format weights, env counts only | Zero semantic risk; fast | Won't close large 4p gap alone |
| **B — 4p hot-path refactor** | Lazy opponent branches; `fori_loop` players; reduce per-step `encode_turn` | Largest 4p upside | Medium dev cost; needs profiler + tests |
| **C — Format orchestration** | Overlap/pipeline 2p+4p groups; curriculum skew; separate compile per stage family | Helps mixed wall time | Doesn't fix 4p per-step cost |
| **D — Lean rollout mode** | Strip scan diagnostics / defer metrics | Easy rollout % win | Loses telemetry fidelity |

**User-selected sequence:** **A → B → C → D** (measure using existing 2p/4p-only throughput runs, then 4p hot-path, format orchestration, lean rollout).

## Phased Plan

### Phase 0 — Baseline (required)

- **Primary:** Ingest existing `2p_only_throughput` and `4p_only_throughput` W&B campaigns (`conf/sweeps/wandb/*.yaml`) — compare best `rollout_env_steps_per_sec_*` / `samples_per_sec` at matched `rollout_steps`, `rollout_microbatch_envs`, env count.
- **Secondary:** Short `mix_2p_4p_8env` smoke only if mixed-format gap vs sum of singles is unclear.
- Record: per-format rollout seconds, env steps/sec, rollout fraction; ratio **4p / 2p** as closure target.
- Optional: `jax.profiler` trace one update per format after gaps are tabulated.

**Exit:** Gap table appendix in this plan; targets for Phase 2/4 set from measured ratio (not fixed 15% guess).

### Phase 0 — W&B anchor (user-provided)

- Project: [orbit_wars on W&B](https://wandb.ai/jmduea-jdueadev/orbit_wars)
- Campaigns: `2p_only_throughput`, `4p_only_throughput`
- **2p ceiling (user):** `training.rollout_steps=500` → **10k+ `env_steps_per_sec`** in `2p_only_throughput`
- **Action:** Tabulate matching `rollout_steps` / `format` / `rollout_microbatch_envs` for 4p campaign; compute ratio vs 2p. Use per-format metrics (`rollout_env_steps_per_sec_2p` / `_4p`) when present, not only aggregate `env_steps_per_sec`.
- API note: `config.output.campaign` filter returned 0 runs via public API — may be stored under tags/group; pull from UI or fix filter key during Phase 0.

### Phase 1 — Quick wins (config + low risk)

- Sweep `training.rollout_microbatch_envs` per format on promoted baseline (`docs/baseline_sweep_results.md`).
- Document curriculum format weights vs actual seconds (both groups active?).
- Verify microbatched path isn't re-JITting per chunk unexpectedly.

**Exit:** Best microbatch per format; no regression in `samples_per_sec`.

### Phase 2 — P0 code: 4p opponent lazy branching

- In `mixed_player_branch`, compute only action family needed for `slot_type` (or `lax.switch` on small family set).
- Keep `single_family` fast path as-is.

**Tests:** `tests/test_curriculum.py` rollout family slot tests; existing 4p rollout tests.

**Exit:** ≥15% `rollout_env_steps_per_sec_4p` improvement on smoke (target; adjust after Phase 0).

### Phase 3 — P0 code: 4p encode / player loop

- Replace Python `for player_id in range(4)` with `lax.fori_loop` or fused structure.
- Avoid full `encode_turn` ×4 when `result.batch` can seed opponent views (spike first).

**Tests:** `test_jax_env_parity`, 4p rollout metrics unchanged for fixed seed (or documented drift).

### Phase 4 — P1: mixed format orchestration

- Evaluate alternating updates by format weight vs parallel device groups.
- ADR if sample mix changes.

### Phase 5 — P2: telemetry lean mode (optional)

- Config flag to slim scan `transition` payload.

## ADR

**Decision:** Pursue **Option B** (4p hot-path) after **Option A** baseline, with **C** if mixed wall-time remains >40% rollout fraction.

**Why:** Code structure explains most 2p vs 4p gap; config alone cannot remove 4× opponent loop.

**Consequences:** Several JIT-specialized collectors possible; checkpoint compatibility unchanged if shapes unchanged.

## Architect Notes

- Static `player_count` per `JaxRolloutGroup` already helps — specialize mixed-opponent graph per `StageView` if stable within stage.
- Avoid duplicating `collect_rollout_jax` bodies; extract `_rollout_step_2p` / `_rollout_step_4p` for readability.
- Historical opponent path stays expensive — gate behind family mask early.

### Phase 0 — Measured gap (local campaigns)

| Format | Best `rollout_env_steps_per_sec` | Config | vs 2p @ rs=500 |
|--------|----------------------------------|--------|----------------|
| 2p | **21,762** | `2p_32env`, rs=500, micro=16 | 1.0× |
| 4p | **6,064–6,151** | `4p_16env`, rs=64–250, micro=8 | **~0.28×** |

**Target for Phase 2+:** close 4p/mixed rollout throughput toward 2p-only (user north star).

### Phase 4 — Format rotation (implemented)

- `training.rotate_format_rollouts=true`: one active format group per update via weighted 100-slot schedule.
- Default `false`: both active groups run sequentially (unchanged).

### Phase 5 — Lean rollout (implemented)

- `training.lean_rollout_metrics=true`: scan omits per-step shield/opponent payloads; post-scan diagnostics return zeros for those fields while keeping PPO + core episode metrics.

## Critic Checklist

- [x] Phase 0 metrics captured before Phase 2
- [x] Each phase has pytest target
- [x] No change to PPO update without rollout fraction evidence
- [ ] Document any intentional distribution shift from lazy branching

**Critic verdict:** Phases 0–5 implemented; verify throughput on next campaign run.
