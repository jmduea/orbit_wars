---
date: 2026-06-07
topic: admission-gate-stack-pr224
status: completed
branch: prep/launch-hygiene-cherry-pick
pr: 224
head: 4770ab2
prior_plan: docs/plans/2026-06-07-002-validate-pr224-admission-preflight-plan.md
---

# Plan: Roadmap Step 2 — Admission gate stack (PR #224)

## Summary

Run the heavier GPU/time admission gates on `prep/launch-hygiene-cherry-pick` (PR #224) at HEAD `4770ab2`, building on Step 1 PASS (`docs/plans/2026-06-07-002-validate-pr224-admission-preflight-plan.md`). Document all metrics; fix only real bugs (crashes, wrong preset, cache not cleared). Do **not** relax calibrated thresholds.

## Scope

| In scope | Out of scope |
|----------|--------------|
| G0–G4 gate execution and metric capture | Full `make gate-admission` composite (deferred until Step 2 green) |
| Surgical fixes for config/preset/crash bugs | Throughput threshold relaxation |
| Plan + validation results commit to PR branch | Push to main |

## Requirements

| ID | Gate | Requirement | Pass bar | Verification |
|----|------|-------------|----------|--------------|
| R1 | G0 Preflight | No pytest/GPU contention; deps synced | Terminals clear; `uv sync --group dev` ok | Check terminals folder; `uv sync --group dev` |
| R2 | G1 Tier-1 throughput | Factorized sampler microbench within calibrated ceiling | `ms_per_sample` ≤ **3.22** (`--assert-max-ms 3.22`) | `make test-launch-hygiene-throughput` |
| R3 | G2 Parity + trace | Step 1 still valid at same HEAD | Skip if HEAD ≥ `4770ab2` and no env/jax hot-path changes since Step 1 | Cite Step 1 plan; re-run only on drift |
| R4 | G3 Cold-cache compile | First-compile wall time on admission-shaped map_pool smoke | `compile_seconds_to_update_3` ≤ **300s** | `env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 uv run ow benchmark training --preset admission --updates 3 --warmup 1 --out /tmp/compile_smoke_pr224.json` plus `task=map_pool` override if admission preset omits it |
| R5 | G4 Tier-2 e2e throughput | Admission-shaped 20-update run completes without crash; document throughput vs learning-first baseline | **Complete without crash** required; `env_steps_per_sec` / `seconds_per_update_mean` vs `docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json` ±10% is informational (expected FAIL on hygiene branch) | `make test-launch-hygiene-e2e-throughput` |

## Verification commands

```bash
cd /home/jmduea/projects/orbit_wars-integration
git fetch && git checkout prep/launch-hygiene-cherry-pick && git pull
git rev-parse HEAD  # expect 4770ab2 or later

# G0
uv sync --group dev

# G1
make test-launch-hygiene-throughput

# G2 — skip when Step 1 valid (see R3)

# G3 — verify task before run
uv run ow train print_resolved_config=true training=smoke task=map_pool
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark training --preset admission --updates 3 --warmup 1 \
  --out /tmp/compile_smoke_pr224.json
# Append task=map_pool via --overrides if admission preset lacks map_pool

# G4
make test-launch-hygiene-e2e-throughput
```

## Expected outcomes (document, do not treat as surprise failure)

- G3 compile may exceed 300s on map_pool integration branch (~434s historical).
- G4 throughput assert may FAIL vs learning-first baseline on hygiene path (~17× rollout regression documented in manifest).

## Validation Results

**Run date:** 2026-06-07  
**HEAD:** `4770ab2` (`docs: record PR #224 Step 1 local validation pass`)

| Gate | Pass/Fail | Key metric | Notes |
|------|-----------|------------|-------|
| G0 | **PASS** | deps synced | Terminals clear after parity/e2e prior runs; `uv sync --group dev` ok |
| G1 | **PASS** | **2.90 ms/sample** (ceiling 3.22) | `make test-launch-hygiene-throughput` exit 0 |
| G2 | **SKIP** | Step 1 valid | HEAD `4770ab2` unchanged for env/jax hot path since Step 1 PASS at `ee929fa`/`4770ab2` — cite `docs/plans/2026-06-07-002-validate-pr224-admission-preflight-plan.md` |
| G3 | **FAIL** | **381.02s** compile to update 3 (ceiling **300s**) | `--preset admission --overrides task=map_pool`; artifact `outputs/benchmarks/admission/compile_smoke_pr224-4770ab2.json`. Expected regression on hygiene+map_pool stack (~434s historical); no fixable bug (cache cleared, correct preset) |
| G4 | **BLOCKED** | — | Agent session SIGTERM (exit 143) during GPU run before completion. **Operator command:** `make test-launch-hygiene-e2e-throughput`. Baseline floor: `env_steps_per_sec` ≥ **4888.8**, `seconds_per_update_mean` ≤ **1.66** (`launch-hygiene-e2e-baseline-learning-first.json`). Throughput FAIL expected on hygiene branch; prior stale `/tmp/launch_hygiene_e2e_gate.json` used wrong `--preset primary` — discard |

### G3 detail

```json
{
  "compile_seconds_to_update_3": 381.0215560710003,
  "seconds_per_update_mean": 48.573055596999744,
  "env_steps_per_sec": 168.65317405532963,
  "rollout_steps": 256,
  "commit_sha": "4770ab239dec6fbd467240a16abe88fdcb658472"
}
```

### G4 operator follow-up

```bash
cd /home/jmduea/projects/orbit_wars-integration
make test-launch-hygiene-e2e-throughput
# Record env_steps_per_sec and seconds_per_update_mean vs baseline floors above.
# Crash-free completion is the hard bar; throughput assert failure is expected/documented.
```

**Verdict:** Step 2 **partial** — G1 PASS, G3 FAIL (expected compile regression), G4 **operator-run required**. No code fixes applied (no crash/preset/cache bugs found).
