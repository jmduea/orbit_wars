---
title: Keep launch hygiene when tier-2 throughput fails but learner ablation wins
date: 2026-06-02
category: tooling-decisions
module: jax-training-benchmarks
problem_type: tooling_decision
component: development_workflow
severity: high
applies_when:
  - "Tier-2 launch-hygiene e2e gate fails vs pre-hygiene baseline after hot-path options are exhausted"
  - "Tier-1 sampler microbench passes but production rollout+PPO path remains out of band"
  - "Deciding whether to revert launch hygiene or accept throughput cost for learnability"
tags:
  - launch-hygiene
  - tier-2-throughput
  - learner-ablation
  - preflight-learn-proof
  - benchmark-gates
  - rollout-throughput
related_components:
  - docs/benchmarks/launch-hygiene-ablation.json
  - docs/benchmarks/launch-hygiene-e2e-baseline.json
  - docs/operator-runbook.md
  - src/jax/action_sampling.py
---

# Keep launch hygiene when tier-2 throughput fails but learner ablation wins

## Context

Launch hygiene (PR #163) prevents degenerate multi-launch sequences during training. After the tier-1 O(K²) sampler fix ([incremental carry doc](../performance-issues/launch-hygiene-incremental-carry-throughput.md)), tier-1 microbench passes (≤ 3.22 ms at K=5), but **tier-2 production e2e** still fails vs the pre-hygiene baseline captured at SHA `79162a2088160b8ed05c3e3a050e064c7f6c9556`.

Measured on RTX 5080 (2026-06-02, `main`):

| Metric | Pre-hygiene baseline (A) | Launch-hygiene main (B) | Tier-2 band (10%) |
|--------|--------------------------|-------------------------|-------------------|
| `env_steps_per_sec` | 9,776 | 2,399 (−75.5%) | floor 8,799 |
| `seconds_per_update_mean` | 1.64 | 6.67 (+307%) | ceiling 1.80 |

Profiling ([rollout throughput design](../../plans/2026-06-01-launch-hygiene-rollout-throughput-design.md)) shows rollout collection (~13.7 s/update) dominates PPO (~0.7 s/update). Hot-path recovery options in `src/jax/action_sampling.py` are **exhausted**; Phase B (U7) is cancelled unless a new rollout sampling design lands.

When throughput parity looks unreachable, **do not** treat tier-2 pass as the only merge authority. Run a **learner ablation** and let preflight learn-proof gate trends decide.

## Guidance

### Gate hierarchy

1. **Tier-1** (optional sanity): `make test-launch-hygiene-throughput` — isolated factorized sampler microbench.
2. **Tier-2** (throughput authority when hot path is recoverable): `make test-launch-hygiene-e2e-throughput` vs `docs/benchmarks/launch-hygiene-e2e-baseline.json` with `--assert-within-pct 10`.
3. **Learner ablation** (tiebreaker when tier-2 fails and hot path is exhausted): compare arm A (pre-hygiene SHA) vs arm B (launch-hygiene `main`) on learn-proof through `beat_random`. **Winner = better learner, not throughput.**

Document outcomes in `docs/benchmarks/launch-hygiene-ablation.json`. CI guards schema via `tests/test_training_benchmark_gate.py::test_committed_launch_hygiene_ablation_artifact`.

### Ablation procedure

```bash
# Arm B (launch-hygiene main) — on canonical GPU host
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  make preflight-learn-proof

# Arm A (pre-hygiene) — git worktree at baseline SHA
git worktree add ../orbit_wars-pre-hygiene 79162a2088160b8ed05c3e3a050e064c7f6c9556
cd ../orbit_wars-pre-hygiene && uv sync --group dev
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark learn-proof \
    --model transformer_factorized_small \
    --through beat_random \
    --out outputs/preflight/ablation_arm_a_pre_hygiene.json
```

Refresh the committed ablation JSON from captured learn-proof artifacts. Thresholds come from `docs/benchmarks/preflight-calibration.json` — do not relax until a run passes under the same recipe.

### Recorded decision (2026-06-02)

| Arm | Learn-proof | Tier-2 e2e |
|-----|-------------|------------|
| A (pre-hygiene, `79162a…`) | NOT_VERIFIED — `beat_noop` win_rate_delta 0.0 | N/A (baseline) |
| B (launch-hygiene, `main`) | VERIFIED through `beat_random` | FAIL (~4× slower sec/update) |

**Decision:** Keep launch hygiene on `main`. Accept throughput cost in favor of learnability. Tier-2 remains **failed out of band**; optional recovery is ROADMAP **Later** only if a new rollout design lands.

## Why This Matters

Turning hygiene off restores throughput but produces a learner that fails the minimum learning signal (`beat_noop`). Keeping hygiene with ~4× slower updates is preferable when the project's north star is **trainable policy quality**, not env_steps/sec alone.

Using tier-2 alone would block merge indefinitely or force a hygiene revert that regresses learning. The ablation framework separates "can we ship faster?" from "does the training stack actually learn?" and prevents burning cycles on unrecoverable hot-path parity.

## When to Apply

- Tier-2 e2e fails after documented hot-path options are exhausted (`hot_path_status: exhausted` in ablation JSON).
- Profiling confirms rollout collection (not PPO) dominates `seconds_per_update_mean`.
- Product deadline pressure tempts disabling hygiene — run ablation first.
- **Do not** use learner ablation to bypass tier-2 when hot-path fixes are still open — ablation is the tiebreaker, not a default skip.

## Examples

### Tier-2 gate (honest fail recording)

```bash
make test-launch-hygiene-e2e-throughput
# Non-zero exit → record metrics in ablation JSON throughput_e2e.gate_failures
```

### Ablation artifact shape (committed snapshot)

Key fields in `docs/benchmarks/launch-hygiene-ablation.json`:

- `criterion`: winner = learn-proof, not throughput
- `hot_path_status`: `exhausted`
- `phase_b_status`: `cancelled_hot_path_exhausted`
- `tier2_status`: `failed_out_of_band`
- `arms.A_pre_hygiene` / `arms.B_launch_hygiene` with `learn_proof.verdict` and `throughput_e2e`
- `winner`: `B_launch_hygiene`
- `decision`: human-readable rationale string

Re-run procedure: `docs/operator-runbook.md` § Learner ablation.

### What didn't work (session history)

- **Throughput parity as sole merge gate** after hot-path exhaustion — tier-2 still ~75% below env_steps/sec floor with no remaining safe optimizations (session history).
- **Assuming tier-1 pass implies tier-2 pass** — sampler microbench green while full rollout+PPO path remains ~4× slower (session history).
- **Pre-hygiene baseline as learner reference** — arm A fails `beat_noop` (0% win-rate delta), so reverting hygiene trades speed for a non-learning stack (session history).

## Related

- Tier-1 O(K²) fix: [launch-hygiene-incremental-carry-throughput.md](../performance-issues/launch-hygiene-incremental-carry-throughput.md)
- Production-path profiling: [production-training-throughput-profiling.md](../developer-experience/production-training-throughput-profiling.md)
- Operator commands: [operator-runbook.md](../../operator-runbook.md)
- Plan: [2026-06-02-013-feat-launch-hygiene-tier2-preflight-plan.md](../../plans/2026-06-02-013-feat-launch-hygiene-tier2-preflight-plan.md)
- Ablation artifact: [launch-hygiene-ablation.json](../../benchmarks/launch-hygiene-ablation.json)
- PR: [#182](https://github.com/jmduea/orbit_wars/pull/182) (merged 2026-06-02)
- Submit-valid agent funnel (#160/#161, hybrid promotion) is a **separate** workflow — see [AGENT_CAPABILITIES.md](../../AGENT_CAPABILITIES.md); do not conflate Docker/tournament proof with throughput or learn-proof gates.
