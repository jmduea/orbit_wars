---
title: Launch hygiene O(K²) prefix recompute regressed factorized sampler throughput
date: 2026-06-01
category: performance-issues
module: jax-policy
problem_type: performance_issue
component: testing_framework
symptoms:
  - "Factorized shield sampler ~56% slower than main at default max_moves_k=5 (batch=32)"
  - "K=8 hygiene path ~390% slower than main in microbenchmark"
  - "PR merge blocked until sampler wall time within 10% of main baseline"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags:
  - launch-hygiene
  - jax
  - factorized-decoder
  - throughput
  - cumulative-forbidden
  - trajectory-shield
  - ppo-rollout
related_components:
  - src/jax/launch_hygiene.py
  - src/jax/action_sampling.py
  - src/jax/factored_sequence_scan.py
  - src/jax/factorized_sampler_benchmark.py
  - src/cli/benchmark/factorized.py
---

# Launch hygiene O(K²) prefix recompute regressed factorized sampler throughput

## Problem

The launch hygiene bundle (PR #163) correctly masked duplicate launches and friendly reverse relays inside the factorized K-step decoder, but implemented hygiene by **recomputing the full launch prefix on every scan step**. That turned per-turn hygiene work from O(K) into O(K²) inside `jax.lax.scan`, adding ~56% wall time at the default `max_moves_k=5` and blocking merge until fixed.

## Symptoms

- Historical measurement via `scripts/benchmark_factorized_sampler.py` (batch=32, `trajectory_shield_mode=cheap`; canonical path is `src/jax/factorized_sampler_benchmark.py` via `ow benchmark factorized-sampler`):

  | K | main (ms) | prefix-recompute hygiene (ms) | Slowdown |
  |---|-----------|-------------------------------|----------|
  | 3 | 1.51 | 2.20 | +46% |
  | **5** | **2.93** | **4.56** | **+56%** |
  | 8 | 2.56 | 12.55 | +390% |

- Training smoke `env_steps_per_sec` looked depressed (~860), but that number came from a non-standard recipe and was not used as the merge gate.
- Merge gate: sampler mean JIT time at K=5 must stay within **110%** of main (~3.22 ms threshold from 2.93 ms baseline).

## What Didn't Work

- **Treating training smoke throughput as proof.** A bare `training.total_updates=100` run with env32/self-play/wandb, piped through `| tail -30`, was not comparable to the factorized sampler microbenchmark.
- **Pytest-based PERF1 gate.** In-process JAX timing under pytest (`conftest` loads JAX + compilation cache) reported ~16 ms/sample while the same code in an isolated subprocess measured ~3 ms. Throughput gates must run outside pytest.
- **Prefix recompute inside the hot scan loop.** Correct for parity-by-construction (original KTD1), but each step called `hygiene_adjusted_bucket_mask_at_step` → `fori_loop(0, step_idx, …)` with expensive `_planet_id_to_row` scans over `MAX_PLANETS=60` and full-grid boolean broadcasts.

## Solution

Replace hot-path prefix recompute with an **incremental `cumulative_forbidden` scan carry** and **turn-static lookups**, while keeping prefix recompute as a **test oracle only**.

### Architecture (unchanged layering)

Trajectory shield and launch hygiene stay separate modules with fixed composition order:

```text
shield(remaining_ships) → hygiene(cumulative_forbidden) → source_mask → sample → [tiered exact reject] → hygiene carry update
```

Do **not** roll tiered/exact shield validation into `launch_hygiene.py`. Tiered reject runs after sampling; hygiene carry updates only when `launch_valid` is true (post-tiered reject).

### Hot-path carry (O(K) per turn)

1. **Once per turn:** `HygieneLookups = build_hygiene_lookups(batch)` (planet-id → row, learner-owned flags).
2. **Scan carry:** `cumulative_forbidden` shape `(env, MAX_PLANETS, k, buckets)`, initialized to zero.
3. **Before step t:** `step_bucket_mask = shield_step_mask & ~cumulative_forbidden`.
4. **After valid launch:** `apply_launch_to_cumulative_forbidden(...)` with sparse `.at[]` updates for dup + friendly-reverse cells.

Rollout (`src/jax/action_sampling.py`) and PPO replay (`src/jax/factored_sequence_scan.py`) share the same helpers from `src/jax/launch_hygiene.py`. Replay stores **shield-only** masks in `bucket_mask_stack` and recomputes hygiene via the same carry during log-prob replay.

### Verification gate

```bash
make test-launch-hygiene-throughput   # delegates to ow benchmark factorized-sampler
uv run ow benchmark factorized-sampler --max-moves-k 5 --batch-size 32 --assert-max-ms 3.22
# legacy: scripts/benchmark_factorized_sampler.py (stderr hints prefer ow command above)
```

Post-fix measured ~3.01 ms (2026-06-01). Do not relax the threshold until a run passes under the same recipe.

### Key tests

- `test_cumulative_carry_matches_oracle_across_steps` — carry ≡ prefix oracle
- `test_tiered_reject_prefix_does_not_apply_hygiene` — tiered reject (`stop=1`, `bucket=0`) must not advance carry
- `test_rollout_replay_logprob_parity_tiered_shield` — R9 parity under tiered mode
- Existing R9 parity tests in `tests/test_factored_sequence_scan.py`

## Why This Works

Prefix-derived hygiene semantics require knowing which `(source_row, slot)` launches actually happened earlier in the turn. Recomputing from stored prefix sequences inside every scan iteration repeats that work quadratically. The carry records the same forbidden set incrementally: each active launch ORs a sparse forbidden mask once. Turn-static lookups eliminate repeated planet-id linear scans.

The oracle path (`compose_hygiene_with_shield_mask` / `hygiene_adjusted_bucket_mask_at_step`) remains for golden tests but is not called from `sequence_scan_body` or replay hot loops.

## Prevention

- **Profile inside `lax.scan` before merging mask layers.** Any per-step `fori_loop(0, step_idx, …)` nested in a K-step scan is a red flag for O(K²) cost at default K=5+.
- **Gate throughput with tiered benchmarks**, not pytest JIT timing or ad-hoc train smokes. Tier-1: `make test-launch-hygiene-throughput` or `uv run ow benchmark factorized-sampler --assert-max-ms`. Tier-2: `make test-launch-hygiene-e2e-throughput` vs `docs/benchmarks/launch-hygiene-e2e-baseline.json` — see `docs/operator-runbook.md`. **Tier-1 pass does not imply tier-2 pass** (sampler microbench can be green while full rollout+PPO remains out of band); when tier-2 fails after hot-path exhaustion, see [learner ablation gate](../tooling-decisions/launch-hygiene-learner-ablation-gate.md).
- **Keep shield and hygiene separate.** Shield answers physics/safety; hygiene answers turn rules (dup/reverse). Compose with AND; update carry only on `launch_valid` after optional tiered reject.
- **Prove carry ≡ oracle** when changing hygiene semantics — do not delete the prefix-recompute path from tests.
- **When adding scan carry state**, extend rollout and replay scans together (R9); pass pre-composed `hygiene_bucket_mask` into replay logprob helpers to avoid double hygiene.

## Related Issues

- Actor–critic encode-once replay contract: `docs/architecture/jax-policy-encoder.md` (shared trunk, separate policy/value heads)
- Requirements: `docs/brainstorms/2026-06-01-launch-hygiene-bundle-requirements.md` (R1–R9)
- Plans: `docs/plans/2026-06-01-launch-hygiene-bundle-plan.md`, `docs/plans/2026-06-01-002-fix-launch-hygiene-throughput-plan.md`
- Production-path timing follow-up: `docs/solutions/developer-experience/production-training-throughput-profiling.md`
- Tier-2 fail / learner ablation tiebreaker: [launch-hygiene-learner-ablation-gate.md](../tooling-decisions/launch-hygiene-learner-ablation-gate.md)
- PR: [#163](https://github.com/jmduea/orbit_wars/pull/163) (merged 2026-06-01)
- Session history: [launch hygiene + throughput arc](c1094a23-fcdd-48fb-941a-e3da7e4fdbfe)
