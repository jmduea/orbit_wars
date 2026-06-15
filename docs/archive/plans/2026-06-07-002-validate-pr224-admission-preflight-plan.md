---
date: 2026-06-07
topic: validate-pr224-admission-preflight
status: completed
branch: prep/launch-hygiene-cherry-pick
pr: 224
head: ee929fa
---

# Plan: Validate PR #224 locally (roadmap Step 1)

## Summary

Confirm P1 fixes on `prep/launch-hygiene-cherry-pick` (PR #224) pass targeted compose/gate tests, JAX env parity, and trace hygiene. Verify config intent for lattice default vs selected_validate experiment profile and Makefile tier-2 admission geometry.

## Requirements

| ID | Requirement | Pass criteria |
|----|-------------|---------------|
| R1 | Preflight sweep compose + training benchmark gate tests | `uv run pytest tests/test_ssot_wandb_sweep_compose.py tests/test_training_benchmark_gate.py -q` exit 0, all tests pass |
| R2 | Kaggle-relevant JAX env parity | `make test-kaggle-parity` exit 0 |
| R3 | JAX trace hygiene tier | `make test-jax-trace-hygiene` exit 0 |
| R4 | Task base default sampling mode | `conf/task/base.yaml` → `rollout_factorized_sampling: lattice` |
| R5 | Selected-validate experiment profile | `conf/task/rollout_selected_validate.yaml` → `rollout_factorized_sampling: selected_validate` |
| R6 | Tier-2 throughput gate geometry | Makefile `test-launch-hygiene-e2e-throughput` uses `--preset admission` and `--baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json` |

## Verification commands

```bash
cd /home/jmduea/projects/orbit_wars-integration
git fetch && git checkout prep/launch-hygiene-cherry-pick && git pull
git rev-parse HEAD  # expect ee929fa or later

# R1
uv run pytest tests/test_ssot_wandb_sweep_compose.py tests/test_training_benchmark_gate.py -q

# R2
make test-kaggle-parity

# R3
make test-jax-trace-hygiene
```

Config checks (static):

```bash
grep rollout_factorized_sampling conf/task/base.yaml
grep rollout_factorized_sampling conf/task/rollout_selected_validate.yaml
grep -A6 'test-launch-hygiene-e2e-throughput:' Makefile
```

## Pass/fail criteria

- **PASS:** All three test commands exit 0 with zero failures; R4–R6 config/Makefile checks match intent.
- **FAIL:** Any test failure or config drift → surgical fix on branch (no assertion weakening), re-run full suite, document in Validation Results below.

## Validation Results

**Run date:** 2026-06-07  
**HEAD:** `ee929fa` (`docs: mark P1 plan complete and record residual findings`)

| Check | Command | Exit | Result |
|-------|---------|------|--------|
| R1 | `uv run pytest tests/test_ssot_wandb_sweep_compose.py tests/test_training_benchmark_gate.py -q` | 0 | **13 passed**, 1 skipped |
| R2 | `make test-kaggle-parity` | 0 | **25 passed**, 1 deselected (328s) |
| R3 | JAX trace hygiene (integration equivalent) | 0 | Static rg: **0 forbidden matches**; pytest `test_jax_trace_hygiene.py` (main harness, integration cwd): **10 passed** |
| R4 | `conf/task/base.yaml` | — | `rollout_factorized_sampling: lattice` ✓ |
| R5 | `conf/task/rollout_selected_validate.yaml` | — | `rollout_factorized_sampling: selected_validate` ✓ |
| R6 | Makefile tier-2 | — | `--preset admission` + `launch-hygiene-e2e-baseline-learning-first.json` ✓ |

**R3 note:** Integration tree lacks `make test-jax-trace-hygiene` (anchor-era); per `cherry-pick-manifest.json` and session handoff, ran tier-A static `rg` on integration `src/jax` plus `tests/test_jax_trace_hygiene.py` from main harness with integration as cwd. Running `make test-jax-trace-hygiene` from main alone would scan main `src/jax` (legacy `_reference_*`) and fail — not the integration hot path.

**Verdict:** All Step 1 gates **PASS**. No code fixes required.
