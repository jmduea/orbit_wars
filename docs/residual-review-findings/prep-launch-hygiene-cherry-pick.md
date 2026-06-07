# Residual review findings — `prep/launch-hygiene-cherry-pick`

**Date:** 2026-06-07  
**PR:** https://github.com/jmduea/orbit_wars/pull/224

## Resolved in this pass

| ID | Finding | Resolution |
|----|---------|------------|
| P1-1 | Preflight compose test drift vs fixed YAML | Test updated to noop_only / 50 updates / action_decision=false |
| P1-2/4 | Tier-2 gate geometry vs old primary baseline | `--preset admission` + learning-first baseline JSON |
| P1-3 | `selected_validate` in task base | Restored `lattice` default in `conf/task/base.yaml` |
| P2-6 | Stale preflight YAML header | Comment corrected |

## Residual (non-blocking)

1. **`make test-fast` collection errors (pre-existing on branch):** `tests/test_seed_scheduler_calibration.py` and `tests/test_ssot_preflight_shortlist.py` import modules not present on integration (`src.jax.seed_scheduler_calibration`, `src.jax.planet_flow_shortlist`). Track as integration spine gap — not introduced by P1 fix commit.

2. **Tier-2 throughput gate may fail on hygiene branch:** Expected when post-hygiene `rollout_s` regresses vs learning-first baseline; gate should run without geometry crash.

3. **ce-test-browser:** N/A — no browser surfaces in this Python RL repo.

## Autofix

No safe_auto code-review autofixes required for P1 diff (config/test/Makefile only).
