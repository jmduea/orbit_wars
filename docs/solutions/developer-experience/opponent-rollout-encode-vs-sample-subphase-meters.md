---
title: Opponent rollout encode vs sample — sub-phase meters prove sample dominates
date: 2026-06-06
category: developer-experience
module: jax-rollout
problem_type: developer_experience
component: tooling
severity: high
applies_when:
  - "Optimizing opponent rollout throughput when rollout_phase_opponent_fraction is ~70%"
  - "Choosing between encode_turn lite paths vs opponent sampling / neural inference work"
  - "Interpreting plans or docs that label opponent fraction as opponent_sample without sub-meters"
tags:
  - opponent-rollout
  - rollout-phase-timing
  - encode-turn
  - profiling
  - throughput
  - collect-timed
related_components:
  - src/jax/rollout/collect_timed.py
  - src/jax/rollout/phase_timing_report.py
  - src/opponents/jax_actions/sampling.py
---

# Opponent rollout encode vs sample — sub-phase meters prove sample dominates

## Context

Offline `ow benchmark rollout-phase-profile` reports **`rollout_phase_opponent_fraction` ~68–70%** on mixed-opponent recipes (`scripted_heavy`, `production_mix`). Plans and ideation often treated that bucket as **encode + sample** and prioritized `encode_turn` reductions (lite scripted encode, `opp_batch_cache` extensions, 4p encode dedup, family-batched dispatch).

The combined **opponent** timer in `collect_timed.py` merged:

- **Sample** — `_sample_opponent_2p_action` / 4p `_four_player_step_action` (shield, family loop, neural K-step decoder, historical pool vmap)
- **Encode** — 2p post-step `_encode_opponent_turn_batch_2p`; 4p inline `_encode_four_player_turn_batches`

Without a split, **encode-first optimizations ran on faith**. ce-optimize experiments (historical batching, 4p carry+cond, edge-only scripted encode) reverted or moved opponent fraction by ≤2.3pp. Family-batched mixed dispatch was already on main and did not lower the ~70% opponent share.

## Guidance

### 1. Add sub-meters before opponent hot-path changes

Extend offline `collect_timed.py` to emit:

| Key | 2p scope | 4p scope |
|-----|----------|----------|
| `rollout_phase_opponent_sample_*` | `_sample_opponent_2p_action` | `_four_player_step_action` only |
| `rollout_phase_opponent_encode_*` | post-step `opp_encode_2p` | `_encode_four_player_turn_batches` only |

`rollout_phase_opponent_seconds` remains **sample + encode**. `phase_timing_report.py` prints indented `sample` / `encode` under `opponent` in breakdown output.

### 2. Profile with the production opponent recipe

```bash
cd /path/to/orbit_wars-integration

env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark rollout-phase-profile \
  --preset admission \
  --train-overrides task=map_pool opponents=default curriculum=default \
  --updates 3 --warmup 2 \
  --out /tmp/opp-split-production_mix.json

uv run ow benchmark rollout-phase-breakdown /tmp/opp-split-production_mix.json
```

Repeat for `scripted_heavy` ladder overrides when isolating scripted vs neural mix.

### 3. Measured split (quick geometry: 2 envs × 8 steps, 2p only, 2026-06-06)

| Rung | Opponent total | Sample (of collect) | Encode (of collect) | Sample % of opponent |
|------|----------------|---------------------|---------------------|----------------------|
| `scripted_heavy` | 78.1% | **74.9%** | 3.2% | **96%** |
| `production_mix` | 73.0% | **67.7%** | 5.3% | **92%** |
| `noop` (reference) | 1.6% | ~1.6% | ~0% | encode skip path |

**Conclusion:** At quick 2p geometry, **`encode_turn` is ~3–5% of collect**, not ~70%. The opponent bucket is almost entirely **sample** (shield vmap, seven-family dispatch, neural forwards, historical pool vmap).

### 4. Optimization priority after measurement

| Priority | Target | Deprioritize at quick 2p |
|----------|--------|-------------------------|
| High | Neural opponent inference batching, historical pool gather without full-pool vmap, scripted shield cost inside sample path | Lite `encode_scripted_opponent_turn`, post-step encode skip |
| Medium | Re-profile at `--full-geometry` before 4p encode bets | Assuming family-batched dispatch will move ~70% opponent share |
| Low | Sub-meters only (diagnostic) | Treating `rollout_phase_opponent_fraction` as `opponent_sample` in plans |

## Why This Matters

Optimizing the wrong half of the opponent bucket wastes GPU cycles and JIT surface:

- Edge-only scripted encode helped `scripted_heavy` by only **~0.3pp** in ce-optimize batch 3 — consistent with encode being a small slice.
- **Family-batched dispatch** already shipped; opponent fraction unchanged — dispatch was not the floor.
- **production_mix** adds ~36% more absolute opponent seconds vs scripted but the **fraction** stays ~70% because sample (neural path) scales with the recipe.

The noop ladder rung (~1.6% opponent) proves post-step encode skip collapses cost when opponents ignore features; non-noop recipes still pay the **sample** path every step.

## When to Apply

- Before any opponent-rollout ce-optimize hypothesis targeting encode paths.
- When a doc or plan cites `opponent_sample ≈ 68%` without sub-meter keys — that number is **`rollout_phase_opponent_fraction`** until split metrics exist.
- After adding sub-meters, re-run breakdown for **both** `scripted_heavy` and `production_mix` at the geometry you will gate on.

Skip sub-meter interpretation when:

- You only need rollout vs PPO split → `ow benchmark training --detailed-timing`.
- Profile JSON predates `rollout_phase_opponent_sample_seconds` keys (no `opponent_details` in breakdown).

## Examples

### Breakdown output (production_mix, quick)

```
  opponent       68.347    73.0%
    sample       63.203    67.7%
    encode        5.143     5.3%
```

### Instrumentation shape (`collect_timed.py`)

2p per step:

1. Time `opponent_phase_2p` → `opponent_sample`
2. After env step, time `opp_encode_2p` → `opponent_encode`

4p per step:

1. Time `_encode_four_player_turn_batches` → `opponent_encode`
2. Time `_four_player_step_action` vmap → `opponent_sample`

Initial `_initial_opponent_batch_cache_2p` encode before the step loop is **not** in per-step encode meters (one-time per rollout scan).

### What didn't work (encode-first hypotheses)

| Hypothesis | Outcome |
|------------|---------|
| Historical snapshot batched forward | +0.9pp opponent fraction (reverted) |
| 4p `player_batches` carry + `lax.cond` | +1.6pp (reverted) |
| `encode_scripted_opponent_turn` | +2.3pp on `production_mix`; +0.3pp on scripted diagnostic only |
| Family-batched mixed dispatch (already on main) | No material change to ~70% opponent share |

## Related

- [offline-rollout-phase-profile-decoupled-from-jit-collect.md](offline-rollout-phase-profile-decoupled-from-jit-collect.md) — parent profiler; documents combined opponent bucket and links sub-meters.
- [production-training-throughput-profiling.md](production-training-throughput-profiling.md) — two-level attribution (rollout vs PPO, then phase profile inside collect).
- Integration implementation: `orbit_wars-integration` branch `optimize/opponent-rollout-throughput` (`collect_timed.py`, `phase_timing.py`, `phase_timing_report.py`).
- Ideation artifact: `docs/ideation/2026-06-06-opponent-encode-sample-throughput-ideation.md` (reprioritize survivors after this measurement).
