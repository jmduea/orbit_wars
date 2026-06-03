---
title: Kaggle bracket ranking foundational slice (qualifier vs main μ/σ)
date: 2026-06-03
category: architecture-patterns
module: artifacts-tournament-bracket
problem_type: architecture_pattern
component: development_workflow
severity: medium
applies_when:
  - "Training with artifacts=bracket_training after preflight gates 0-3"
  - "Separating Gate 5 proof floors (0.76) from qualifier training floors (1.0)"
  - "Crowning incumbent or skipping qualifier via lineage"
tags:
  - bracket-ranking
  - trueskill
  - qualifier-ladder
  - bracket-training
  - weak-config
  - lineage-skip
related_components:
  - src/artifacts/tournament/bracket/
  - src/jax/train/bracket_training.py
  - conf/artifacts/bracket_training.yaml
  - src/artifacts/qualifier_eval.py
---

# Kaggle bracket ranking foundational slice (qualifier vs main μ/σ)

## Context

Kaggle Orbit Wars ranks submissions with **margin-independent μ/σ updates** (TrueSkill-style), after a **qualifier ladder** at combined win rate **1.0** vs noop → random → nearest_sniper. PR [#186](https://github.com/jmduea/orbit_wars/pull/186) ships the **foundational slice (U1–U6)**: bracket state, qualifier mode on the unified ladder executor, lineage skip, 500M env-step `weak_config` budget, bracket self-play hooks, and `ow eval bracket` status — while **U7–U8** (full async worker round-robin) remain operator follow-up.

Gate 5 **proof** still uses calibrated **0.76** combined floors; qualifier mode uses **1.0** — do not conflate the two.

## Guidance

### Two phases: qualifier vs main bracket

| Phase | Win requirement | Purpose |
|-------|-----------------|--------|
| Qualifier (until incumbent crowned) | Combined 1.0 vs noop, random, nearest_sniper | Enter main bracket / crown incumbent |
| Gate 5 proof (held-out) | Combined ≥ 0.76 (calibrated) | Submit-valid verification |
| Main bracket | μ/σ updates, draws regress to mean | Competition-style ranking among qualified entries |

State persists at `outputs/campaigns/<campaign>/bracket/state.json` (`src/artifacts/tournament/bracket/state.py`).

### Qualifier mode on unified ladder

`qualifier_mode=True` on `run_unified_ladder` applies **1.0 floors** and adds nearest_sniper as a qualifier stage before main-bracket entry. Reuses combined 2p+4p scoring from `unified/scoring.py` — same executor as Gate 5, different thresholds (`src/artifacts/tournament/bracket/qualifier.py`).

### Lineage skip

Checkpoints record `parent_checkpoint_path`. If parent resolves to the current promoted/crowned incumbent, **skip** the 1.0 qualifier ladder and enter main bracket directly (`src/artifacts/tournament/bracket/lineage.py`).

### Training profile

```bash
uv run ow train artifacts=bracket_training \
  output.campaign=<name> \
  task=shield_cheap \
  curriculum=off
```

`bracket_training_tick` in `src/jax/train/bracket_training.py` tracks env steps, emits `bracket_training_phase`, queues **`qualifier_eval`** optional jobs on interval, and sets **`weak_config: true`** if 500M steps pass without qualifier clear.

### Worker follow-up (deferred)

`src/artifacts/qualifier_eval.py` mirrors `checkpoint_eval` (Docker → qualifier ladder → bracket state). Smoke at u50 may **not** queue jobs if checkpoint save timing misses the interval — re-test with shorter `qualifier_eval_interval_updates` and `ow eval worker --run <run_dir>`.

## Why This Matters

Treating Gate 5 proof as qualifier training mis-calibrates expectations (0.76 vs 1.0). Without bracket state and lineage skip, every checkpoint re-runs noop/random/sniper gates and cannot model Kaggle's μ/σ main bracket. The slice intentionally separates **calibration evidence** (`weak_config`) from **submit-valid proof**.

## When to Apply

- Choosing `artifacts=bracket_training` vs `artifacts=hybrid_promotion`
- Inspecting campaign bracket: `uv run ow eval bracket status --campaign <name>`
- Planning U7–U8 worker/scheduling work in `docs/plans/2026-06-03-005-feat-kaggle-bracket-ranking-plan.md`

## Examples

**Novel config path:** Pass gates 0–3 → `bracket_training` → periodic qualifier eval → clear 1.0 ladder → main bracket μ/σ → eventual hybrid promotion proof at 0.76.

**Incumbent lineage path:** Train from promoted parent → lineage skip → main bracket without re-running 1.0 qualifiers.

**Budget exhaust:** 500M env steps without qualifier clear → `weak_config` metric for next shaping iteration (not submit-valid).

## Related

- Unified Gate 5 proof (0.76, Docker-first): `docs/solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md`
- Plan U7–U8 deferred + PR #186 remaining work: `docs/plans/2026-06-03-005-feat-kaggle-bracket-ranking-plan.md`
- Requirements: Kaggle μ/σ semantics in plan 005 Sources
