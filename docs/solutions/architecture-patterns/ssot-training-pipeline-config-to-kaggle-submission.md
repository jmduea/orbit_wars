---
title: SSOT training pipeline from config to Kaggle submission
date: 2026-06-03
category: architecture-patterns
module: training-pipeline
problem_type: architecture_pattern
component: development_workflow
severity: high
applies_when:
  - "Defining or documenting the path from Hydra config to Kaggle submission"
  - "Operator or agent sees parallel narratives (hybrid_promotion, bracket_training, Gate 5, lane A/B/C)"
  - "Implementing or reviewing GitHub #211 / plan 2026-06-03-013 SSOT pipeline work"
  - "Placing Docker validation, JAX tournament qualifiers, or rollout curriculum on the spine"
  - "Teardown or relocation of legacy artifact profiles per SSOT R29"
tags:
  - ssot
  - training-pipeline
  - packaging-validation
  - tournament-qualifiers
  - preflight
  - teardown-legacy
  - operator-docs
  - github-211
related_components:
  - docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md
  - docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md
  - docs/competition/COMPETITION_OVERVIEW.md
  - docs/competition/COMPETITION_SUBMISSION.md
  - AGENTS.md
  - conf/artifacts/hybrid_promotion.yaml
  - conf/artifacts/bracket_training.yaml
---

# SSOT training pipeline from config to Kaggle submission

## Context

Operators and coding agents hit the same wall: several **co-equal submit-valid narratives** with overlapping names but different ordering, thresholds, and runtime lanes. At once they could be told to use preflight Gates 0–5 (`learn-proof`, `curriculum_staged`, `tournament-proof` at 0.76), `artifacts=default` vs `artifacts=hybrid_promotion` vs `artifacts=bracket_training`, hybrid `checkpoint_eval` (Docker → tournament → promote), Gate 5 win proof, and bracket async `qualifier_eval` with noop-first ladders at 1.0. Internal shorthand (“lane B promotions”, “lane C”) made it worse — agents picked whichever doc they read last.

Submit-valid closure stalled partly because Gate 4 wall clock and naming overlap obscured which path was authoritative. (session history) Prior tracker work (#206–#210 for PR #187 residuals) was superseded when the team chose a single SSOT requirements doc instead of patching parallel spines.

The fix in this session was **documentation and planning**: requirements SSOT, plain spine terminology, implementation plan, and tracker issue [#211](https://github.com/jmduea/orbit_wars/issues/211) under epic [#205](https://github.com/jmduea/orbit_wars/issues/205). **No implementation code shipped yet.**

## Guidance

**One spine, plain names.** Treat [`docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`](../../brainstorms/2026-06-03-training-pipeline-ssot-requirements.md) as the single source of truth from config to Kaggle submission. Operator and agent docs should describe **only this spine**; anything else is an explicit alternate track (Planet Flow research, launch-hygiene perf gates), not a footnote on the spine.

**Canonical spine order (six phases):**

1. **Config setup** — resolved Hydra config, feature compatibility declared
2. **Preliminary tests** — wiring tier (`make test-fast`); failure blocks everything downstream
3. **Short preflight** — learning-stability gate (trend/KL/entropy from committed calibration JSON); saves a checkpoint for step 4
4. **Packaging validation** — Docker + `kaggle_environments` smoke on the **preflight checkpoint** (seed 0, 4-player, all agents = packaged agent); config registry can skip re-smoke for known-good fingerprints when invalidation dimensions are unchanged
5. **Long train** — ≤500M env steps; rollout curriculum (random → noop-heavy → sniper-heavy) advanced by **tournament qualifiers** (fast JAX held-out eval on `eval_seed_set` only); then **main bracket** (μ/σ)
6. **Submission** — Docker packaging smoke with **trained weights**, held-out noop/random legs, then upload

**Three runtime modes on the spine** (not parallel products):

| Stage | When | Runtime |
|-------|------|---------|
| **Packaging validation** | After preflight, before long train | Docker + `kaggle_environments` |
| **Tournament qualifiers** | During long train on checkpoint ticks | Fast JAX on eval-only seeds |
| **Submission** | After bracket clearance | Docker + upload with trained weights |

**Terminology map — use SSOT names in operator/agent docs:**

| Retire (legacy / confusing) | Use instead |
|----------------------------|-------------|
| Lane B promotions, hybrid `checkpoint_eval` funnel as default spine | **Tournament qualifiers** (JAX, during long train) |
| Lane C, async `qualifier_eval` noop-first ladder as spine | **Long train** curriculum + tournament qualifiers; bracket is post–stage-3 |
| Gate 5 / `tournament-proof` / 0.76 as parallel submit-valid spine | Absorbed into **submission** path after bracket clearance |
| `artifacts=hybrid_promotion`, `artifacts=bracket_training` as production defaults | Future `artifacts=ssot_pipeline` (plan [#211](https://github.com/jmduea/orbit_wars/issues/211)) |

**W&B / Hydra sweeps (research track).** Sweeps may **exit after short preflight passes**. Packaging validation, long train, tournament qualifiers, and submission are **not required** in sweep mode.

**Implementation tracker.** Requirements → plan [`docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`](../../plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md) → [#211](https://github.com/jmduea/orbit_wars/issues/211). Teardown policy (R29): remove or relocate legacy spines; no “demoted but still default” paths in `AGENTS.md` / `docs/AGENT_CAPABILITIES.md`.

**Competition rules SSOT.** Game rules and packaging expectations live in [`docs/competition/COMPETITION_OVERVIEW.md`](../../competition/COMPETITION_OVERVIEW.md) and [`docs/competition/COMPETITION_SUBMISSION.md`](../../competition/COMPETITION_SUBMISSION.md); pipeline docs link there first.

## Why This Matters

Parallel submit-valid paths waste GPU time and agent turns on the wrong funnel. Agents trained on `hybrid_promotion` poll contracts may never run packaging validation on the preflight checkpoint; agents following Gate 5 may treat 0.76 proof floors as production promotion gates while bracket docs demand 1.0 qualifier floors — same words, different semantics. Without one spine, “submit-valid” means different things depending on which doc was indexed last.

The SSOT doc separates concerns cleanly: preflight proves **learning stability**, packaging validation proves **Kaggle loader survival**, tournament qualifiers prove **held-out stage promotion on final score**, submission proves **trained-weight packaging + upload legs**. Registry skips save wall clock on repeated configs; submission always re-validates trained weights. Until [#211](https://github.com/jmduea/orbit_wars/issues/211) lands in code, legacy paths remain in the repo but should be labeled **legacy** in operator guidance — not cited as competing defaults.

## When to Apply

- An operator or agent asks “what is the path from config to Kaggle submit?” or “which artifacts profile do I use?”
- Documentation work, runbooks, or `AGENT_CAPABILITIES.md` edits touch submit-valid, preflight, promotion, or bracket flows
- A session proposes `artifacts=hybrid_promotion`, Gate 5 `tournament-proof`, or `artifacts=bracket_training` as the **canonical** production path
- Planning teardown of hybrid/Gate-5-first/bracket-first narratives (R29)
- Scoping W&B sweeps — confirm sweep exit point is **after short preflight**, not full spine
- Before inventing thresholds: floors come from committed calibration JSON (`preflight-calibration.json`, future `qualifier-seed-calibration.json`), never operator-time round numbers

**Do not apply this as “already implemented”** until [#211](https://github.com/jmduea/orbit_wars/issues/211) units land; until then, point to the SSOT requirements + plan and mark existing hybrid/bracket/Gate-5 docs as legacy.

## Examples

### Before — lane B promotions / lane C as competing spines

Agent prompt: *“Prove this checkpoint is submit-valid.”*

Typical confused flow:

```
ow train ... artifacts=hybrid_promotion
  → ow eval status --watch
  → validation_ok in checkpoint_eval manifest

# OR, from a different doc:

ow benchmark tournament-proof ...

# OR, from bracket docs:

ow train ... artifacts=bracket_training
  → qualifier_eval async noop → random → nearest_sniper at 1.0
```

Three “canonical” orders, three threshold regimes (0.76 vs 1.0 vs hybrid poll), Docker at different stages.

### After — tournament qualifiers / submission on one spine

```
# Steps 1–2: config + make test-fast
# Step 3: short preflight (learning stability)
# Step 4: ow eval package --validate-docker on preflight checkpoint
# Step 5: long train (future: artifacts=ssot_pipeline)
#   tournament qualifiers (JAX, eval_seed_set) drive rollout curriculum
# Step 6: submission — trained weights + noop/random legs → ow eval submit
```

**W&B sweep — exit after short preflight:**

```
ow sweep / wandb agent ...
  → short preflight pass
  → STOP (no packaging validation or long train required)
```

## Related

- Requirements: [`docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`](../../brainstorms/2026-06-03-training-pipeline-ssot-requirements.md)
- Plan: [`docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`](../../plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md)
- Epic [#205](https://github.com/jmduea/orbit_wars/issues/205), implementation [#211](https://github.com/jmduea/orbit_wars/issues/211), perf [#204](https://github.com/jmduea/orbit_wars/issues/204)
- Legacy submit-valid funnel (superseded for spine, still operational until teardown): [`gate5-unified-tournament-submit-valid-funnel.md`](gate5-unified-tournament-submit-valid-funnel.md)
- Legacy bracket training slice (superseded for qualifier order/runtime): [`kaggle-bracket-ranking-foundational-slice.md`](kaggle-bracket-ranking-foundational-slice.md)
- Env parity CI (distinct concern; still valid): [`jax-comet-kaggle-parity-ci-gate.md`](jax-comet-kaggle-parity-ci-gate.md)
- Benchmark CLI package split (operator commands on spine): [`benchmark-cli-package-split-agent-native-parity.md`](benchmark-cli-package-split-agent-native-parity.md)
- Long CLI progress (stderr, no tail pipe): [`../developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md`](../developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md)
