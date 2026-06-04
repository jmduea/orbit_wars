---
title: SSOT training pipeline from config to Kaggle submission
date: 2026-06-03
last_updated: 2026-06-04
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
  - wandb
  - sweep-preflight
  - teardown-legacy
  - operator-docs
  - github-211
  - interactive-flowchart
  - svg-layout
related_components:
  - docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md
  - docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md
  - docs/tools/ssot-training-pipeline-flowchart.html
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

The fix started as **documentation and planning** (requirements SSOT, plan, flowchart, [#211](https://github.com/jmduea/orbit_wars/issues/211) under epic [#205](https://github.com/jmduea/orbit_wars/issues/205)). **Foundation slice (PR [#212](https://github.com/jmduea/orbit_wars/pull/212))** implements U1–U4: seed partition, W&B preflight sweep recipe + scoring, packaging CLI flags, `artifacts=ssot_pipeline` stub. U5–U8 (JAX qualifiers, calibration, bracket/submission, legacy teardown) remain open on #211.

**Operator map (interactive).** [`docs/tools/ssot-training-pipeline-flowchart.html`](../../tools/ssot-training-pipeline-flowchart.html) is the click-through spine: R# labels on nodes, aside panel with requirement text, and side paths for fail/retry/terminal outcomes. The plan mermaid in [`docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`](../../plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md) must stay in sync with this chart (gates, loops, terminals).

## Guidance

**One spine, plain names.** Treat [`docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`](../../brainstorms/2026-06-03-training-pipeline-ssot-requirements.md) as the single source of truth from config to Kaggle submission. Operator and agent docs should describe **only this spine**; anything else is an explicit alternate track (Planet Flow research, launch-hygiene perf gates), not a footnote on the spine.

**Canonical spine order (six phases):**

1. **Config setup** — resolved Hydra config, feature compatibility declared
2. **Preliminary tests** — wiring tier (`make test-fast`); failure blocks everything downstream
3. **W&B sweep · short preflight** — coordinate ablations via `ow make wandb_sweep`, `ow sweep create`, `wandb agent`; Gates 2–3 per run; checkpoint **artifacts** in W&B (no local config registry or bad-config cache)
4. **Packaging validation** — Docker + `kaggle_environments` smoke on the **sweep winner checkpoint** (seed 0, 4-player, all agents = packaged agent)
5. **Long train** — ≤500M env steps with **W&B observability**; rollout curriculum (random → noop-heavy → sniper-heavy) advanced by **tournament qualifiers** (fast JAX held-out eval on `eval_seed_set` only); then **main bracket** (μ/σ)
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

**W&B on the SSOT spine.** Step 3 **is** the W&B sweep — short preflight runs, metrics, and checkpoint artifacts. Failed runs stay in W&B; operator selects a **winner** for packaging validation. Long train (step 5) runs with `telemetry.wandb` for observability and artifact lineage. **Sweep-only ablations** may stop after preflight pass without packaging or long train.

**Implementation tracker.** Requirements → plan [`docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`](../../plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md) → interactive flowchart → [#211](https://github.com/jmduea/orbit_wars/issues/211). Teardown policy (R29): remove or relocate legacy spines; no “demoted but still default” paths in `AGENTS.md` / `docs/AGENT_CAPABILITIES.md`.

**Flowchart control-flow invariants** (plan mermaid + HTML must match):

| Path | From | To | Meaning |
|------|------|-----|---------|
| fail → stop | Tests pass? | Stop · fix locally (left attach) | Step 2 CPU gate |
| fail run | Gates 2–3 + winner? | W&B sweep | Pick next sweep agent / candidate |
| fail → pick next run | Smoke ok? | W&B sweep (right rail) | Packaging fail on winner ckpt |
| retry | Stage 3 clear? | Long train (left rail) | Qualifier not cleared; keep training |
| weak_config stop | Stage 3 clear? | Terminal (left attach) | 500M exhaust — **no return to sweep** |
| yes | Stage 3 clear? | Main bracket | All three legs cleared |

**SVG layout conventions** for dense operator flowcharts (learned building the SSOT chart):

1. **Wider canvas** — `viewBox` ~720×900; spine centered (~x=360); nodes need ~18px vertical gaps minimum.
2. **Outer side rails** — left x≈8, right x≈712; route loops through gaps, never through node columns (x≈160–560).
3. **Terminal attach** — stop nodes connect on the **left** edge from their gate (short horizontal), not a loop around to the right.
4. **Opaque layers** — draw edges first, then white `masks` rects per node bbox, then styled nodes (semi-transparent fills let edges bleed through).
5. **Label alignment** — use plan **KTD#** on flowchart nodes (not origin KD#); gate names must match plan (`Gates 2–3 + winner?`, `Stage 3 clear?`).

**What didn't work (layout iteration).**

- Narrow viewBox with side paths at x=40/x=400 → paths crossed wide W&B and packaging boxes.
- Semi-transparent node fills (`rgba(...)`) → edges visible “under” nodes even with correct z-order.
- Vertical segments at x=282 inside the node column → retry path cut through Tournament qualifiers.
- Plan mermaid `weak_config → W&B` contradicted flowchart terminal and R12/AE4 — doc-review caught it.

**Competition rules SSOT.** Game rules and packaging expectations live in [`docs/competition/COMPETITION_OVERVIEW.md`](../../competition/COMPETITION_OVERVIEW.md) and [`docs/competition/COMPETITION_SUBMISSION.md`](../../competition/COMPETITION_SUBMISSION.md); pipeline docs link there first.

## Why This Matters

Parallel submit-valid paths waste GPU time and agent turns on the wrong funnel. Agents trained on `hybrid_promotion` poll contracts may never run packaging validation on the preflight checkpoint; agents following Gate 5 may treat 0.76 proof floors as production promotion gates while bracket docs demand 1.0 qualifier floors — same words, different semantics. Without one spine, “submit-valid” means different things depending on which doc was indexed last.

The SSOT doc separates concerns cleanly: W&B sweep preflight proves **learning stability** and ranks config candidates, packaging validation proves **Kaggle loader survival**, tournament qualifiers prove **held-out stage promotion on final score**, submission proves **trained-weight packaging + upload legs**. W&B holds sweep history and artifacts — no local fingerprint cache. Submission always re-validates trained weights. Legacy paths remain in the repo until U8 teardown — label **legacy** in operator guidance, not competing defaults. Full spine completion tracks [#211](https://github.com/jmduea/orbit_wars/issues/211).

## When to Apply

- An operator or agent asks “what is the path from config to Kaggle submit?” or “which artifacts profile do I use?”
- Documentation work, runbooks, or `AGENT_CAPABILITIES.md` edits touch submit-valid, preflight, promotion, or bracket flows
- A session proposes `artifacts=hybrid_promotion`, Gate 5 `tournament-proof`, or `artifacts=bracket_training` as the **canonical** production path
- Planning teardown of hybrid/Gate-5-first/bracket-first narratives (R29)
- Scoping W&B sweeps — step 3 **is** the sweep on SSOT spine; confirm artifact handoff winner → packaging → long train
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
# Step 3: W&B sweep short preflight (Gates 2–3 per agent)
uv run ow make wandb_sweep=ssot_preflight
uv run ow sweep create --config conf/wandb_sweep/ssot_preflight.yaml
wandb agent <entity>/<project>/<sweep_id>
# Step 4: ow eval package --validate-docker on W&B winner checkpoint
# Step 5: long train (future: artifacts=ssot_pipeline, telemetry.wandb.enabled=true)
#   tournament qualifiers (JAX, eval_seed_set) drive rollout curriculum
# Step 6: submission — trained weights + noop/random legs → ow eval submit
```

**Sweep-only ablation — stop after preflight pass:**

```
ow sweep / wandb agent ...
  → short preflight pass
  → STOP (no packaging validation or long train required)
```

Open the operator map: [`docs/tools/ssot-training-pipeline-flowchart.html`](../../tools/ssot-training-pipeline-flowchart.html) — click nodes for R# text and CLI snippets.

**Plan ↔ flowchart sync** (verify after editing either file):

- Step 2 fail → **Stop · fix locally** (not GPU)
- Packaging fail → **W&B sweep** (pick next winner)
- `weak_config` → **terminal stop** (not back to W&B)
- **Stage 3 clear?** yes → bracket; retry → long train

## Related

- Interactive operator map: [`docs/tools/ssot-training-pipeline-flowchart.html`](../../tools/ssot-training-pipeline-flowchart.html)
- Requirements: [`docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`](../../brainstorms/2026-06-03-training-pipeline-ssot-requirements.md)
- Plan: [`docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`](../../plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md)
- Epic [#205](https://github.com/jmduea/orbit_wars/issues/205), implementation [#211](https://github.com/jmduea/orbit_wars/issues/211), perf [#204](https://github.com/jmduea/orbit_wars/issues/204)
- Legacy submit-valid funnel (superseded for spine, still operational until teardown): [`gate5-unified-tournament-submit-valid-funnel.md`](gate5-unified-tournament-submit-valid-funnel.md)
- Legacy bracket training slice (superseded for qualifier order/runtime): [`kaggle-bracket-ranking-foundational-slice.md`](kaggle-bracket-ranking-foundational-slice.md)
- Env parity CI (distinct concern; still valid): [`jax-comet-kaggle-parity-ci-gate.md`](jax-comet-kaggle-parity-ci-gate.md)
- Benchmark CLI package split (operator commands on spine): [`benchmark-cli-package-split-agent-native-parity.md`](benchmark-cli-package-split-agent-native-parity.md)
- Long CLI progress (stderr, no tail pipe): [`../developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md`](../developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md)
