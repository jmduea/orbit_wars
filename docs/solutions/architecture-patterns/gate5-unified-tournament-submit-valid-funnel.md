---
title: Gate 5 unified tournament proof with Docker-first submit-valid funnel
date: 2026-06-03
category: architecture-patterns
module: artifacts-tournament
problem_type: architecture_pattern
component: development_workflow
severity: high
applies_when:
  - "Proving a checkpoint is submit-valid before Kaggle upload or hybrid promotion"
  - "Running Gate 5 or ow benchmark tournament-proof on a held-out ladder"
  - "Configuring Stage 2 incumbent for unified 2p+4p combined scoring"
tags:
  - gate5
  - unified-tournament
  - submit-valid
  - docker-validation
  - nearest-sniper
  - preflight-calibration
  - hybrid-promotion
related_components:
  - src/artifacts/submit_valid_funnel.py
  - src/artifacts/checkpoint_eval.py
  - src/artifacts/tournament/unified/
  - src/artifacts/tournament/unified/incumbent.py
  - docs/benchmarks/preflight-calibration.json
  - conf/artifacts/hybrid_promotion.yaml
---

# Gate 5 unified tournament proof with Docker-first submit-valid funnel

## Context

Kaggle submission requires packaging that runs in the competition Docker image **and** held-out win rates against baseline opponents. Earlier Gate 5 used separate 2p-only floors and could treat the challenger checkpoint as its own Stage 2 incumbent. PR [#186](https://github.com/jmduea/orbit_wars/pull/186) ships a **unified 2p+4p combined ladder**, **calibrated Stage-1 floors** in `docs/benchmarks/preflight-calibration.json`, and a **submit-valid order** that never spends tournament compute on checkpoints that fail Docker validation.

## Guidance

### Submit-valid order (always)

1. **Docker packaging** — `run_submit_valid_docker_gate` / `ow eval package --validate-docker` must return `validation_ok: true`.
2. **Unified tournament ladder** — held-out seeds, combined 2p+4p score, Stage 1 noop/random floors, optional Stage 2 vs incumbent.
3. **Upload / promote** — only after both pass (`ow eval submit`, hybrid manifest promotion).

`run_checkpoint_eval_job` enforces this: on Docker failure it returns `tournament_skipped: true` and does not run the ladder.

```python
# src/artifacts/checkpoint_eval.py (simplified)
docker_manifest = run_submit_valid_docker_gate(...)
if not docker_gate_passed(docker_manifest):
    return {"validation_ok": False, "tournament_skipped": True, ...}
run_tournament_promotion_job(job, result_dir=tournament_dir)
```

Primitives: `ow eval package --validate-docker`, `ow benchmark tournament-proof`, hybrid `checkpoint_eval` when `artifacts=hybrid_promotion`.

### Unified ladder and calibrated floors

- **Combined score** across 2p and 4p formats (`src/artifacts/tournament/unified/scoring.py`).
- **Stage 1:** noop and random must meet `noop_min_combined` / `random_min_combined` (default **0.76** with `enforcement: true` in calibration JSON).
- **Stage 2:** challenger vs incumbent at per-seed **100%** win requirement when a promoted incumbent exists.
- Recalibrate floors with `ow benchmark calibrate-unified-tournament` before changing JSON thresholds — do not relax floors to make a failing run pass.

Spec: `docs/brainstorms/2026-06-03-gate5-unified-tournament-requirements.md`. Plan: `docs/plans/2026-06-03-001-feat-gate5-unified-tournament-plan.md`.

### Bootstrap incumbent is scripted nearest_sniper (not the checkpoint)

Until a campaign has a promoted manifest, Stage 2 resolves the incumbent via **`incumbent_bootstrap_opponent: nearest_sniper`** in calibration JSON — a **scripted baseline**, not the challenger pickle path.

```python
# src/artifacts/tournament/unified/incumbent.py
def resolve_incumbent(spec, *, campaign, output_root):
    if campaign:
        incumbent = resolve_promoted_agent(campaign, str(output_root))
        if incumbent is not None:
            return incumbent
    if spec.incumbent_bootstrap_opponent is not None:
        return agent_from_baseline(spec.incumbent_bootstrap_opponent, agent_id="incumbent")
    return None
```

After a successful Stage 2 pass, `swap_incumbent_on_unified_pass` writes the promoted manifest for the campaign.

### Operator proof command

```bash
uv run ow benchmark tournament-proof \
  --eval-checkpoint outputs/.../jax_ckpt_last.pkl \
  --verbose \
  --out /tmp/gate5.json
```

Use stderr progress (`--verbose`); keep stdout for final JSON. Post-merge verification may still be **NOT_VERIFIED** until a post-hygiene checkpoint clears Stage 1 (e.g. noop combined 0.75 vs 0.76 floor) — see PR #186 **Remaining work**.

## Why This Matters

Running tournaments before Docker validation wastes GPU/time on unpublishable checkpoints. Using the challenger as incumbent made Stage 2 self-referential and hid real regression vs a fixed scripted opponent. Unified combined scoring aligns Gate 5 with how 2p and 4p both matter on Kaggle.

## When to Apply

- Configuring `artifacts=hybrid_promotion` or `conf/benchmark/gates/win_proof_tournament.yaml`
- Debugging `checkpoint_eval` jobs under `evaluations/checkpoint_eval_u*/`
- Interpreting `unified_verdict.json` and `validation_ok` in eval status JSON

## Examples

**Wrong:** Tournament-only proof on a checkpoint that fails `kaggle_environments` Docker smoke.

**Right:** `validation_ok: true` in `docker_manifest.json`, then `unified_verdict.json` with `passed: true` at calibrated floors.

**Wrong:** Pointing Stage 2 at the same `.pkl` being evaluated.

**Right:** No promoted manifest → `nearest_sniper` baseline; after promotion → manifest checkpoint from `resolve_promoted_agent`.

## Related

- Bracket training qualifier slice (1.0 floors, separate from Gate 5 0.76 proof): `docs/solutions/architecture-patterns/kaggle-bracket-ranking-foundational-slice.md`
- Long CLI progress (stderr, no tail pipe): `docs/solutions/developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md`
- Subprocess train streaming (calibration arms): `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`
- Operator prompts: `docs/AGENT_CAPABILITIES.md`
- Incumbent fix plan: `docs/plans/2026-06-03-004-fix-incumbent-nearest-sniper-bootstrap-plan.md`
