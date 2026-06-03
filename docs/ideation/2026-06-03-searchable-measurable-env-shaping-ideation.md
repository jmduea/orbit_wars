---
date: 2026-06-03
topic: searchable-measurable-env-shaping
focus: searchable, measurable env shaping (ICML auto-env-shaping outer loop; Orbit Wars Hydra + preflight + tournament)
mode: repo-grounded
---

# Ideation: Searchable, Measurable Environment Shaping

## Grounding Context

Orbit Wars already implements heavy MDP shaping across Hydra groups (`reward/`, `task/`, `opponents/`, `curriculum/`, `training.reseed_*`, shield) with **inner-loop** fitness from calibrated preflight Gates 2–4 (JAX trend on `logs/*_jax.jsonl`) and **outer-loop** proof from Docker + unified tournament (Gate 5 / hybrid `checkpoint_eval`). Infrastructure is ~70% of an auto-env-shaping stack; gaps are a **unified searchable catalog**, **metric semantics guards**, and **bilevel scoring** (train shaped MDP → measure on reference opponents/tournament).

**Codebase context:** `src/config/schema.py`, `conf/config.yaml`, `src/jax/env.py`, `conf/benchmark/gates/`, `docs/benchmarks/preflight-calibration.json`, `ow benchmark` / `ow eval` primitives, existing W&B spaces (`reward_shaping.yaml`, `shield_modes.yaml`).

**Past learnings:** Never invent gate numbers; self-play ~50% is not a learning signal; Planet Flow sweep showed gameable objectives without activity floors; seed scheduler calibrated on **held-out eval**, not training self-play.

**External context:** ICML 2024 position (bilevel MDP design, joint ablation beats single-axis); EnvCoderBench (parallel candidate eval); Eureka (LLM reward code + reflection); transferable pattern = **reference vs training MDP** with cheap inner probes and expensive held-out ladder only for survivors.

## Topic Axes

1. Shaping catalog & versioning (searchable registry of MDP bundles)
2. Inner-loop fitness metrics (trend, denominators, decomposed shaping telemetry)
3. Held-out reference eval (noop/random/unified ladder as \(\mathcal{E}^{test}\))
4. Parallel candidate evaluation (batch gates, bracket tournaments, queue fan-out)
5. Joint multi-axis search (reward × opponents × reseed × curriculum interactions)

## Ranked Ideas

### 1. Env Shaping Catalog + Run Fingerprint

**Description:** Introduce `conf/shaping_profiles/*.yaml` (or `conf/shaping/catalog/`) as named, versioned MDP bundles (`reward` + `task` + `opponents` + `curriculum` + `reseed`). Every train writes `artifacts/shaping_manifest.json` (profile id, SHA256 of resolved shaping subtree, calibration panel id). Add primitives `ow shaping list`, `ow shaping diff`, and include fingerprint in `ow runs show` / `make agent-context` so agents search campaigns by env vector, not run names.
**Axis:** Shaping catalog & versioning
**Basis:** `direct:` MDP knobs scattered across `conf/` groups; gate YAML and `conf/sweep_arm/*` duplicate overrides; `outputs/campaigns/` layout in AGENTS.md
**Rationale:** Outer-loop search requires enumerable, diffable candidates—the MIT paper’s \(f(\mathcal{E})\) needs a first-class representation in-repo.
**Downsides:** Profile drift vs gate recipes must stay synced; migration work for existing sweeps.
**Confidence:** 82%
**Complexity:** Medium
**Status:** Unexplored

### 2. Decomposed Shaping Telemetry → Calibrate → Gate Chain

**Description:** Log `shaping_reward` and per-term sums (`reward_capture_planet`, `reward_ship_delta`, `reward_production_delta`) into `logs/*_jax.jsonl` via `metric_contract.py`. Add `ow benchmark calibrate-reward-shaping` (mirror seed-scheduler) → `docs/benchmarks/reward-shaping-calibration.json`. Wire `conf/benchmark/gates/reward_shaping_signal.yaml` to calibrated floors and **forbid** gating on raw `episode_reward_mean` when shaping is active.
**Axis:** Inner-loop fitness metrics
**Basis:** `direct:` `JaxStepResult` already computes terms in `src/jax/env.py` but omits them from logged scalars; seed-scheduler calibration pattern in `docs/benchmarks/seed-scheduler-calibration.json`
**Rationale:** Makes shaping **measurable** in the training loop before tournament spend; compounds every future search.
**Downsides:** Metric registry + golden test updates; calibration campaign GPU cost once.
**Confidence:** 78%
**Complexity:** Medium
**Status:** Unexplored

### 3. Two-Stage Reference Sandwich (`ow benchmark shaping-holdout`)

**Description:** New primitive: Stage A = short noop/random rollout under a **frozen reference profile** from calibration (cheap beatability); Stage B = existing `ow benchmark tournament-proof` only if A passes calibrated inner thresholds. Records both training profile and reference profile on the report. Aligns train-on-shaped / score-on-unshaped with Gate 5 ordering (Docker still first in submit-valid paths).
**Axis:** Held-out reference eval
**Basis:** `direct:` Gates 2–4 vs Gate 5 split; `external:` ICML auto-env-shaping eval on \(\mathcal{E}^{test}\)
**Rationale:** Rejects bad shaping combos before tournament compute; makes “reference MDP” explicit in agent loops.
**Downsides:** Must pin reference panel version with calibration commit; risk of false negatives if Stage A too harsh.
**Confidence:** 75%
**Complexity:** Medium
**Status:** Unexplored

### 4. Dual-Contract Success: Train Trend ∧ Tournament Transfer

**Description:** Define shaping-search success only when (a) Gates 2–4 pass on the training MDP **and** (b) unified combined noop/random (or full ladder) improves vs baseline at a fixed checkpoint cadence. Auto-flag “shaping lie” when (a) passes and (b) fails. Extend `ow benchmark gate run` report and hybrid status JSON with explicit metric context (opponent mix, reward mode, trigger field).
**Axis:** Inner-loop fitness metrics
**Basis:** `direct:` AGENTS.md metric-gates warnings; Planet Flow gameable-objective learning doc; unified tournament enforcement in preflight JSON
**Rationale:** Stops outer loops from optimizing JSONL trends that Gate 5 disproves—the core measurable env-shaping failure mode.
**Downsides:** More runs per candidate; needs disciplined baseline checkpoint.
**Confidence:** 80%
**Complexity:** Low–Medium
**Status:** Unexplored

### 5. Factorized Joint Search (`ow make` × Shaping Profiles)

**Description:** Extend `ow make` to emit sweep arms as a Cartesian product of catalogued profiles across reseed × reward profile × opponent family × shield tier, each arm tagged with axis metadata. Attach `ow benchmark gate run --dry-run` per arm. Replaces hand-writing a dozen `conf/sweep_arm/*.yaml` files for interaction effects (e.g. reseed 50 × curriculum staged).
**Axis:** Joint multi-axis search
**Basis:** `direct:` `conf/wandb_sweep/space/post_encoder_once_overnight.yaml`; `external:` ICML joint ablation failure modes (Eureka combined shaping collapse)
**Rationale:** Single-axis sweeps miss coupled MDP effects; this is the searchable outer loop without new PPO math.
**Downsides:** W&B cost; must cap grid size; compile-cache invalidation across shield/feature arms.
**Confidence:** 72%
**Complexity:** Medium
**Status:** Unexplored

### 6. `ow benchmark shape-calibrate` (MDP Outer Loop Operator)

**Description:** Agent-native calibrator: subprocess N short trains over a declared shaping grid (from catalog or W&B space), `--analyze-only` pass, emit `docs/benchmarks/shaping-calibration.json` with winning bundles and pinned `run_dir`/`log_path` per cell—same contract as `calibrate-seed-scheduler` / `calibrate-unified-tournament`.
**Axis:** Joint multi-axis search
**Basis:** `direct:` existing `ow benchmark calibrate-*` primitives; repo scan notes missing “env config → short train → score → next” loop
**Rationale:** Turns ad-hoc shaping experiments into a repeatable, documented decision with calibrated floors—not invented thresholds.
**Downsides:** Campaign design burden; joint grids explode without factorial limits.
**Confidence:** 70%
**Complexity:** Medium–High
**Status:** Explored → `docs/brainstorms/2026-06-03-shape-calibrate-requirements.md`

### 7. Parallel Shaping Brackets via `checkpoint_eval`

**Description:** From one policy checkpoint, queue multiple `checkpoint_eval` jobs differing only in Hydra shaping overrides (reward profile, shield, format mix). Same weights, multiple MDPs—tournament ladder picks the shaping winner. Preserves Docker → tournament → promote order; caps concurrent jobs for one-GPU hosts.
**Axis:** Parallel candidate evaluation
**Basis:** `direct:` hybrid `checkpoint_eval` composite job; `ow eval status` / `ow eval worker` primitives
**Rationale:** Separates policy regression from MDP regression without full retrain—high leverage for bilevel search.
**Downsides:** Eval queue contention; only applies post-checkpoint, not pre-train MDP design.
**Confidence:** 68%
**Complexity:** Medium
**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Auto-sync AGENTS threshold block only | Too narrow; absorbed into calibration operator pattern (#6) |
| 2 | `ow benchmark gate run --batch` alone | Strong but secondary to catalog+fingerprint (#1) |
| 3 | Tournament-first with frozen random policy | Too expensive / weak signal for board-game PPO |
| 4 | YAML-ban / `synthesize-reward` only | Extreme; high integration cost; partial overlap #1 |
| 5 | Micro-tournament on every genome | Tournament cost prohibitive; use reference sandwich (#3) instead |
| 6 | Logistic surrogate without PPO | Weak `reasoned:` basis; risky false positives |
| 7 | Parametric curriculum surface (full) | High complexity; defer until catalog exists |
| 8 | Gate YAML as “patches” narrative | Already how gates work—not net-new |
| 9 | 16-slot GPU scheduler (underspecified) | Ops vague; bracket eval (#7) more grounded |
| 10 | CMA-ES on frozen policy only | Narrow reward-only search; misses joint axes |
| 11 | Seeds as primary search axis | Interesting but borderline scope; panel versioning in #3 covers holdout stability |
| 12 | Shaping microbench tier-1 only | Good tactical add-on; lower leverage than telemetry chain (#2) |
| 13 | Eureka repair brief codegen | Valuable follow-on; depends on catalog + eval manifests (#1) |
| 14 | Remove long trains entirely | Over-stated; smoke+gates is policy choice, not product requirement |
| 15 | OpenSpiel portfolio narrative only | Documentation clarity, not new capability |

## Related documentation

- Solutions (operator design): `docs/solutions/developer-experience/shape-calibrate-env-shaping-calibration-operator.md`
- Requirements: `docs/brainstorms/2026-06-03-shape-calibrate-requirements.md`
- Plan: `docs/plans/2026-06-03-003-feat-shape-calibrate-plan.md`
- Prior art in-repo: `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`, `docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md`
