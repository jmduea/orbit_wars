---
title: "feat: SSOT training pipeline — config to submission"
type: feat
status: active
date: 2026-06-03
origin: docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md
---

# feat: SSOT training pipeline — config to submission

## Summary

Implement the single canonical pipeline from the SSOT requirements doc: **preflight → packaging validation (preflight checkpoint) → long train → tournament qualifiers (JAX) → main bracket → submission**. Adds a config registry, disjoint train/eval seeds, a new Hydra train profile, and tears down legacy `hybrid_promotion` / `bracket_training` / Gate-5-first spines from operator docs.

Tracker: GitHub #205. Perf dependency: #204.

---

## Problem Frame

Operators and agents face parallel submit-valid narratives (`default`, `hybrid_promotion`, `bracket_training`, Gate 5, async `qualifier_eval`). The SSOT requirements doc defines one spine with plain names: **long train**, **tournament qualifiers**, **submission**. This plan implements that spine and removes competing defaults. (see origin: `docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`)

---

## Requirements

Traceability to origin R-IDs (subset — full list in origin doc):

| ID | Plan coverage |
|----|----------------|
| R1–R2 | U1, U8 — six-step spine + operator docs |
| R3–R8 | U3 — packaging validation |
| R5–R6, R9–R11 | U1, U3, U4 — registry + short preflight (Gates 2–3 family) + packaging ordering |
| R12–R20 | U4, U5, U6, U7 — long train, tournament qualifiers, bracket MVP |
| R21–R22 | U7 — submission |
| R23–R25 | U2, U5 — seed partition + JAX parity for qualifiers |
| R26–R28 | U6 — calibration JSON |
| R29–R31 | U8 — teardown + tracker |

---

## Key Technical Decisions

**KTD1 — Terminology in code and docs.** User-facing spine: **packaging validation**, **long train**, **tournament qualifiers**, **submission**. Internal module names may retain legacy tokens during migration but CLI/help and operator docs use SSOT names only.

**KTD2 — Config registry as local JSON + CLI.** Store under `outputs/config_registry/registry.json` (gitignored) with optional seed copy in `docs/benchmarks/config-registry-dogfood.json` for tests. Fingerprint hashes resolved config inference surface + invalidation dimensions (R5). Top-level primitive: `ow registry list|show|record` (matches `ow eval`, `ow runs` pattern).

**KTD3 — Preflight checkpoint drives packaging validation.** Extend `ow eval package --validate-docker` to accept `--checkpoint` from short preflight output (existing path) with seed-0 / 4p-only mode matching R7. No random-weight packaging path.

**KTD4 — Seed partition replaces `heldout_eval_seed_set` for training.** Add `training_seed_set` and `eval_seed_set` to `src/config/schema.py`; `SeedScheduler` draws reseeds only from `training_seed_set`; tournament qualifiers consume `eval_seed_set` exclusively. Remove training use of `heldout_eval_seed_set` (R29).

**KTD5 — Tournament qualifiers = new JAX harness, not async Docker `qualifier_eval`.** Fast held-out match runner in `src/jax/eval/` (or `src/jax/tournament_qualifiers/`) invoked from training loop on checkpoint ticks. Promotion metric: final-score win fraction (R18), wired to `_terminal` in `src/jax/env.py`. No Docker per tick.

**KTD6 — Hydra profile `artifacts=ssot_pipeline`.** Single composition replacing production use of `default` / `hybrid_promotion` / `bracket_training` for the canonical path. Long train enables rollout curriculum stages R15–R17 and tournament qualifier ticks.

**KTD7 — Qualifier calibration before enforcement.** Ship `docs/benchmarks/qualifier-seed-calibration.json` via `ow benchmark calibrate-qualifier-seeds` before R19 floors are enforced in production profile. Until committed, profile uses conservative interim rules (R19).

**KTD8 — Teardown is hard delete of production defaults.** Sole-operator context: remove legacy YAML from default train path, strip hybrid/Gate-5-first narratives from `AGENTS.md` / `docs/AGENT_CAPABILITIES.md`, close the loop on #205 sub-issues. No phased deprecation labels (origin KD7).

**KTD9 — Bracket MVP reuses plan 005 artifacts where possible.** `src/artifacts/tournament/bracket/` state + μ/σ updates from plan 005 U1–U5; SSOT changes *when* entrants qualify (tournament qualifiers during long train, not Docker noop-first async ladder). Plan 005 superseded for qualifier order and eval runtime (origin KD6).

---

## High-Level Technical Design

### Canonical spine

```mermaid
flowchart TD
  PF[Short preflight]
  PV[Packaging validation]
  LT[Long train]
  TQ[Tournament qualifiers JAX]
  MB[Main bracket]
  SUB[Submission]
  PF --> PV --> LT
  LT --> TQ
  TQ -->|stage promote| LT
  TQ -->|stage 3 clear| MB
  MB --> SUB
```

### Training loop integration

```mermaid
sequenceDiagram
  participant Train as long train loop
  participant Reg as config registry
  participant TQ as tournament qualifiers
  participant Br as bracket state

  Train->>Reg: skip packaging if known good
  Train->>TQ: on tick eval_seed_set only
  TQ-->>Train: stage promotion / fail
  Train->>Br: on stage 3 clear
```

---

## Implementation Units

### U1. Config registry

**Goal.** Persistent fingerprint → preflight/packaging pass-fail with invalidation dimensions (origin R5–R6, R10–R11).

**Requirements.** R5, R6, R10, R11; AE1, AE2.

**Dependencies.** None.

**Files.**
- Create `src/config_registry/` (`fingerprint.py`, `store.py`, `schema.py`)
- Create `src/cli/registry.py`; wire in `src/cli/__init__.py` as `ow registry`
- Tests: `tests/test_config_registry.py`

**Approach.** Hash inference/runtime surface from resolved Hydra config + model schema version. Record pass/fail, timestamps, code SHA, competition doc revision. Lookup before step 3/4.

**Test scenarios.**
- Covers AE1. Known good fingerprint + valid invalidation → skip steps 3–4.
- Covers AE2. Known fail fingerprint → reject before long train.
- Invalidation drift (code SHA change) → re-smoke required.
- Test expectation: none for pure schema module if covered by integration tests above.

**Verification.** `ow registry` CRUD works; unit tests pass; dogfood one pass + one fail entry (origin success criteria).

---

### U2. Train / eval seed partition

**Goal.** Disjoint `training_seed_set` and `eval_seed_set`; remove training reseed from `heldout_eval_seed_set` (origin R14, R25, R29).

**Requirements.** R14, R25; AE6.

**Dependencies.** None (parallel with U1).

**Files.**
- Modify `src/config/schema.py`, `conf/config.yaml`
- Modify `src/training/seed_scheduler.py`
- Modify `src/jax/train/loop.py` (pass seed sets)
- Tests: `tests/test_seed_scheduler.py`, extend `tests/test_jax_env_parity.py` or new `tests/test_eval_seed_contamination.py`

**Approach.** Default `eval_seed_set` to held-out list; assert disjoint at train start. CI test: eval seed in reseed pool → fail (AE6).

**Test scenarios.**
- Covers AE6. Eval seed in training reseed → build/test fails.
- `training_seed_set ∩ eval_seed_set = ∅` enforced at config resolve.
- Reseed draws only from `training_seed_set` when set.

**Verification.** `make test-fast` green; contamination test fails when violated.

---

### U3. Packaging validation (preflight checkpoint)

**Goal.** Docker smoke after preflight using saved checkpoint, seed 0, 4p identical agents (origin R4, R7–R8).

**Requirements.** R3, R4, R7, R8, R9, R11.

**Dependencies.** U1 (record results).

**Files.**
- Modify `src/artifacts/kaggle_submission.py`, `src/cli/eval.py`
- Modify `scripts/validate_kaggle_docker_submission.py` or fold into `ow eval package` only
- Tests: `tests/test_eval_package_validate_docker.py` (extend or create)

**Approach.** After short preflight, invoke `ow eval package --checkpoint <preflight_ckpt> --validate-docker` with 4p/seed-0-only defaults per R7. Integrate registry record on pass/fail.

**Test scenarios.**
- Packaging validation runs only after preflight pass (ordering).
- Pass records registry; fail blocks long train.
- Checkpoint load exercises trained weights from preflight (not random init).

**Verification.** One local Docker smoke succeeds on dogfood config (operator machine with Docker).

---

### U4. SSOT Hydra profile and long train entry

**Goal.** Single `artifacts=ssot_pipeline` composition for long train with rollout curriculum stages (origin R12–R13, R15–R17).

**Requirements.** R12, R13, R15, R16, R17.

**Dependencies.** U2, U3.

**Files.**
- Create `conf/artifacts/ssot_pipeline.yaml`
- Modify `src/jax/train/loop.py`, `src/jax/train/bracket_training.py` (refactor or replace curriculum hooks)
- Modify `conf/config.yaml` defaults documentation
- Tests: `tests/test_ssot_pipeline_config.py`

**Approach.** Rollout opponent mix follows random → noop-heavy → sniper-heavy based on current qualifier stage. Env-step budget 500M with exhaustion → bad config + registry (R12). **Short preflight** reuses existing Gates 2–3 infrastructure (`src/jax/preflight_gate_loader.py`, `docs/benchmarks/preflight-calibration.json`); saves checkpoint path consumed by U3.

**Test scenarios.**
- Resolved config selects ssot_pipeline composition.
- Stage 1 opponents predominantly random at train start.
- 500M exhaustion marks bad config when stage 3 uncleared (AE4 shape).

**Verification.** `uv run ow train artifacts=ssot_pipeline training.total_updates=5` smoke starts without legacy hybrid hooks.

---

### U5. Tournament qualifiers (JAX)

**Goal.** Checkpoint-tick held-out JAX eval for stage promotion using final-score wins (origin R18–R19, KD3).

**Requirements.** R18, R19, R23, R24; AE3 (illustrative).

**Dependencies.** U2, U4.

**Files.**
- Create `src/jax/tournament_qualifiers/` (`runner.py`, `promotion.py`, `metrics.py`)
- Modify `src/jax/train/loop.py` (tick hook)
- Modify `src/jax/env.py` (expose terminal final score for eval aggregation if needed)
- Tests: `tests/test_tournament_qualifiers.py`, golden parity with `tests/test_jax_env_parity.py`

**Approach.** On interval, run N games per leg on `eval_seed_set` only. Aggregate win fraction from `_terminal` final score. Compare to calibration JSON floors (interim conservative rules until U6). Emit promotion events to shift rollout stage.

**Test scenarios.**
- Promotion uses final-score wins, not rollout JSONL `overall_win_rate`.
- Eval seeds never appear in rollout batch (integration with U2).
- Block promotion when calibration JSON missing (R19 interim rules).

**Verification.** Unit tests with fixed seeds; promotion event logged in metrics JSONL.

---

### U6. Qualifier seed calibration

**Goal.** Committed `qualifier-seed-calibration.json` and calibration primitive (origin R26–R28).

**Requirements.** R26, R27, R28.

**Dependencies.** U5 (needs promotion API to calibrate against).

**Files.**
- Create `src/cli/benchmark_calibrate_qualifier_seeds.py` (or extend `src/cli/benchmark.py`)
- Create `docs/benchmarks/qualifier-seed-calibration.json` (after campaign)
- Tests: `tests/test_qualifier_calibration_loader.py`

**Approach.** Campaign runs fixed checkpoints vs legs; commit floors + seed counts. Loader used by U5 promotion. Do not relax floors ad hoc (R28).

**Test scenarios.**
- Loader reads committed JSON; missing file → interim conservative mode.
- Calibration JSON schema validated in CI once committed.

**Verification.** `ow benchmark calibrate-qualifier-seeds --help` documents campaign; loader unit tests pass.

---

### U7. Bracket MVP and submission

**Goal.** Main bracket entry after stage 3; submission with trained weights + opponent legs (origin R20–R22).

**Requirements.** R20, R21, R22; AE5.

**Dependencies.** U5.

**Files.**
- Reuse/adapt `src/artifacts/tournament/bracket/` from plan 005
- Modify `src/cli/eval.py` (`ow eval submit`, package paths)
- Tests: `tests/test_submission_requirements.py`, bracket transition tests

**Approach.** Stage 3 clear → write bracket state, enable self-play hook (MVP). Submission: trained checkpoint Docker smoke + noop/random legs at trained weights (R21). Separate from packaging validation (preflight ckpt).

**Test scenarios.**
- Covers AE5. Bracket-qualified checkpoint → submission path allowed after Docker pass.
- Tournament qualifier clearance alone insufficient for submission without trained-weight smoke (R22).

**Verification.** Submission CLI documents SSOT order; integration test mocks Docker where needed.

---

### U8. Legacy teardown and operator docs

**Goal.** Remove parallel spines; point operators/agents to SSOT only (origin R2, R29–R31).

**Requirements.** R2, R29, R30, R31.

**Dependencies.** U1–U7 (profile must exist before deleting defaults).

**Files.**
- Remove or relocate `conf/artifacts/hybrid_promotion.yaml`, `conf/artifacts/bracket_training.yaml` from production docs (delete or move to `conf/artifacts/_legacy/`)
- Modify `AGENTS.md`, `docs/AGENT_CAPABILITIES.md`, `docs/README.md`, `docs/ONBOARDING.md`
- Modify agent capability tests if present
- Update GitHub #205 epic body via `gh issue edit` (manual step in verification)

**Approach.** Hard teardown: `uv run ow train` without profile targets `ssot_pipeline` or errors with pointer to SSOT doc. Demote `learn-proof` composer from primary workflow. Annotate superseded plans in `docs/plans/2026-06-03-005-*.md` headers only (do not delete plan files).

**Test scenarios.**
- Agent capability map lists SSOT primitives only (origin success criteria).
- Import/hydra smoke: default train does not enable hybrid_promotion funnel.

**Verification.** `make test-fast` green; docs link to SSOT requirements; #204 referenced as long-train dependency.

---

## Scope Boundaries

**Deferred for later** (from origin — unchanged)
- Wilson/binomial formula details (calibration campaign)
- Full bracket async round-robin worker (plan 005 U7–U8)
- Gate 4 `curriculum_staged` as non-Kaggle research track

**Deferred to Follow-Up Work**
- Launch hygiene tier-2 recovery as hard gate on long train (#204 implementation)
- Planet Flow track relocation (origin R30)

**Outside this product's identity** (from origin)
- Replacing Kaggle Docker
- Planet Flow as default competition path

---

## Open Questions

**Resolved in this plan (assumptions)**
- Preflight length: inherit `docs/benchmarks/preflight-calibration.json` Gates 2–3 window unless dogfood shows need for shorter recipe.
- Hydra profile name: `ssot_pipeline`.
- Registry staleness: R5 invalidation dimensions; full TTL policy deferred.

**Deferred to implementation**
- Exact tournament qualifier tick interval vs checkpoint save frequency
- Whether submission opponent legs reuse unified ladder executor or slim Docker matrix

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Shipping teardown before SSOT profile works | U8 last; smoke test in U4 before doc flip |
| Tournament qualifiers diverge from Docker submission | R23–R24 parity tests; submission opponent legs at trained weights (R21) |
| Long train wall clock (#204) | Document before enforcing R12 in production; not blocking U1–U5 merge |
| Plan 005 code assumes Docker qualifiers | Reuse bracket state, replace qualifier *trigger* with U5 |
| Wrong Hydra profile during short preflight | U4 must use SSOT profile + `preflight-calibration.json`; see `docs/solutions/integration-issues/planet-flow-preflight-calibration-profile.md` |

---

## Sources & Research

- Origin: `docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`
- Superseded qualifier flow: `docs/plans/2026-06-03-005-feat-kaggle-bracket-ranking-plan.md`
- Existing packaging: `src/cli/eval.py`, `src/artifacts/kaggle_submission.py`
- Seed scheduler: `src/training/seed_scheduler.py`, `src/config/schema.py`
- Training loop hooks: `src/jax/train/loop.py`, `src/jax/train/bracket_training.py`
- Preflight calibration: `docs/benchmarks/preflight-calibration.json`
- Institutional learnings: `docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md` (canonical spine), `docs/solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md` (legacy), `docs/solutions/architecture-patterns/jax-comet-kaggle-parity-ci-gate.md`, `docs/solutions/architecture-patterns/kaggle-bracket-ranking-foundational-slice.md` (legacy), `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`
