---
date: 2026-06-05
topic: orbit-wars-continuation-directions
focus: organize user ideas (rollout/decoder/features/obs/agent/framework) + path forward
mode: repo-grounded
related_ideation:
  - docs/ideation/2026-06-01-agent-action-selection-ideation.md
  - docs/ideation/2026-06-03-searchable-measurable-env-shaping-ideation.md
---

# Ideation: Orbit Wars Continuation Directions

Organizes the user's broad and targeted ideas into a coherent taxonomy, critiques and ranks directions, and recommends a sequenced path forward aligned with [ROADMAP.md](../ROADMAP.md) and measured throughput evidence.

**Prior ideation (last 30 days, related but not resumed):**
- [2026-06-01 Agent Launch Action Selection](2026-06-01-agent-action-selection-ideation.md) — within-turn launch dedup masks (launch hygiene), dedup, masking (shipped; production throughput gate regression followed).
- [2026-06-03 Searchable Env Shaping](2026-06-03-searchable-measurable-env-shaping-ideation.md) — outer-loop MDP search (defer until train path is usable).

---

## Grounding Context

### Codebase context

Orbit Wars is a **Python 3.12 / JAX / Hydra PPO** project targeting Kaggle submit-valid agents. The production hot path is:

```
obs → encode_turn (TurnBatch) → encoder once → multi-launch pointer decoder (K-step)
     → launch trajectory legality filter + within-turn launch dedup masks (launch hygiene) → PPO replay (factored_sequence_scan)
```

| Area | Key paths | State |
|------|-----------|-------|
| Rollout | `src/jax/rollout/collect.py` | Dominates update time post-hygiene (~13.7 s vs ~0.7 s PPO) |
| Decoder | `src/jax/policy.py`, `src/jax/action_sampling.py` | Static `max_moves_k` scan; pays decode/shield/hygiene even when `mean_active_launches_per_turn ≈ 0` |
| Features | `src/jax/features.py`, `src/features/catalog/` | Hand-crafted planet-edge v2 encoding; golden tests in `tests/test_feature_encoding_golden.py` |
| Throughput gates | `docs/benchmarks/launch-hygiene-e2e-baseline.json`, `ow benchmark training` | Production training throughput gate (tier-2): pre-hygiene vs current main — see baseline + ablation JSON (do not rehash numbers here) |
| Kaggle mechanics parity | `tests/test_jax_env_parity.py`, `make test-kaggle-parity` | CI-enforced correctness substrate (planets, comets, combat, obs replay); orthogonal to throughput |
| Alt agent path | `ComposablePlanetFlowPolicy` | Demand-heatmap action compiler (Planet Flow) — potential K-step elimination, separate maturity |
| Verification | Preflight gates, hybrid promotion, unified tournament | Learning signal calibrated; hygiene **wins** learner ablation vs baseline despite throughput loss |

**ROADMAP posture (2026-06-02):** Phase = submit-valid. Production training throughput gate (tier-2) recovery is **Later**, conditional on "new rollout sampling design." Within-turn launch dedup masks (launch hygiene) assessment **Done** — hot-path micro-opts exhausted; learner ablation favors keeping masks.

### Past learnings

- [launch-hygiene-incremental-carry-throughput](../solutions/performance-issues/launch-hygiene-incremental-carry-throughput.md) — O(K²) prefix recompute regressed sampler; fixed with incremental carry; tier-1 green.
- [launch-hygiene-learner-ablation-gate](../solutions/tooling-decisions/launch-hygiene-learner-ablation-gate.md) — When tier-2 fails, **learn-proof** tiebreaker favors hygiene over revert.
- [production-training-throughput-profiling](../solutions/developer-experience/production-training-throughput-profiling.md) — Profile production path before micro-optimizing helpers.
- [launch-hygiene rollout design](../plans/2026-06-01-launch-hygiene-rollout-throughput-design.md) — Failed: all-inactive fast path, compact carry, stop-first; **next safe design: selected-action validation**.

### Nomenclature debt (process theme — survivor-worthy)

Opaque terms (`tier-1/2`, `launch hygiene`, `Planet Flow`) hide two parallel tracks: **Kaggle mechanics parity** (correctness, CI-gated via `make test-kaggle-parity`) vs **production training throughput** (performance, GPU-gated via the production training throughput gate). New docs should use [nomenclature RFC](../nomenclature-rfc.md) terms with one parenthetical alias for legacy symbols. Ideation and plans must cross-link parity when discussing "broken/unusable" — a throughput regression does not imply env wrongness; cherry-picks and rollout redesigns sit on top of the parity substrate in `tests/test_jax_env_parity.py`.

### External context

- **PureJaxRL / CleanRL scan variants** — `jax.lax.scan` compiles rollout loops; `unroll` tuning trades compile vs runtime ([JAX scan docs](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)).
- **AlphaStar-style autoregressive masking** — mask/validate selected units without replacement (cited in 2026-06-01 ideation).
- **Env-in-JAX frameworks (Brax, MJX, TCGJax)** — full-pipeline JAX enables 10–40× throughput when env+policy are pure functions; pattern for eventual framework extraction.
- **Strategy-game embeddings** — discrete entity/edge embedding tables complement hand features (TCG/strategy RL literature).

### User ideas as constraints

All items below are **in scope** for taxonomy and ranking; none were dropped from consideration.

---

## Topic Axes

1. **Throughput & hot-path performance** — rollout collection, multi-launch decoder cost, within-turn launch dedup masks (launch hygiene), factorized sampler microbenchmark (tier-1) / production training throughput gate (tier-2)
2. **Observation & feature representation** — `encode_turn` gaps, embeddings, golden coverage
3. **Agent architecture & action semantics** — factorized decoder vs compiler/set policies, deterministic vs stochastic sampling
4. **Framework generalization** — extracting a reusable JAX RL shell for similar games
5. **Engineering process & knowledge** — lock-in baseline, complexity analysis, abstractions, doc archival, onboarding

---

## Organized Taxonomy (All User Ideas)

### Theme A: Performance — Rollout, Decoder, Throughput Lock-In

| User idea | Placement | Dependencies | Notes |
|-----------|-----------|--------------|-------|
| Rollout optimizations (broad) | A1 | Throughput measurement first | Rollout collect is proven bottleneck, not PPO |
| Decoder optimizations (broad) | A2 | Rollout semantics | Micro-opts on mask representation exhausted |
| Eliminating K-step decoder cost (targeted) | A3 | Agent architecture choice | Static K scan + inactive substeps = core tax; semantic redesign required |
| Lock-in "this works" throughput (targeted) | A4 | Baseline SHA + tier-2 gate | Pre-hygiene `79162a20` is throughput anchor; must reconcile with hygiene learn-proof |
| Nuclear rollback + cherry-pick (targeted) | A5 | A4 | Formalized recovery when main may be "broken/unusable" |
| Replace probabilistic with deterministic (targeted) | A6 | A2/A3 | Scoped to rollout collection or argmax diagnostics—not whole training |

**Relationships:** A1→A2→A3 is the causal chain from profiling. A4/A5 are process wrappers ensuring A1–A3 changes are validated. A6 is a tactic inside A2, not a global policy flip.

### Theme B: Representation — Features, Obs, Agent Fundamentals

| User idea | Placement | Dependencies | Notes |
|-----------|-----------|--------------|-------|
| Feature optimizations (broad) | B1 | Obs audit | Hand-crafted v2 catalog; dim 6×N+7 |
| obs as embeddings (broad) | B2 | B3 | Fusion layer atop catalog, not blind replacement |
| Identifying obs representation gaps (targeted) | B3 | Golden tests | Replay showed 72% duplicate-launch turns — symptom of action interface + possibly obs |
| General agent approach (broad) | B4 | A3 or demand-heatmap compiler | "How agent works fundamentally" — pointer AR vs compiler/set policy |

**Relationships:** B3 should precede B1/B2 investment. B4 converges with throughput theme when K-step AR is the fundamental choice under review.

### Theme C: Platform — Framework, Abstractions, Docs, Onboarding

| User idea | Placement | Dependencies | Notes |
|-----------|-----------|--------------|-------|
| Generalized RL framework (broad) | C1 | A4 throughput lock-in | Extract Hydra+JAX+PPO+benchmark shell; Orbit Wars = game plugin |
| Better documented abstractions (targeted) | C2 | — | `docs/architecture/jax-policy-encoder.md` is template |
| Rigorous time-complexity analysis (targeted) | C2 | A1 | O(K²) regression proves need; tie to `ow benchmark training` |
| Doc archival policy (targeted) | C3 | — | `brain_dump.md` retired; ideation sprawl |
| Contributor onboarding (targeted, low priority) | C4 | C2/C3 | Explicitly deprioritized by user |

**Relationships:** C1 is **blocked by** Theme A until the production training throughput gate (tier-2) is green. C2/C3 parallelize with throughput recovery. C4 is last.

---

## Ranked Ideas

### 1. Nuclear Rollback + Cherry-Pick Manifest

**Description:** Pin a `throughput-baseline` branch at SHA `79162a2088160b8ed05c3e3a050e064c7f6c9556` (documented in [launch-hygiene-e2e-baseline.json](../benchmarks/launch-hygiene-e2e-baseline.json)). Run the production training throughput gate (tier-2) and the learning proof ladder on baseline. Bisect `main` commits cherry-picking only changes that pass **both** the throughput gate and learning proof (within-turn launch dedup masks, incremental carry, split decoder contracts, etc.). Record outcomes in a machine-readable cherry-pick manifest (extend [launch-hygiene-ablation.json](../benchmarks/launch-hygiene-ablation.json) pattern). **Env-parity constraint:** every cherry-pick candidate must keep `make test-kaggle-parity` green — Kaggle mechanics parity is non-negotiable substrate (`tests/test_jax_env_parity.py`, comet/planet generation in `src/jax/env.py`); throughput recovery is about rollout/decoder stack on verified game rules, not rewriting env physics. See [comet subsystem plan](../plans/2026-06-03-008-feat-jax-comet-subsystem-plan.md), [CI parity plan](../plans/2026-06-03-009-feat-ci-kaggle-jax-parity-plan.md), and `AGENTS.md` Kaggle env parity facts.

**Axis:** Engineering process & knowledge (supports Throughput theme)

**Basis:** `direct:` user nuclear option; throughput regression in baseline/ablation JSON artifacts; baseline SHA committed; `direct:` ablation doc — within-turn launch dedup masks win learning vs revert; parity CI gate

**Rationale:** User suspects current implementation may be unusable for **training volume**; incremental hot-path opts are exhausted per ROADMAP. Cherry-pick gives a disciplined alternative to blind revert (loses dedup masks) or blind forward (loses throughput) while preserving Kaggle mechanics parity.

**Downsides:** Merge conflict cost; risk of missing interacting commits; baseline lacks dedup-mask learnability wins.

**Confidence:** 88%

**Complexity:** Medium

**Status:** Unexplored

---

### 2. Selected-Action Validation Rollout Redesign

**Description:** Replace mask-before-sample full legality lattice in rollout with: (1) sample candidate launch from cheaper policy masks; (2) validate only the selected `(source, target_slot, bucket)` against trajectory shield + launch hygiene; (3) invalid → no-op/stop with PPO-consistent stored log-probs. Mirror semantics in `factored_sequence_scan.py` replay. This is the documented next safe design in [launch-hygiene rollout design](../plans/2026-06-01-launch-hygiene-rollout-throughput-design.md).

**Axis:** Throughput & hot-path performance

**Basis:** `direct:` rollout design doc §Current Decision; profiling — 13.68 s rollout vs 0.70 s PPO; failed mask-representation experiments

**Rationale:** Directly attacks user "eliminating K step decoder cost" at the semantic level — majority turns pay lattice for zero launches. Addresses rollout optimizations (broad) without removing hygiene.

**Downsides:** Log-prob / IAM consistency risk; must not weaken launch-hygiene teaching signal; significant correctness test burden.

**Confidence:** 85%

**Complexity:** High

**Status:** Unexplored → candidate for `ce-brainstorm`

---

### 3. Throughput Lock-In Branch with Production Throughput Gate

**Description:** Establish a protected integration path where **no merge to `main`** (or to `throughput-baseline` successor) occurs without `make test-launch-hygiene-e2e-throughput` (production training throughput gate, tier-2) passing vs `docs/benchmarks/launch-hygiene-e2e-baseline.json` (±10%). Pair with the learning proof ladder on the same preset (`primary`, `task=shield_cheap`, `model=transformer_factorized`). Document the dual gate in [operator-runbook.md](../operator-runbook.md).

**Axis:** Throughput & hot-path performance

**Basis:** `direct:` user "lock-in this works"; AGENTS.md launch-hygiene tier-2 procedure; ROADMAP Later item

**Rationale:** Converts implicit frustration into an enforceable definition of "works" — user requirement made operational.

**Downsides:** CI GPU host dependency; may block merges until #2 lands; baseline predates hygiene.

**Confidence:** 90%

**Complexity:** Low–Medium

**Status:** Unexplored

---

### 4. Policy No-Op / Launch Gate Before Legality Lattice

**Description:** Add an explicit sampled gate (launch vs no-op) **before** source/target/bucket mask construction in `_sample_shielded_factored_sequence_with_params`. When gate selects no-op, skip decode/shield/hygiene sub-steps for that env row (static scan carries zeros). Complements #2 for turns where policy intends no launches.

**Axis:** Agent architecture & action semantics

**Basis:** `direct:` rollout design — stop-first neutral because no-op flows through bucket path; `mean_active_launches_per_turn: 0.0` with expensive mask path

**Rationale:** Implements "general agent approach" refinement — separate **intent to act** from **action construction**; reduces K-step tax for idle turns.

**Downsides:** Changes action factorization; must align with Kaggle submission act path; may need curriculum to avoid collapse to always-no-op.

**Confidence:** 78%

**Complexity:** High

**Status:** Unexplored

---

### 5. Obs Representation Gap Audit + Golden Test Expansion

**Description:** Systematic audit of `encode_turn` / edge catalog vs degenerate behaviors documented in agent-action-selection ideation (duplicate launches, friendly reverse relays, dribbling). Deliver: gap matrix (missing signal → symptom → proposed feature), new golden vectors in `tests/test_feature_encoding_golden.py`, and replay metrics on baseline vs main checkpoints.

**Axis:** Observation & feature representation

**Basis:** `direct:` user "identifying gaps in obs representation"; `direct:` 72% duplicate-turn replay stats; golden test infrastructure exists

**Rationale:** Feature optimizations and obs-as-embeddings (user broad ideas) need evidence-backed targets; prevents representation work from guessing.

**Downsides:** Audit latency; may show gaps are action-interface not obs; embeddings still Phase 2.

**Confidence:** 82%

**Complexity:** Medium

**Status:** Unexplored

---

### 6. Time-Complexity Ledger for JAX Hot Paths

**Description:** Create `docs/benchmarks/hot-path-complexity.json` documenting asymptotic cost and measured `rollout_collect_seconds` / sampler ms for: `action_sampling.py`, `factored_sequence_scan.py`, `encode_turn`, `launch_hygiene.py`. Require PRs touching these files to update ledger + run tier-1 gate. Link from module docstrings.

**Axis:** Engineering process & knowledge

**Basis:** `direct:` user rigorous time-complexity analysis; `direct:` O(K²) regression postmortem; `ow benchmark training --detailed-timing` exists

**Rationale:** Prevents repeat of "sampler passes, training fails"; compounds every future rollout/decoder optimization.

**Downsides:** Maintenance overhead; asymptotics alone miss XLA constant factors.

**Confidence:** 80%

**Complexity:** Low

**Status:** Unexplored

---

### 7. Defer Framework Generalization Until Throughput Restored (Sequencing)

**Description:** Explicitly sequence the user's "generalized framework for similar RL problems" **after** survivor #3 passes. Preparation only: maintain `docs/architecture/*` one-pagers (survivor #6's doc twin) identifying extractable boundaries (`orbit-rl-core` vs `orbit-wars-game`). Do not split packages or rename modules until train path is submit-valid practical again.

**Axis:** Framework generalization

**Basis:** `direct:` user framework idea; `reasoned:` extracting while hot path is 4× slow produces abstracted broken reference; Brax/MJX pattern requires working end-to-end loop first

**Rationale:** Honest path forward — preserves user intent without scope overrun that starves throughput recovery.

**Downsides:** Framework ambition delayed; risk of never returning if throughput work stalls.

**Confidence:** 75%

**Complexity:** Low (decision) / High (later execution)

**Status:** Unexplored

---

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Remove launch hygiene from training rollout | `direct:` learner ablation — hygiene wins learn-proof; contradicts shipped learning evidence |
| 2 | Dynamic all-inactive fast path inside K-scan | `direct:` failed experiment — 15.3 s rollout vs 13.7 s baseline |
| 3 | Compact ForbiddenCarry in rollout only | `direct:` failed experiment — slower than baseline |
| 4 | Ship training at current ~2.4K env_steps/sec | Fails meeting-test for submission-scale compute; user requires throughput lock-in |
| 5 | Framework extraction / package split now | Scope overrun; sequencing survivor #7 defers explicitly |
| 6 | obs-as-embeddings wholesale replacement | Too vague without audit (#5); downgrade to Phase 2b after gaps identified |
| 7 | Planet Flow as immediate K-step replacement | Strong direction but overlaps #2/#4; better as brainstorm fork after throughput brainstorm |
| 8 | 100% deterministic training globally | Entropy/learn-proof risk; scope to rollout-only diagnostic per user deterministic preference |
| 9 | K=1 always (max_moves_k=1) | Untested learn-proof impact; variant of decoder redesign not independent strategic bet |
| 10 | Contributor onboarding overhaul | User explicitly lower priority |
| 11 | Env shaping catalog (2026-06-03 #1) | Already ideated; defer until train path usable — duplicate survivor |
| 12 | Delete deprecated docs | User asked archival policy; deletion loses bisect/debug context |
| 13 | Accept permanent tier-2 failure | Contradicts user lock-in + ROADMAP conditional recovery |
| 14 | Incremental mask micro-optimizations | `direct:` ROADMAP — hot path exhausted |

---

## Recommended Path Forward

### Phase 0 — Stabilize definition of "works" (1–2 days)

1. Create `throughput-baseline` branch at `79162a20`; run production training throughput gate (tier-2) + learning proof ladder; record JSON artifacts.
2. Stand up survivor **#3** dual gate documentation in operator runbook.
3. Start survivor **#6** complexity ledger seeded from baseline JSON artifacts (no duplicate profiling prose).

**Decision point:** If baseline learn-proof fails modern gates, adjust baseline anchor to post-hygiene main with **relaxed tier-2 only** — ablation doc already legitimizes this tiebreaker.

### Phase 1 — Throughput recovery track (critical path, 1–3 weeks)

Parallel workstreams:

| Stream | Survivor | Blocks |
|--------|----------|--------|
| A: Semantic rollout redesign | #2 Selected-action validation | Production throughput gate green |
| B: No-op gate (can prototype with A) | #4 Policy launch gate | Production throughput gate green |
| C: Nuclear cherry-pick insurance | #1 Cherry-pick manifest | Safer integration while A/B iterate |

**Verification order** (from rollout design doc):

1. Launch hygiene + factorized sampling correctness tests
2. `uv run ow benchmark training --preset primary --detailed-timing`
3. `make test-launch-hygiene-throughput` (factorized sampler microbenchmark, tier-1)
4. `make test-launch-hygiene-e2e-throughput` (production training throughput gate, tier-2)
5. `make preflight-learn-proof` (learning tiebreaker)

**Decision point:** If #2+#4 fail to reach the throughput gate band, execute #1 cherry-pick onto baseline and port validation design there (keeping `make test-kaggle-parity` green throughout).

### Phase 2 — Representation track (after production throughput gate green OR in parallel on baseline branch)

1. Survivor **#5** obs gap audit + golden expansion.
2. Only then: phased obs-as-embeddings fusion (user broad idea) targeting confirmed gaps.
3. Revisit **demand-heatmap action compiler** (Planet Flow) as `ce-brainstorm` competing with optimized multi-launch pointer path.

### Phase 3 — Platform & knowledge (after Phase 1 gate green)

1. Doc archival policy: `docs/archive/` + `status: deprecated` frontmatter + link lint (user targeted).
2. Abstraction one-pagers for remaining hot modules (user abstractions).
3. Framework extraction brainstorm (user broad idea) — package boundaries, Hydra groups, `ow` CLI factoring.

### Phase 4 — Deferred

- Contributor onboarding (user low priority)
- Env shaping catalog outer loop ([2026-06-03 ideation](2026-06-03-searchable-measurable-env-shaping-ideation.md))

### Alignment with ROADMAP.md

| ROADMAP item | This plan |
|--------------|-----------|
| Later: production throughput gate recovery if new rollout design | **Phase 1** implements that design (#2, #4) |
| Done: within-turn launch dedup ablation | Informs dual-gate tiebreaker (#3) |
| Submit-valid phase | Throughput lock-in is prerequisite for practical submit-valid training volume |

### Related docs

- [ROADMAP.md](../ROADMAP.md)
- [launch-hygiene rollout throughput design](../plans/2026-06-01-launch-hygiene-rollout-throughput-design.md)
- [launch-hygiene e2e baseline](../benchmarks/launch-hygiene-e2e-baseline.json)
- [launch-hygiene ablation](../benchmarks/launch-hygiene-ablation.json)
- [feature-encoding-v2](../feature-encoding-v2.md)
- [jax-policy-encoder architecture](../architecture/jax-policy-encoder.md)
- [operator-runbook](../operator-runbook.md)
- [nomenclature RFC](../nomenclature-rfc.md) — alias table, parity vs throughput tracks
- [JAX comet subsystem plan](../plans/2026-06-03-008-feat-jax-comet-subsystem-plan.md)
- [CI Kaggle/JAX parity plan](../plans/2026-06-03-009-feat-ci-kaggle-jax-parity-plan.md)
- [AGENTS.md](../../AGENTS.md) — calibrated learning gates, async checkpoint eval promotion, throughput gate commands

---

## Scratch artifacts

- Raw candidates: `/tmp/compound-engineering/ce-ideate/2026-06-05-orbit-wars-151648/raw-candidates.md`
- Survivors checkpoint: `/tmp/compound-engineering/ce-ideate/2026-06-05-orbit-wars-151648/survivors.md`
