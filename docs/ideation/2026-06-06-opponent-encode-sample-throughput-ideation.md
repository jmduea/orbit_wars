---
date: 2026-06-06
topic: opponent-encode-sample-throughput
focus: Fix ~70% rollout time in opponent encode/sample for production mixed opponent setups (noop not the problem)
mode: repo-grounded
related:
  - docs/brainstorms/2026-06-06-opponent-rollout-ce-optimize-requirements.md
  - docs/plans/2026-06-06-005-feat-family-batched-mixed-sampling-plan.md
---

# Ideation: Opponent Encode/Sample Throughput

How to fix rollout where **~68–70% of collect time** sits in the opponent phase (encode + sample) under **production mixed opponent** curricula — while **noop-only paths are already fast** and must stay so.

---

## Grounding Context

### Codebase context

Orbit Wars is a **Python 3.12 / JAX / Hydra PPO** project. Production hot path:

```
obs → encode_turn (TurnBatch) → policy K-step decoder + shield → env_step
```

| Signal | Source | Implication |
|--------|--------|-------------|
| Opponent ~68%, policy ~17%, env ~10% | Offline `ow benchmark rollout-phase-profile` (quick `task=map_pool`) | Opponent bucket is the lever — not learner hygiene (already exhausted) |
| 2p `opp_batch_cache` skip | `collect.py` — only when `mode.opponent=noop` literally | `opponents=noop_only` still pays full encode (`mode.opponent=self`) |
| 4p encode | `collect.py` vmmaps `encode_turn` across **all 4 perspectives every step** | Structural 4× multiplier on mixed 2p/4p training |
| Scripted families | `builders.py` heuristics wired through `TurnBatch` for edge layout | Do not semantically need full planet-edge ML features |
| Mixed dispatch today | `sampling.py` — **family loop + masked reorder already landed** on main | Did **not** materially reduce opponent fraction (~70% persists) |
| ce-optimize experiments (integration) | `.context/.../opponent-rollout-throughput/experiment-log.yaml` | 3 reverted hypotheses; `production_mix` worst rung is **100% latest/historical** (neural encode dominates) |
| Measurement loop | `docs/brainstorms/2026-06-06-opponent-rollout-ce-optimize-requirements.md` | Ladder noop→scripted_heavy→self_play→production_mix; score at **worst_rung** |

**Constraint:** User pain is **production_mix / mixed curriculum**, not noop admission recipe.

### Past learnings

- Profile before micro-optimizing (`production-training-throughput-profiling.md`)
- Offline phase profile only — never `telemetry=rollout_phase_timing` on production train (host sync stall)
- Launch hygiene / factorized sampler targets **policy** phase, not opponent
- Prior encode experiment: edge-only scripted path helped `scripted_heavy` but **regressed `production_mix`** — need worst-rung gates

### External context

- JAX `lax.switch` under `vmap` evaluates all branches → **partition-then-dense-batch** (already started in mixed 2p reorder)
- AlphaStar / SEED: batched heterogeneous inference at service layer
- StarCraft heuristics: **filtered features**, not full game tensors
- Compiler DCE analogy: full `encode_turn` for scripted families is structurally dead work

---

## Topic Axes

1. **encode_turn cost reduction** — skip, lite paths, cache, 4p dedup
2. **mixed-family sampling dispatch** — family-batched paths vs per-env vmap
3. **neural opponent inference** — batched latest/historical forwards
4. **rollout scheduling / curriculum** — when families mix within a scan window
5. **measurement & admission alignment** — sub-meters, worst-rung proof

---

## Ranked Ideas

### 1. Per-step family-aware encode after sampling (not post-step full vmap)

**Description:** Move opponent encode from unconditional post-step refresh into the opponent phase **after** family is known: neural (`latest`/`historical`) rows get full `encode_turn`; scripted rows get edge-only lite path; noop rows skip. Avoids paying full encode for all envs before sampling and avoids stale-cache skip mistakes (ce-optimize batch 3 proved post-step skip unsafe for scripted).

**Axis:** encode_turn cost reduction · mixed-family sampling dispatch

**Basis:** `direct:` ce-optimize experiment-log iteration 3 — `production_mix` is 100% neural so lite encode alone regressed +2.3pp; learnings recommend "per-step family-aware encode after sampling (not post-step full vmap)". `collect.py` still refreshes opponent batch before `_sample_opponent_*`.

**Rationale:** At `production_mix` worst rung, **encode** (not dispatch batching) is the floor — family-batched dispatch already shipped without moving the needle. Fusion targets the remaining structural waste: encode work that runs before the branch is chosen.

**Downsides:** Touches scan carry; must preserve learner `feature_history` vs `opp_batch_cache` separation.

**Confidence:** 82%

**Complexity:** Medium–High

**Status:** Unexplored (ce-optimize backlog priority: high)

---

### 2. Opponent family capability registry + dual-path encode

**Description:** Central registry (`OPPONENT_FAMILY_*` → `{needs_full_encode, needs_k_step_decoder, needs_shield, edge_fields}`). Scripted families use `encode_scripted_edges(game, cfg)` (edge mask, src/tgt, minimal globals); `latest`/`historical` keep full `encode_turn(..., history=None)`. Rollout, cache refresh, and sampling read one capability table.

**Axis:** encode_turn cost reduction

**Basis:** `direct:` AGENTS.md — only `latest`/`historical` semantically need full `encode_turn`; `builders.py` `_edge_scripted_context` already reads edge subgraph only. `ce-optimize` R16 seeds families that do not read edge features.

**Rationale:** In `production_mix`, a large share of opponent slots are scripted but pay full v2 catalog encode every step — the largest **avoidable encode** cost. Registry makes every future family path a lookup, not a three-file audit.

**Downsides:** Must define golden/validity scope for lite tensors; prior edge-only experiment regressed `production_mix` when neural rungs still needed full path — registry must gate correctly.

**Confidence:** 80%

**Complexity:** High

**Status:** Unexplored

---

### 3. 4p encode three opponents only (learner batch reuse, minimal cond)

**Description:** Reuse learner `turn_batch` for the learner player slot; encode only three opponent perspectives per step. Prefer **static scatter/gather** over per-flat-index `lax.cond` — ce-optimize batch 2 showed carry+cond **regressed** +1.6pp (cond overhead ate skipped encodes).

**Axis:** encode_turn cost reduction

**Basis:** `direct:` ce-optimize iteration 2 reverted; experiment log: "4p encode only 3 opponent slots via static scatter without per-flat-index cond". `collect.py` lines 284–296 still vmmap 4× encode.

**Rationale:** 4p is a structural 4× multiplier on mixed 2p/4p training; prior carry attempt failed due to branch overhead, not because skip is wrong.

**Downsides:** Perspective-specific field audit; immutable `collect_timed.py` may need mirror PR for profiling parity.

**Confidence:** 70%

**Complexity:** High

**Status:** Unexplored

---

### 4. Predicate-gated `opp_batch_cache` refresh (beyond noop)

**Description:** Extend 2p skip rules: refresh opponent batch only when (a) neural family active, (b) edge topology / ownership delta exceeds threshold, or (c) reset/`done`. Scripted-heavy stages skip per-step re-encode when game slice unchanged. Wire predicates from capability registry.

**Axis:** rollout scheduling · encode_turn cost reduction

**Basis:** `direct:` `skip_opp_batch_refresh` gated on literal noop only; `scripted_heavy` ladder opponent fraction ~0.70 — encode dominates even without neural forwards.

**Rationale:** Cache carry without smart refresh still re-pays `encode_turn` every step for mixed production. Extends proven Phase B pattern without new algorithms.

**Downsides:** Invalidation matrix (comets, fleet arrival, planet capture) must be exhaustive — one miss poisons actions.

**Confidence:** 72%

**Complexity:** Medium

**Status:** Unexplored

---

### 5. Opponent sub-phase meters (encode / sample / shield)

**Description:** Extend offline `rollout-phase-profile` to emit `opponent_encode`, `opponent_sample`, `opponent_shield` fractions inside the opponent bucket; store in ladder baseline and ce-optimize experiment log. Primary gate stays total opponent fraction; diagnostics become actionable.

**Axis:** measurement & admission alignment

**Basis:** `direct:` `offline-rollout-phase-profile-decoupled-from-jit-collect.md` merges encode+sample today; ce-optimize SC2 requires ≥10% improvement but R10 isolates mutable scope — without sub-meters, wins cannot prove which lever to compound.

**Rationale:** At ~70% total, a 10% win could be encode-only or sample-only. Sub-meters prevent repeating the `scripted_heavy` win / `production_mix` regression mistake.

**Downsides:** Small overhead in `collect_timed.py`; not a throughput fix by itself.

**Confidence:** 88%

**Complexity:** Low–Medium

**Status:** Unexplored

---

### 6. Episode-bound family homogeneity (freeze family per episode)

**Description:** At reset/reseed, assign each env one opponent family for the full episode (curriculum-weighted). Mixed **distribution** preserved across batch; per-step family entropy drops — enables single-family fast paths and true batched neural inference within scan windows.

**Axis:** rollout scheduling · mixed-family sampling dispatch

**Basis:** `reasoned:` `effective_type_ids` static within episode; per-step mixture forces worst-case dispatch every step. Curriculum controller already owns stage weights.

**Rationale:** Operators need mixed opponents for learning signal, not per-step family roulette that blocks batching. Trades short-horizon diversity for throughput.

**Downsides:** Curriculum bias risk — family histograms over 1k updates must match baseline; may need manifest documentation if within-window diversity is required.

**Confidence:** 65%

**Complexity:** Medium

**Status:** Unexplored

---

## Recommended sequencing

1. **#5** sub-phase meters — split encode vs sample vs shield inside opponent bucket
2. **#1** per-step family-aware encode (ce-optimize top backlog item)
3. **#3** 4p three-slot encode without cond overhead
4. **#2** registry + dual-path encode — wins on `scripted_heavy` / mixed stages; **insufficient alone** for `production_mix` (100% neural)
5. **#4** predicate cache — only with correct invalidation; **not** blanket skip for scripted (batch 3)
6. **#6** episode homogeneity — fallback if encode fusion insufficient

**Deprioritized:** family-batched mixed dispatch — **already on main** (`sampling.py` family loop + masked reorder); operator confirmed no material opponent-fraction improvement.

Verify every merge at **`production_mix` worst_rung** via `ow benchmark rollout-phase-profile --full-geometry` and ce-optimize keep rules — not noop-only admission.

---

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | 10× GPU opponent sidecar | Too expensive; pipelining speculative before algorithmic fixes |
| 2 | Zero-neural / all-scripted ablation | Diagnostic rung, not a production fix |
| 3 | All-neural collapse (remove scripted) | Increases inference cost; wrong learning recipe |
| 4 | Seven pre-compiled collect_fn variants | Compile explosion; duplicates episode homogeneity (#7) |
| 5 | Snapshot-frozen TurnBatch (stale encode) | High correctness risk; weak invalidation story |
| 6 | 4p perspective signature dedup cache | Complex equivalence proof; high review burden |
| 7 | Encode-heavy curriculum throttling | Schedules around cost; does not reduce encode/sample |
| 8 | Worst-rung admission JSON only | Already specified in ce-optimize brainstorm — process, not fix |
| 9 | Cross-format 2p+4p neural fusion | Defer until family batching lands |
| 10 | Opponent shield lite (standalone) | Absorbed into #2 registry path as implementation detail |
| 11 | Per-env oracle extrapolation only | Complements #6 but insufficient alone for merge proof |
| 12 | Duplicate family-batch / lite-encode variants | Merged into survivors #1–#2 |
| 13 | Family-batched mixed dispatch (as new work) | **Already shipped** on main; opponent fraction ~70% unchanged per operator + ce-optimize baseline |
