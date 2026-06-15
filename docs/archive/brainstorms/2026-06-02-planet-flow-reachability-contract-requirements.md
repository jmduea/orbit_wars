---
date: 2026-06-02
topic: planet-flow-reachability-contract
origin: docs/solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md
supersedes_partially: docs/brainstorms/2026-06-01-planet-flow-policy-requirements.md
---

# Requirements: Planet Flow Reachability-Aligned Compiler Contract (P0 v2)

## Summary

Amend Planet Flow P0 so the **learned pressure distribution, PPO credit assignment, and sweep gates** all operate on the same target set the compiler can actually satisfy through the candidate edge catalog. Keep the deterministic per-source argmax compiler for P0 throughput; add reachability masking at sample/logprob time and hard gates on unreachable demand. Defer flow-based compiler rewrites to P1.

---

## Problem Frame

P0 shipped a deliberate split: the policy outputs demand over **all active planets**, while the compiler emits launches only through **top-K candidate edges per owned source**. That split was meant to keep tensors bounded and treat unreachable demand as a diagnostic.

The u150 replay diagnosis (`docs/solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md`) showed the split is not tenable for learning:

- On seed 192 the learner owns one home planet; the enemy home never enters any owned source's catalog for 150 steps.
- The policy still assigns mass to unreachable targets (~65% unreachable demand rate at u150).
- PPO assigns credit/blame on the full heatmap while the compiler fires at the best **legal** neutral — producing replay visuals that look like broken targeting even when angles and inference match training.
- Policy collapse (entropy ≈ 3.8×10⁻⁵, no expansion) compounds the trap but does not explain the structural mismatch.

Orbiting-source angle error was ruled out. The fix is **contract alignment**, not replay plumbing.

---

## Key Decisions

- **Primary direction: Option A + Option D (composite).** Reachability-mask the pressure action at sampling and PPO replay; add preflight/sweep **hard gates** on unreachable demand rate. This aligns credit assignment with compiler feasibility without rewriting the hot-path compiler.

- **Keep P0 compiler mechanism.** Per-owned-source argmax over catalog edges stays for the next proof slice. Changing allocation semantics (Option B) is a P1 variant comparison, not a blocker patch.

- **Catalog bias is optional follow-up (Option C-lite), not P0.** A reserved threat slot for enemy homes in top-K ranking may help early-game attack visibility but does not fix PPO credit on unreachable mass by itself. Evaluate only if A+D still fail expansion/attack coverage gates.

- **Unreachable demand remains a diagnostic, but not a learnable degree of freedom.** After masking, sampled unreachable mass should be ~0 by construction; diagnostics may still report pre-mask intent for debugging.

- **Replay interpretability requirement.** When demand on planet X is unreachable, the learner must not emit launches that visually imply satisfaction of X. Acceptable outcomes: hold, or launch at highest-pressure **reachable** target with diagnostics showing held/unreachable mass. Neutral argmax spill under reachable-only demand is expected P0 compiler behavior, not a replay bug.

- **Strategic scope (partial supersession).** This amend retires the June 1 hybrid-global-heatmap as a **learnable** PPO target universe. June 1 goals still in force: variant family comparison, factorized baseline comparison, compiler-control baselines, throughput floors, and action-quality diagnostics. P0 v2 proves contract alignment; it is **necessary but not sufficient** to declare Planet Flow direction chosen.

- **Escalation budget.** If P0 v2 passes contract gates but June 1 expansion/attack coverage still trails the factorized baseline after a calibrated update window, escalate to Option B (minimal coordinated-allocation slice) before further sweep iteration or P0.5 C-lite alone.

---

## Approach Comparison (brainstorm record)

| Option | Mechanism | Verdict |
|--------|-----------|---------|
| **A. Reachability-masked head** | Mask sample/logprob/entropy to catalog-reachable planets each turn | **Adopt (P0 v2 core)** |
| **B. Compiler rewrite** | Flow / coordinated allocation instead of independent per-source argmax | **Defer to P1 variant** |
| **C. Catalog change** | Threat/distance bias so enemy homes enter top-K | **Optional P0.5 if A+D insufficient** |
| **D. Training-only gates** | Fail sweep/preflight on high `unreachable_demand_rate` | **Adopt alongside A** |

**Why A+D over B now:** B is the largest carrying cost and invalidates current sweep/compiler baselines mid-proof. A+D fixes the demonstrated PPO/compiler divergence with bounded changes and preserves throughput path.

**Why not D alone:** Gates prevent celebrating bad policies but leave the learner optimizing noise on unreachable dimensions.

**Why not C alone:** Catalog changes help specific early-game boards but leave credit assignment misaligned for any unreachable target class (distant neutrals, reinforcement, etc.).

---

## Requirements

**Reachability definition**

- R0. Local R1–R16 in this document supersede identically numbered June 1 requirements where they conflict; inherited parent requirements are always cited as **June 1 R#**.

- R1. A planet is **catalog-reachable** this turn when at least one owned active source has a candidate edge to that planet with positive catalog feasibility (`edge_mask` ∧ owned source). Reachability for sampling/PPO masks is independent of sampled demand pressure (distinct from compiler demand-conditioned diagnostics).

- R2. Reachability must be computed from the same turn batch and ownership state used by the compiler, so rollout, PPO replay, eval, and submission inference agree.

- R3. Reachability is turn-local; a planet unreachable early may become reachable after expansion. The mask updates every step without requiring policy re-init.

**Learned action / PPO contract**

- R4. The policy may retain a fixed-size demand tensor over active planets for encoder convenience, but **sampled pressure actions, log-probability, entropy, and KL** must apply only to catalog-reachable planets. Unreachable planets are masked out of the action distribution at the **logit level before softmax** (same pattern as factorized `_safe_masked_logits`), not only by zeroing post-sample pressure buckets.

- R5. Unreachable planets must not receive gradient credit through the pressure head on the learner path. Post-sample zeroing or summed logprob masking alone does not satisfy R5 if full per-target softmax/KL still routes signal through unreachable heads.

- R6. PPO replay semantics remain defined on the **learned pressure action**, not the compiled launch list (inherits June 1 R11).

**Compiler contract (P0 v2 — mostly unchanged)**

- R7. The compiler continues to convert reachable demand into launches via bounded candidate edges per owned source (inherits June 1 R5–R7, R9 bounded catalog).

- R8. When positive **reachable** demand exists, the compiler emits launches at feasible edges ranked by reachable demand pressure (current per-source argmax behavior for P0 v2).

- R9. When no planet has positive reachable demand after masking, the compiler emits **no launches** (hold). Masking must not renormalize unreachable mass onto reachable planets in a way that creates synthetic reachable demand; unreachable logits are excluded, not redistributed.

**Diagnostics**

- R10. Retain and report: `unreachable_demand_mass/rate`, `held_demand_mass/rate`, emitted launch mass, small-launch rate, and entropy on the **masked** action distribution.

- R12. Diagnostics must distinguish **unreachable demand** (off-catalog target) from **held demand** (reachable pressure not converted to launches). Compiler diagnostics may remain demand-conditioned; sampling reachability must not be.

**Proof gates**

- R13. Planet Flow preflight and W&B sweep guardrails must **fail** when post-mask unreachable demand rate exceeds a calibrated ceiling near zero (derived via `make preflight-calibrate`, not invented). Post-mask rate is primarily a **construction invariant** once R4–R5 ship; preflight promotion also inherits June 1 learn-proof trends, masked entropy floors, and held-demand bounds from the parent spec.

- R14. Sweep score and promotion artifacts must not treat high unreachable demand, launch-collapse, or entropy collapse as success (inherits sweep v2 intent from `planet-flow-sweep-gameable-objective` solution).

- R15. Candidate-coverage audit (June 1 success criteria) must report reachability-conditioned coverage: share of attack/expansion intent that is **catalog-reachable** from current ownership. This extends the existing June 1 audit; it is not a separate evaluation product.

**Compatibility**

- R16. Eval/submission agents must apply the same reachability mask before sampling or argmax pressure at inference (inference parity is P0 v2; checkpoint schema migration is deferred to June 1 R12 / P1).

---

## Key Flows

- F1. **Reachable-only pressure sample**
  - **Trigger:** Learner rollout step with Planet Flow profile.
  - **Actors:** Turn batch encoder, reachability mask builder, policy sampler, compiler.
  - **Steps:** Build catalog-reachability mask from owned sources and edge catalog. Sample pressure only on reachable planets. Compile with existing P0 compiler. Log masked and optional pre-mask diagnostics.
  - **Outcome:** PPO trains on actions the compiler can act on.

- F2. **Early single-home game**
  - **Trigger:** Learner owns one planet; enemy home outside all top-K catalogs.
  - **Actors:** Mask builder, compiler.
  - **Steps:** Enemy home excluded from sample space. Learner may assign reachable demand to neutral/expansion targets in catalog. Unreachable enemy demand cannot be sampled.
  - **Outcome:** No PPO credit on impossible enemy-home mass; replays show neutral launches only when reachable demand warrants them.

- F3. **Post-expansion reachability expansion**
  - **Trigger:** Learner captures a planet that adds enemy home to catalog.
  - **Actors:** Mask builder, policy.
  - **Steps:** Enemy home enters reachable set automatically. Pressure head may now sample demand there without architecture change.
  - **Outcome:** Attack intent becomes learnable when the game state allows it.

- F4. **Gate failure on contract misalignment**
  - **Trigger:** Training run maintains post-mask unreachable rate above the calibrated near-zero ceiling (masking bug or legacy global-heatmap path still active).
  - **Actors:** Preflight, sweep guardrails.
  - **Steps:** Run fails promotion/sweep shortlist. Optional debug telemetry may log pre-mask raw-head unreachable intent for operators but is not a promotion gate unless explicitly calibrated later.
  - **Outcome:** Proof slice cannot pass while PPO and compiler target universes diverge.

---

## Acceptance Examples

- AE1. **Covers R1, R4, R7**
  - **Given:** One owned source whose catalog lists neutrals A/B but not enemy home E; raw logits favor E.
  - **When:** Rollout samples pressure and compiler runs with reachability mask.
  - **Then:** E is not sampled; logprob/entropy exclude E; compiler may launch toward A or B only if reachable sampled demand warrants it.

- AE2. **Covers R2, R16**
  - **Given:** A checkpoint evaluated via submission/tournament agent path.
  - **When:** Inference runs on the same board state as training.
  - **Then:** Reachability mask matches training; no extra unreachable mass enters the action distribution.

- AE3. **Covers R8, R9, R12**
  - **Given:** Reachable demand on neutral N only; all mass on unreachable E masked out.
  - **When:** Compiler argmax runs.
  - **Then:** Launch targets N if pressure exceeds emission threshold; held mass reflects unconverted reachable pressure, not phantom E satisfaction.

- AE4. **Covers R13, R14**
  - **Given:** A run with post-mask unreachable rate above the calibrated near-zero ceiling.
  - **When:** Preflight or sweep guardrails evaluate the run.
  - **Then:** Run fails guardrails; not promoted as sweep winner.

- AE5. **Covers F3, R3**
  - **Given:** Mid-game capture adds edge to enemy home.
  - **When:** Next turn mask is built.
  - **Then:** Enemy home becomes sampleable without checkpoint or architecture migration.

---

## Success Criteria

**P0 v2 proof (after implementation):**

- Post-mask `planet_flow_unreachable_demand_rate` ≈ 0 in steady-state training (by construction once R4–R5 ship).
- Preflight learn-proof trend gates pass on `training=2p4p_16_split` with reachability-masked profile (June 1 parent gates still apply).
- June 1 action-quality gates still apply: held/dropped pressure within calibrated bounds, early expansion/attack coverage not worse than the selected factorized baseline without explicit reason, and compiler-control baseline distinguishes learned pressure from compiler-only behavior.
- Replay spot-checks on previously failing seeds no longer show heatmap-vs-launch contradiction on **unreachable** targets; neutral launches under reachable-only demand remain valid P0 behavior.
- Candidate-coverage audit reports reachability-conditioned attack/expansion/reinforcement feasibility share (R15) and is reviewed before sweep promotion.
- Sweep guardrails reject configs that collapse masked entropy or held-demand quality per existing `planet_flow_sweep_score` / preflight floors (not a new undefined metric in P0 v2).
- Throughput remains within calibrated launch-hygiene e2e bounds on the proof machine (no full compiler rewrite regression).

**P0.5 optional (Option C-lite — only if attack coverage still fails after A+D):**

- Enemy home appears in catalog from relevant owned sources when within configured threat/distance criteria.
- Coverage audit shows improved early-game attack feasibility without breaking factorized decoder catalog invariants.

**P1 (Option B — deferred):**

- Second compiler variant (coordinated flow allocation) compared under same proof deck before replacing argmax compiler.

---

## Scope Boundaries

**In scope:**

- Reachability mask builder shared by rollout, PPO replay, eval, submission.
- Masked sampling/logprob/entropy/KL for Planet Flow pressure head.
- Updated diagnostics and calibrated gates on unreachable demand.
- Documentation superseding the hybrid-global-heatmap PPO contract in the June 1 doc (R9/R10 intent).

**Deferred for planning:**

- Exact mask tensor layout and where reachability lives in `TurnBatch` vs runtime lookups.
- Calibrated numeric thresholds for R13 post-mask ceiling (must come from measurement).
- June 1 R12 checkpoint metadata for reachability-masked vs legacy global-heatmap profiles (P1).
- Option C-lite catalog ranking rules and interaction with `candidate_count`.
- Option B compiler rewrite and multi-hop intent.
- Additional acceptance examples for R5 logit masking, masked entropy reporting, and coverage audit fields.

**Outside this contract's identity:**

- Factorized decoder / launch-hygiene changes unrelated to Planet Flow.
- Reward shaping to penalize unreachable demand (masking preferred over penalty).
- Full all-pairs source-target tensors.
- Tournament win claims as P0 success.

---

## Dependencies / Assumptions

- Candidate edge catalog and `edge_mask` from `encode_turn` remain the feasibility authority (same as factorized path).
- **Current state vs target:** compiler diagnostics and `_target_reachability` exist today, but learner rollout/PPO replay still mask on `planet_mask` (activity-only), not catalog reachability. P0 v2 must replace or augment that mask per R1–R4.
- Static-shape JAX constraints still apply; reachability mask must be bool tensor over fixed planet dimension.
- June 1 Planet Flow requirements remain the parent spec for variants, throughput floors, and proof deck structure; this doc amends the **P0 target-universe / PPO alignment** slice only.
- u150 failure evidence and tests in `tests/test_planet_flow_compiler.py` remain regression fixtures for compiler spillover behavior.

---

## Outstanding Questions

**Resolve before planning:**

- None — recommended direction is A+D with B deferred and C-lite optional.

**Deferred to planning:**

- Whether pre-mask unreachable intent is worth a permanent debug metric vs training-only logging (demoted from normative requirement; optional under `metric_groups.debug`).
- Whether held-demand gate thresholds need tightening when unreachable mass is eliminated from the action space.
- Whether to add an explicit **reachable attack pressure** sweep metric for early-game boards, or rely on June 1 coverage audit + existing entropy/held-demand floors until calibration proves a gap.
- Behavior when `reachable_count=0` for a turn (skip PPO pressure term vs documented hold/no-op entropy expectation).
- Whether pre-mask unreachable intent should become a calibrated promotion gate after P0 v2 lands (currently post-mask only).

---

## Sources / Research

- Parent spec: `docs/brainstorms/2026-06-01-planet-flow-policy-requirements.md`
- Diagnosis: `docs/solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md`
- Compiler: `src/jax/planet_flow.py`
- Edge catalog: `src/features/catalog/edge.py`
- Sweep/guardrails: `src/jax/train/sweep_score.py`, `conf/wandb_sweep/planet_flow_ppo_signal.yaml`
- Regression tests: `tests/test_planet_flow_compiler.py`
