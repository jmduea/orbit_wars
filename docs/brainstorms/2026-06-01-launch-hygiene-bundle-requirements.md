---
date: 2026-06-01
topic: launch-hygiene-bundle
origin: docs/ideation/2026-06-01-agent-action-selection-ideation.md#1-launch-hygiene-bundle
---

# Requirements: Launch Hygiene Bundle

## Summary

Within a single game turn, the factorized launch decoder must stop wasting K-step slots on redundant or ping-pong launches. Each turn applies three coordinated rules: no re-picking the same source→target edge, no same-turn reverse friendly relay after a forward friendly relay, and a final merge of identical decoded launches before the environment executes them. Masking applies consistently during action sampling and policy-gradient replay so training learns the constrained behavior.

## Problem Frame

The current K-step launch scan decrements remaining ships per pick but carries no memory of prior picks beyond ship counts. Replay analysis on a u5000 checkpoint showed roughly 72% of acting turns contained duplicate-identical launch tuples, ~157 turns with same-turn reverse-angle friendly pairs, and mean 1.0 ships per launch under continuous fraction mode — slot-filling and dribbling rather than consolidated attacks.

These patterns are objectively wasteful in most cases: three identical launches to the same target in one step are worse than one combined launch, and same-turn friendly A→B then B→A shuttles ships that cannot arrive in time to matter. The policy has no structural reason to stop because every slot remains legally sampleable.

## Key Decisions

- **Mask at sample and replay, not env-only.** Outcome-only dedup in the environment would fix replays but not train the policy out of spam. All hygiene rules must constrain what the policy can sample and what log-probabilities PPO recomputes.

- **Three mechanisms, one bundle.** Cumulative edge mask, friendly reverse ban, and builder merge ship together. Each covers gaps the others miss; builder merge is a safety net, not a substitute for training-time masks.

- **Mask duplicate edges; do not auto-merge in-scan (v1).** When the policy repeats the same (source, target slot), later steps mask that edge rather than silently accumulating ship counts into the first pick. In-scan merge is deferred — it changes credit assignment and is a separate decision.

- **Narrow ping-pong definition.** Reverse ban applies only to friendly→friendly edges where the target planet of an earlier pick becomes the source pointing back at the earlier source. Staging A→B then B→C toward an enemy remains legal.

- **Telemetry paired but not blocking v1.** Launch hygiene metrics (duplicate rate, ping-pong rate) should land in the same effort when cheap, but behavioral requirements do not depend on metrics existing first.

## Requirements

### Within-turn deduplication

R1. After an active launch selects source *s* and target slot *t* in step *k* of a turn, steps *k+1 … K* must treat edge (*s*, *t*) as illegal for sampling and for log-probability replay.

R2. Dedup key is (owned source row index, target slot index) in the factorized decoder — the same pair the shield and bucket mask already use — not merely identical decoded planet/angle/ship tuples after the fact. Mask dedup (R1–R2) is per (source row, slot); builder merge (R7) collapses identical planet/angle/ship tuples and may still apply when slot keys differ or a path bypasses the scan.

R3. When a masked edge would have been the only legal non-stop choice, the step must fall through to stop/no-op behavior consistent with existing stop-head semantics (no invalid samples, no silent renormalization bugs).

### Anti-ping-pong (friendly reverse ban)

R4. If step *k* selects a launch from friendly source *i* to friendly target *n* (both owned by the learner at turn start), then for steps *k+1 … K* the edge from *n* back to *i* must be illegal when *n* is used as source and *i* appears in its target slot list.

R5. The reverse ban applies only to friendly→friendly pairs. Launches involving neutral or enemy targets are unaffected.

R6. Multi-hop staging in one turn that does not reverse (e.g., A→B then B→C where C is not A) remains legal.

### Builder safety net

R7. Before fleets are submitted to the environment for a turn, identical decoded launch tuples `[source_planet_id, angle, ships]` within that turn's action list must be merged into a single launch with summed ship counts.

R8. Builder merge must not change the training-time masks or log-probs; it is an execution-layer backstop for submission, eval, and any path that bypasses the scan.

### Training and eval parity

R9. Rollout sampling, PPO replay, and submission/eval inference must apply the same cumulative mask state machine so stored sequences remain self-consistent under importance sampling.

R10. Hygiene rules apply to learner rollout, PPO replay, submission/eval inference, and neural opponent paths that use the factorized K-step decoder (`OPPONENT_LATEST`, `OPPONENT_HISTORICAL`, self-play copies). Heuristic single-step edge-batch opponents (random, turtle, sniper, etc.) are out of scope for v1 — document this exclusion in tests.

### Verification

R11. Automated tests must cover: duplicate edge masked on step 2+; reverse friendly edge masked after forward friendly pick; builder merge of identical tuples; replay log-prob matching sampled sequence under masks (R9); stop/no-op fall-through when hygiene masks leave no legal non-stop launch (R3); and opponent factorized paths documented per R10 boundary.

R12. When a replay-parser regression fixture exists, it must demonstrate materially reduced duplicate-launch and ping-pong rates on the fixed checkpoint fixture, per Success Criteria and project calibration policy (not invented round numbers). Fixture creation is not blocking v1 ship — R11 behavioral tests are the v1 gate.

## Key Flows

F1. **Normal multi-target turn**
- **Trigger:** Learner owns planets A and B; enemies C and D are legal targets; K≥2.
- **Steps:** Step 1 launches A→C. Step 2 may launch B→D or A→D if ships remain, but not A→C again. Stop when stop head fires or no legal launches remain.
- **Outcome:** At most one launch per (source, slot) per turn; multiple distinct attacks still allowed.

F2. **Ping-pong blocked**
- **Trigger:** Step 1 selects friendly A→friendly B.
- **Steps:** Step 2 cannot select B→A. Step 2 may still select B→enemy E or A→enemy E.
- **Outcome:** No same-turn reverse shuttle between two friendly planets.

F3. **Builder merge backstop**
- **Trigger:** Decoder or legacy path emits two identical `[planet, angle, ships]` entries in one turn's action list.
- **Steps:** Builder collapses to one entry with combined ships before env launch phase.
- **Outcome:** Environment sees one fleet launch; replay HTML shows merged action if applicable.

F4. **Hygiene forces stop**
- **Trigger:** Step 1 consumes the only high-value edge; step 2 would repeat the same (source, slot) but no other non-stop launch remains legal after hygiene masks.
- **Steps:** Step 2 masks the duplicate edge; sampling resolves to stop/no-op per R3 (no invalid sample, no silent renormalization bug).
- **Outcome:** Turn ends early instead of emitting a duplicate or illegal pick.

## Acceptance Examples

AE1. **Covers R1, R3**
- **Given:** Step 1 active launch A→slot 2 (enemy target). Step 2 begins with ships still on A and slot 2 still shield-legal.
- **When:** Policy samples step 2.
- **Then:** Edge A→slot 2 is masked; sample is either a different edge or stop.

AE2. **Covers R4, R6**
- **Given:** Step 1 friendly A→friendly B. Step 2 B has ships; B→A slot exists and is friendly→friendly.
- **When:** Policy samples step 2.
- **Then:** B→A is masked; B→enemy C remains legal if present.

AE3. **Covers R7**
- **Given:** Action list `[ [A, θ, 3], [A, θ, 3] ]` for one turn.
- **When:** Builder runs.
- **Then:** Env receives `[ [A, θ, 6] ]` (single combined launch).

AE4. **Covers R9**
- **Given:** A sampled sequence stored in rollout buffer with hygiene masks applied during collection.
- **When:** PPO recomputes log-prob for that sequence.
- **Then:** Recomputed log-prob matches within numerical tolerance; no masked edge receives non-zero probability mass.

## Success Criteria

**P0 (blocking v1):** R11 behavioral tests pass; rollout↔replay log-prob parity holds with hygiene enabled (R9); PPO training smoke shows finite loss and no mask/replay mismatch crashes.

**P1 (verify after ship, not blocking):** On the u5000 replay fixture when available, duplicate-identical launch turns and same-turn friendly reverse pairs drop materially vs pre-change baseline (~72% / ~157-turn reference), with thresholds set from measured calibration — not invented round numbers.

**Non-goals for v1 success:** Mean ships per launch increase and tournament win-rate are not gating criteria for this bundle (dedup/sizing floor is deferred ideation #7; tournament proof is a separate eval gate).

## Scope Boundaries

**In scope (v1):**
- Factorized K-step decoder hygiene (train, eval, submission)
- Builder merge backstop on submission, eval, and any legacy or scan-bypass path that can emit duplicate launch tuples (R7–R8)
- Unit and integration test coverage (R11); replay-parser fixture when cheap (R12, non-blocking)

**Priority (v1):** P0 = R1–R6 + R9 + R11; P1 = R7–R8 builder backstop; P2 = R12 fixture / rate regression when metrics work lands.

**Deferred for later:**
- Ownership-tier edge ranking (ideation #2)
- In-scan ship accumulation / auto-merge (ideation #3)
- Multi-hop synthetic edges and reachability features (ideation #4–5)
- Minimum ship fraction floor (ideation #7)
- Promotion gates keyed on hygiene metrics (metrics emission may ship alongside v1 if cheap)

**Outside this bundle's identity:**
- Replacing the K-step decoder with single-action or set-transformer policies
- Removing all friendly→friendly edges from the candidate set
- Env-only dedup without training-time masks

## Dependencies / Assumptions

- Factorized pointer decoder remains the canonical action path (v2-only preference).
- Friendly ownership is determined from planet owner at turn start for mask rules; mid-turn ownership changes during launch phase are out of scope.
- Moving-planet timing nuances (defer single large launch vs split launches) are not solved by this bundle — dedup stops spam, not intercept scheduling optimization.
- Trajectory shield continues to run per step; cumulative hygiene masks compose with shield bucket masks, not replace them.

## Outstanding Questions

**Deferred to planning:**
- Exact representation of cumulative mask state in the scan carry (bitset vs slot list vs hash) while staying JIT-friendly.
- Whether continuous ship fraction mode needs angle/tolerance in builder merge key or slot index is sufficient.
- Opponent builder parity: confirm all jax_actions sampling entrypoints.

**Resolve before planning:**
- **R9 parity (blocking):** Hygiene masks must be derived from the within-turn prefix in both sampling and PPO replay via a shared helper — not only from stored shield bucket masks at collection time.
- Default: hygiene rules always on for factorized decoder; no Hydra toggle in v1 unless testing demands it (planner to confirm).

## Sources / Research

- Ideation seed: `docs/ideation/2026-06-01-agent-action-selection-ideation.md` (idea #1, explored 2026-06-01)
- Prior replay analysis session: u5000 2p/4p docker validation replays (duplicate and ping-pong rates)
- External patterns: AlphaStar autoregressive without-replacement masking; invalid action masking in policy gradient (Huang & Ontañón 2022)
