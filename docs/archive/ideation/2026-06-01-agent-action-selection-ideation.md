---
date: 2026-06-01
topic: agent-action-selection
focus: launch dedup, anti-ping-pong masking, edge ordering, multi-hop routing
mode: repo-grounded
---

# Ideation: Agent Launch Action Selection

## Grounding Context

Orbit Wars uses a **factorized K-step launch decoder**: each env turn allows up to `max_moves_k` sequential picks of (owned source → target slot → ship bucket/fraction). The scan in `src/jax/action_sampling.py` updates `remaining_ships` after each slot but **does not** deduplicate identical edges or suppress reverse friendly relays within the same turn.

Edge candidates come from `src/jax/features.py`: per owned source, targets are ranked by sun-blocked → distance (or `intercept_min`) → planet id, keeping top `candidate_count - 1` slots. **Ownership is not used in ranking** — friendly and enemy planets compete only on geometry.

Prior replay analysis (u5000 checkpoint, post\_encoder campaign) showed **\~72% of turns** contained duplicate-identical launch tuples, **\~157 turns** with same-turn reverse-angle friendly pairs, and **mean 1.0 ships per launch** under continuous fraction mode — consistent with slot-filling and dribbling rather than consolidated attacks.

External patterns: AlphaStar autoregressive masking without replacement; invalid-action masking in PPO (sample + loss); GNN routing with k-edge candidates; optional set-based policies when order within turn is irrelevant.

## Topic Axes

1. Within-step launch deduplication & sizing
2. Sequential cross-step action masking (anti ping-pong)
3. Edge candidate generation & target prioritization
4. Multi-hop / indirect routing
5. Turn structure & stop semantics

## Ranked Ideas

### 1. Launch Hygiene Bundle: cumulative edge mask + friendly reverse ban + builder merge

**Description:** Implement three coordinated changes in the K-step scan and action builders: (a) after each active launch, mask the exact `(source, target_slot)` pair from later steps (AlphaStar-style); (b) if step *k* selects `friendly_source_i → friendly_target_n`, mask `friendly_target_n → friendly_source_i` for steps *k+1..K*; (c) as a safety net, merge identical decoded `[planet_id, angle, ships]` tuples in `builders.py` before `_launch_fleets`. Apply masks at **sample time and PPO replay** so log-probs stay consistent.
**Axis:** Within-step dedup + cross-step masking
**Basis:** `direct:` replay stats (72% duplicate turns, 157 reverse pairs); `external:` AlphaStar selected-unit masking + Huang & Ontañón IAM (mask in sample and loss)
**Rationale:** Addresses all three user hypotheses with low conceptual risk; builder merge handles edge cases masks miss; reverse ban matches the stated ping-pong mechanism (same-turn friendly shuttling before fleets arrive).
**Downsides:** Must thread cumulative mask through shield refresh each step; PPO replay in `factored_sequence_scan.py` must mirror masks exactly; builder merge alone doesn't improve training signal without masks.
**Confidence:** 85%
**Complexity:** Medium
**Status:** Explored → `docs/brainstorms/2026-06-01-launch-hygiene-bundle-requirements.md`

### 2. Ownership-tier edge ranking (enemy/neutral before friendly)

**Description:** Add an ownership **tie-breaker** to `lexsort` in `features.py` (after blocked + distance/intercept, before planet id): tier 0 = enemy, tier 1 = neutral, tier 2 = friendly. Distance stays primary — a nearer enemy still outranks a farther friendly. Among targets within a small distance band (or equal intercept\_min), prefer enemy/neutral over friendly so attack slots are not permanently occupied by a static adjacent owned planet. **Avoid** tier-before-distance or a close-friendly-always-wins crossover on static maps — that can make one permanent neighbor dominate top-K forever and starve exploration of other targets.
**Axis:** Edge candidate generation & target prioritization
**Basis:** `direct:` current ranking uses only blocked/distance/id (`features.py` \~242–261); user hypothesis that friendly slots crowd attack opportunities
**Rationale:** Cheap, JIT-friendly, compounds across all sources; shifts inductive bias toward offense without hard-removing reinforcement paths.
**Downsides:** Wrong tier if ownership stale; tie-breaker alone may not surface distant enemies if geometry always ranks a friendly first; needs golden test updates for edge slot ordering.
**Confidence:** 75%
**Complexity:** Low
**Status:** Unexplored

### 3. Auto-merge duplicate picks in-scan (accumulate ships, don't re-sample)

**Description:** Instead of only masking duplicate `(src, slot)` picks, **add launched ships to the first pick's count** when a later step repeats the same edge (within tolerance for continuous fraction). Keeps a single env launch with combined fleet size — directly implements "3 identical launches worse than 1 combined."
**Axis:** Within-step dedup & sizing
**Basis:** `reasoned:` game mechanics reward larger fleets (speed + consolidated arrival); user reasoning on identical launches; complements idea #1
**Rationale:** Stronger than mask-only (which wastes a slot as stop/no-op); teaches optimal sizing even if policy repeats the edge.
**Downsides:** Log-prob attribution for merged slots is subtle; continuous fraction merging needs a defined rule; may hide policy bugs that masks would expose.
**Confidence:** 70%
**Complexity:** Medium–High
**Status:** Unexplored

### 4. Two-hop synthetic edges in top-K (relay without full A\* action space)

**Description:** For each owned source *p₁*, precompute a bounded set of **relay routes** *p₁→p₂→t* where *p₂* is owned friendly and *t* is enemy/neutral, and expose the first hop *p₁→p₂* as a **synthetic candidate** with edge features encoding ultimate target *t* (min-hop, intercept to *t* via *p₂*, relay id). Cap at 1–2 synthetic slots per source to preserve static JIT layout.
**Axis:** Multi-hop / indirect routing
**Basis:** `reasoned:` user A\* hypothesis; `external:` GNN routing k-candidate pattern (Rusek 2019) — expand candidates, not full path macro
**Rationale:** Addresses "no direct edge p₁→t₁" without exploding action space or requiring multi-action path execution in one env step (which the game doesn't support).
**Downsides:** Feature/encoder changes; relay correctness on rotating maps; risk of stale relay if *p₂* captured mid-turn; more complex than ownership ranking alone.
**Confidence:** 60%
**Complexity:** High
**Status:** Unexplored

### 5. Reachability features on existing edges (defer full routing)

**Description:** Add edge catalog channels: graph hop distance to nearest enemy from target, whether target lies on a shortest path from source to frontier, "relay usefulness" score. Lets policy learn multi-hop intent while action space stays single-hop.
**Axis:** Multi-hop / indirect routing
**Basis:** `direct:` parametric edge catalog in `src/features/catalog/edge.py`; lower risk than synthetic edges
**Rationale:** Smallest step toward routing awareness; pairs well with ownership-tier ranking (#2).
**Downsides:** Slower learning than structural masks; doesn't help if correct first hop never enters top-K.
**Confidence:** 65%
**Complexity:** Medium
**Status:** Unexplored

### 6. Launch hygiene telemetry + training gates

**Description:** Emit `duplicate_launch_rate`, `friendly_ping_pong_rate`, `mean_ships_per_launch`, `slots_used_per_turn` from rollout metrics; add optional curriculum gate or benchmark threshold before promotion. Use replay parser as regression test fixture.
**Axis:** Turn structure (observability)
**Basis:** `direct:` replay parser from prior session; AGENTS.md threshold discipline
**Rationale:** Verifies fixes and prevents regressions; cheap to add in parallel with structural changes.
**Downsides:** Metrics alone don't fix behavior; denominator definitions need care (per-turn vs per-slot).
**Confidence:** 80%
**Complexity:** Low
**Status:** Unexplored

### 7. Minimum effective ship fraction mask (continuous mode)

**Description:** After first launch from a source in a turn, mask ship fractions below `max(f₁, remaining * min_fraction)` unless stop; or mask re-picking same source with fraction below cumulative threshold. Targets 1-ship dribbling specifically.
**Axis:** Within-step dedup & sizing
**Basis:** `direct:` u5000 replay mean 1.0 ships/launch with `ship_action_mode: continuous_fraction`
**Rationale:** Complements dedup; addresses sizing when duplicates are numerically distinct but economically identical.
**Downsides:** May block legitimate probe launches; threshold tuning; less principled than merge (#3).
**Confidence:** 55%
**Complexity:** Low–Medium
**Status:** Unexplored

## Rejection Summary

| #  | Idea                                                | Reason Rejected                                                                                                    |
| :- | :-------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| 1  | Remove all friendly→friendly edges                  | Too aggressive; blocks rare legitimate same-turn staging user may not have seen; prefer tier ranking + reverse ban |
| 2  | Set-transformer / SAINT policy (order-free)         | Subject-replacement scale; full decoder rework vs surgical masks                                                   |
| 3  | NPM learned dedup only                              | No hard structural rules first; slow and unverifiable for obvious spam                                             |
| 4  | Full A\* macro action (whole path one pick)         | Game executes one hop per launch; macro doesn't map to env API; high complexity                                    |
| 5  | max\_moves\_k=1                                     | Treats symptom; loses multi-target turns (attack two enemies same turn)                                            |
| 6  | Env-only dedup in `_launch_fleets`                  | Fixes outcomes without training signal; policy keeps learning spam                                                 |
| 7  | Reward penalty for small launches only              | Slow vs structural masks; threshold invention risk                                                                 |
| 8  | Defer-launch timing as sole fix                     | User's moving-planet counterpoint valid; doesn't stop duplicate same-target spam                                   |
| 9  | Global edge ranking (not per-source)                | Deferred — revisit if per-source tie-breaker insufficient; JIT/layout break acceptable when ROI is clear           |
| 10 | axis: turn structure — marginal Shapley reward only | High implementation cost; better after hygiene bundle proves insufficient                                          |

## Ping-Pong Devil's Advocate

**Could same-turn friendly A→B then B→A ever help?** Only if intermediate production or capture changes ownership mid-turn — not in standard step order (launch phase completes before movement). **Staging A→B then B→C (toward enemy)** is useful and is **not** blocked by reverse-edge ban (only blocks B→A after A→B). **Multi-turn relay** remains legal across turns. User's "no ping-pong" claim holds for same-turn reverse pairs; ideation accepts it with the narrow mask definition above.

## Suggested Sequencing

1. Telemetry (#6) + ownership-tier ranking (#2) — low risk, measurable
2. Launch hygiene bundle (#1) — core user ask
3. In-scan merge (#3) if duplicates persist
4. Reachability features (#5) before synthetic two-hop edges (#4)
