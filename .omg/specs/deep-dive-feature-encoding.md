# Deep Dive Spec: Feature Encoding

**Status:** Draft (interview in progress)  
**Trace:** [.omg/specs/deep-dive-trace-feature-encoding.md](deep-dive-trace-feature-encoding.md)  
**Ambiguity:** ~55% (Goal clarified; constraints/success criteria pending)

## Goal

Balanced improvement across all three trace themes:

1. **Redundancy (Lane A):** Identify and remove/compress provably redundant features; reduce encoder maintenance burden between Python and JAX paths.
2. **Config leverage (Lane B):** Make feature volume explicit (dim math), resolve dead normalization config, and gather evidence for history-depth tuning.
3. **Parity & measurement (Lane C):** Harden Python↔JAX semantic equivalence, add cross-encoder tests, and establish ablation/measurement hooks before risky changes.

## Trace Findings (Summary)

### Proven redundancies (high confidence)

| Issue | Dims | Location |
|-------|------|----------|
| `history_present_flag` ≡ `ownership_stable_flag` | 1 | `encoding.py` L360–361, `jax/features.py` L325–326 |
| Self ↔ global owner block + aggregates | ~17 | Shared `owner_relative_summary()` broadcast per source |
| Candidate `source_ships` duplicates self | 1 × C slots | Same source value in every candidate row |
| Python `always_on_marker` constant 1.0 | 1 | Zero information; JAX uses slot for `ordered_valid` |

### Config / volume

- Default: **171 floats/decision row** (H=1, C=4). Max sweep: **9180** (H=20, C=16).
- `model.normalize_observations` and `ObservationNormalizer` are **not wired** into JAX training.
- `feature_history_steps` default remains 1; no in-repo win-rate ablation results for H>1.

### Parity risks (ranked)

1. **`candidate_mask`:** Python includes sun-crossing targets; JAX excludes them (`ordered_valid & ~crosses`).
2. **Candidate selection:** Python uses trajectory shield in `build_candidates`; JAX uses sun-only lexsort.
3. **Hardcoded Python slice indices** vs JAX registry slices for history deltas.
4. **No cross-encoder value parity test** exists.

## Proposed Execution Sequence

1. **Parity harness** — cross-encoder test + document mask contract (Lane C).
2. **Low-risk dedup** — remove exact duplicates (ownership_stable alias, constant always_on_marker) with dim bump + checkpoint metadata update (Lane A).
3. **Ablation baseline** — short runs to validate self↔global block compressibility and H=1 vs H=10 (Lanes A + B).
4. **Encoder consolidation** — shared primitives, registry-driven slices in Python, reconcile JAX slot 23 (Lanes A + C).
5. **Config cleanup** — wire or remove `normalize_observations`; document dim cost formula (Lane B).

## Constraints (Pending User Confirmation)

- [ ] Checkpoint breaking changes acceptable for feature dim reduction?
- [ ] Python and JAX encoders must converge to identical mask/selection semantics?
- [ ] Minimum win-rate regression tolerance for ablation-gated dedup?

## Non-Goals (Tentative)

- New game features / reward shaping changes (unless ablation reveals gap).
- Policy architecture changes (separate MLPs per group) in first pass.
- Full encoder rewrite before parity tests land.

## Success Criteria (Draft)

- [ ] Cross-encoder parity test passes on fixtures including sun-crossing targets.
- [ ] Feature dim reduction documented with before/after counts.
- [ ] Proven exact duplicates removed or merged in registry + both encoders.
- [ ] `normalize_observations` either wired into JAX path or removed from config.
- [ ] Ablation script or documented procedure for history depth / dedup validation.
- [ ] All existing tests pass; new parity/redundancy tests added.

## Interview Transcript

| Round | Question | Answer |
|-------|----------|--------|
| 1 | Primary goal across trace themes? | **All three** — balanced pass (redundancy + config + parity) |
