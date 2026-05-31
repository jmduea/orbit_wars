# Deep Dive Trace: Feature Encoding

## Problem

Explore how Orbit Wars performs feature engineering in depth: identify redundancies, improvement areas, and optimizations across the self / candidate / global encoder pipeline without silently breaking JAX/Python parity or checkpoint compatibility.

## Trace Lanes

### Lane A: Code-path & Schema Redundancy

**Most likely explanation:** A meaningful share of the ~99 base feature dims is redundant across groups, and dual Python/JAX encoders add maintenance cost with semantic drift at a few slots.

**Evidence for:**

- **Within-self exact duplicate:** `history_present_flag` ≡ `ownership_stable_flag` — both set from the same variable in `src/features/encoding.py` (L360–361) and `src/jax/features.py` (L325–326). Tests confirm stacked indices 25 and 26 both equal `1.0` (`tests/test_feature_history.py` L54–55).
- **Self ↔ global cross-group duplicate (~17 dims/decision):** Both call `owner_relative_summary()` and broadcast identical owner-relative blocks. Tests assert byte-for-byte equality on overlapping slices (`tests/test_features.py` L74–96, `tests/test_jax_env.py` L223–246). Self aggregate scalars (`friendly_planet_count`, etc.) match global `planet_fractions` / `ship_fractions`.
- **Global broadcast per source:** `np.repeat` / `jnp.repeat` duplicates the same global vector onto every self row (`encoding.py` L170–174, `jax/features.py` L92–96).
- **Candidate `source_ships` duplicates self `source_ships`:** Same normalized value per source in every candidate row (`encoding.py` L348, L456).
- **Python `always_on_marker` is constant 1.0** — zero information (`encoding.py` L463). JAX repurposes slot 23 as `ordered_valid` target-validity mask (`jax/features.py` L458–463) — semantic drift.
- **Policy architecture amplifies overlap:** Separate MLPs per group (`src/jax/policy.py` L67–79) then concatenate embeddings that share raw information.
- **Within-candidate geometric redundancy:** `delta_coords` (2) + `distance_to_target` (1) are derivable from each other.
- **Dual ownership encoding on candidates:** 3-way `target_ownership_flags` plus 4-slot `relative_owner_slots` (partially redundant in 2p).

**Evidence against:**

- `incoming_*_pressure` differs by context (source vs target planet) — not blanket redundant.
- `relative_owner_slots` disambiguates enemies in 4p where flags collapse to trichotomy.
- Self-only signals (`source_coords`, `outgoing_friendly_ships`, per-source `ship_delta`) are not in global.
- Global-only signals (fleet totals, production, four delta slot groups) are not in self.
- Separate MLPs *could* learn complementary projections — unproven without ablation.

**Critical unknown:** Does the policy rely on redundant dimensions for performance, or are they compressible without metric loss?

**Discriminating probe:** Short controlled ablation — zero self owner-global block (`self[11:24]`) or global equivalent and compare `overall_win_rate` vs baseline.

---

### Lane B: Config, History & Normalization Leverage

**Most likely explanation:** `feature_history_steps` and `candidate_count` linearly inflate feature volume; ablation evidence is thin; `ObservationNormalizer` and `model.normalize_observations` are disconnected from JAX training.

**Evidence for:**

- **Dim math:** Base dims self=30, candidate=24/slot, global=45. Total per decision row = `H × (75 + 24C)` where H=`feature_history_steps`, C=`candidate_count`. Default (H=1, C=4): **171 floats/row**. Max sweep (H=20, C=16): **9180 floats/row** (20× history jump on default C=4: 171→3420).
- **History stacking:** `FeatureGroupRegistry.total_dim = base_dim × history_steps` (`registry.py` L48–50). Both encoders concatenate H frames.
- **Default remains minimal:** `conf/task/default.yaml` sets `feature_history_steps: 1` despite `conf/sweeps/wandb/task_complexity.yaml` sweeping {5,10,20}. No in-repo published win-rate results for history depth.
- **Normalization dead on JAX path:** `ObservationNormalizer` only used in optional Python opponent inference (`src/opponents/runtime.py`). JAX rollout/PPO passes raw `encode_turn` output to `policy.apply` (`jax/ppo_update.py`, `opponents/jax_actions/builders.py`). `model.normalize_observations: true` in model YAMLs is never read in `src/`.
- **`player_count` does not multiply dims** — fixed 4-slot owner encoding (padding for shape stability).
- **Schema vs Hydra default mismatch:** `TaskConfig.candidate_count` default 8 in schema vs 4 in Hydra.

**Evidence against:**

- Hand-normalization at encode time (coords/BOARD_SIZE, ships/max_ships, etc.).
- Policy uses LayerNorm inside attention/GNN encoders.
- `task_complexity` sweep infrastructure exists with `overall_win_rate` objective — machinery present, results not documented in-repo.
- H=1 coherently disables both history stacking and temporal delta features.

**Critical unknown:** Does increasing `feature_history_steps` from 1 to {5,10,20} improve win rate enough to justify linear compute/memory cost?

**Discriminating probe:** 2×2 ablation at fixed model/updates: (H=1 vs H=10) × (normalizer off vs wired into JAX rollout/PPO).

---

### Lane C: Measurement, Parity & Drift Risk

**Most likely explanation:** Parity guardrails are dimensional/schema-static, not semantic; Python vs JAX diverge on candidate masking and selection; optimization can pass tests while changing what the policy sees.

**Evidence for:**

- **No Python↔JAX feature value parity harness.** Tests cover shapes, history alignment per-path, and a few hardcoded slice indices — not cross-encoder diff.
- **`candidate_mask` divergence (highest risk):** Python marks all ordered real targets `True` including sun-crossing (`encoding.py` L467–468). JAX uses `ordered_valid & (~crosses)` — sun-crossing excluded from mask (`jax/features.py` L477–478). No cross-path test.
- **Candidate selection divergence:** Python `build_candidates` uses trajectory shield (`any_ship_bucket_is_safe`); JAX ranks all active non-self planets via lexsort with sun-crossing only. Python `build_candidate_features` re-sorts pre-filtered list by sun only — internal Python inconsistency too.
- **Hardcoded Python slice indices:** Global deltas use `prior[8:12]`, `[12:16]`, etc. (`encoding.py` L498–502). JAX uses `GLOBAL_FEATURE_SCHEMA.base_slice(...)`. Tests mirror hardcoded indices — brittle if registry reordered.
- **Checkpoint compat dim-only:** `validate_checkpoint_feature_compatibility` checks three aggregate dims; no semantic hash; silent pass if metadata missing.
- **No feature ablation/importance/redundancy tooling** in repo. Rollout metrics are behavioral, not feature-quality.

**Evidence against:**

- Registry validates base dims at import; slice tests pin critical positions.
- Trajectory shield tests document Python's permissive mask policy (`test_trajectory_shield.py`).
- Checkpoint metadata written on JAX save.
- History alignment well-tested per-path for reorder scenarios.

**Critical unknown:** Are Python replay/submission and JAX training meant to share identical `candidate_mask` semantics, or is the permissive/restrictive split an accepted contract?

**Discriminating probe:** Parametrized parity test — same GameState → Python vs JAX `encode_turn`, assert `candidate_ids`, `candidate_features`, `candidate_mask` including sun-crossing fixture.

---

## Rebuttal Round

**Challenge to Lane A leader (self↔global duplication):** Separate MLPs may orthogonalize shared inputs, so raw overlap ≠ learned redundancy.

**Resolution:** Existence of redundancy is established (tests prove value identity). Whether pruning hurts performance requires ablation — but ~17/30 self dims and compute bandwidth are wasted pre-MLP regardless of post-MLP decorrelation.

**Challenge to Lane C (mask divergence intentional):** Python defers legality to trajectory shield; JAX pre-filters sun crossings.

**Resolution:** Even if intentional, it is undocumented as a cross-path invariant. Refactors touching candidate ordering or masks can silently change training behavior — treat as high drift risk until parity probe confirms contract.

---

## Convergence

All three lanes **substantially support** the investigation premise:

| Theme | Confidence | Top action candidates |
|-------|------------|----------------------|
| Redundancy (Lane A) | **High** | Remove `ownership_stable_flag` alias; dedupe self↔global owner block; drop constant `always_on_marker`; reconcile JAX slot 23 |
| Config leverage (Lane B) | **High** | Document dim cost formula; run history/normalizer ablation; wire or remove dead `normalize_observations` config |
| Parity safety (Lane C) | **High** | Add cross-encoder parity test; document mask contract; migrate Python hardcoded slices to registry |

**Recommended sequencing for execution:**

1. **Safety first:** Cross-encoder parity harness + document mask semantics (Lane C probe).
2. **Low-risk wins:** Remove proven exact duplicates (history/ownership alias, constant marker) with checkpoint dim update.
3. **Evidence-gated:** Self↔global dedup and history depth only after ablation shows no regression.
4. **Perf:** JAX encoder consolidation / shared primitive module after parity locked.

---

## Quantified Summary

| Metric | Value |
|--------|-------|
| Base dims (self / candidate / global) | 30 / 24 / 45 |
| Default total floats per decision row (H=1, C=4) | 171 |
| Proven exact duplicate dims (within self) | 1 (`ownership_stable_flag`) |
| Proven cross-group duplicate dims (self↔global) | ~17 |
| Candidate per-row duplicate from self | 1 (`source_ships` × C slots) |
| Python/JAX semantic drift slots | ≥2 (`always_on_marker`, self `bias`) |
| Cross-encoder parity test coverage | **None** (value-level) |

---

## Interview Injection Queue

Per-lane critical unknowns for deep-interview Phase 4:

1. **Goal priority:** Redundancy removal vs perf optimization vs new features vs ablation tooling?
2. **Parity contract:** Should Python and JAX encoders converge on mask/selection semantics?
3. **Checkpoint policy:** OK to bump feature dims (breaking old checkpoints) for dedup?
4. **History depth:** Treat H=1 as canonical until ablation proves otherwise?
5. **Normalization:** Wire `ObservationNormalizer` into JAX or remove dead config?
