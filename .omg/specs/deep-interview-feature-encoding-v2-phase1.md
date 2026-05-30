# Deep Interview Spec: Feature Encoding v2 — Phase 1

**Status:** Approved → **implemented** (2026-05-25)  
**Parent spec:** `.omg/specs/deep-dive-feature-encoding.md`  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**Phase 0 contract:** `docs/feature-encoding-v2.md`, `docs/feature-encoding-v2-phase0-results.md`  
**Ambiguity at close:** ~22% (config wiring assumed from ralplan default — user skipped Round 5)

## Goal

Implement the **JAX v2 feature encoder** (`encode_turn_v2` → `JaxTurnBatchV2`) as the new canonical encoding implementation, while **v1 remains the runtime default** until ablation cutover (Phase 5). Phase 1 is **encoder-only** — no env dispatch, rollout, policy, or submission wiring.

## Constraints

| Constraint | Source |
|------------|--------|
| Locked schema P=13, E=12, G=46, K=C−1 | Phase 0 |
| Top-K edges per source, learner frame (θ_ref = sun → owned centroid) | ADR-002, ADR-004 |
| `task.ship_feature_scale` (default 1000), separate from v1 `max_ships` | ADR-003 |
| H=1 and H>1 global history both required before Phase 2 | Interview R4 |
| Side-by-side v1; default `encoding_version=v1` | Interview R3 + assumed R5 |
| No Layer D planet sort | User lock |
| v1 tests unchanged; `make test-domain-features` green | Ralplan Phase 1 |
| No rollout/PPO/shield/submission changes | Ralplan phase boundary |

## Non-Goals (Phase 1)

- `env.py` encode dispatch / training path integration (Phase 2–3)
- Policy `gnn_pointer_v2`, joint pointer decoder, shield edge variant
- Python `encode_turn` removal or submission migration
- Transformer v2, ablation runs, cutover
- Cross-encoder Python↔JAX v2 value parity harness (optional follow-up)

## Acceptance Criteria

1. **`src/features/registry_v2.py`** — ordered planet/edge/global schemas; dim validation against P/E/G lock.
2. **`src/jax/features_v2.py`** — `JaxTurnBatchV2` NamedTuple + `encode_turn_v2`:
   - Planet tensor `(MAX_PLANETS, P)` with sun-polar learner frame
   - Edge tensor `(MAX_PLANETS, K, E)` + masks + src/tgt id side channels
   - Global `(G,)` or `(H*G,)` when `feature_history_steps > 1`
   - Planet `ship_delta`; global-only history stack; edges recomputed each step
   - Edge sort: distance ascending, sun-blocked deprioritized (v1 lexsort spirit)
3. **Config** — `TaskConfig.encoding_version: Literal["v1","v2"]` default `"v1"`; `ship_feature_scale: float = 1000.0`; Hydra `conf/task/` exposure.
4. **Tests** — `tests/test_feature_encoding_v2_golden.py`:
   - Golden vectors: 2p, 4p, sun-cross fixture, H>1 history
   - JIT smoke: `vmap(encode_turn_v2)` at default config
5. **Verification** — `make test-domain-features` passes; no regressions in v1 tests.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| Replace Python encoder in Phase 1? | **No** — JAX v2 canonical implementation, v1 side-by-side (R3) |
| Phase 1 integration depth? | **Encoder-only** — no train/rollout wiring (R2 implied + ralplan) |
| H>1 in Phase 1? | **Yes** — global history stack + golden test (R4) |
| Config default for encoding_version? | **v1** — assumed (R5 skipped, ralplan default) |
| Python registry mirror? | **Yes** — `registry_v2.py` per ralplan |

## Ontology (Key Entities)

| Entity | Role |
|--------|------|
| `JaxTurnBatchV2` | Fixed-shape v2 policy input batch (Phase 2 consumer) |
| `registry_v2` | Schema source of truth for P/E/G slices |
| `encode_turn_v2` | JAX encoder entrypoint |
| `θ_ref` | Learner reference angle for canonical frame |
| `edge_mask` | Top-K validity per source row |
| `encoding_version` | Config dispatch key (wired Phase 2+) |

## Interview Transcript

| Round | Question | Answer |
|-------|----------|--------|
| 1 | Interview scope? | Feature Encoding v2 Phase 1+ |
| 2 | Phase 1 boundary? | (Freeform) "Full replacement of python" |
| 3 | Python replacement meaning? | JAX v2 canonical, side-by-side v1 until ablation |
| 4 | History scope? | H=1 and H>1 both in Phase 1 |
| 5 | Config defaults? | *(skipped — assume encoding_version=v1 default)* |

## Execution Bridge (recommended)

**Ralplan refreshed (iteration 3).** Execute Phase 1 per `.omg/plans/ralplan-feature-encoding-v2.md`:

```bash
make test-domain-features   # after implementation
```

Suggested path: implement directly or `/team` / `/omg-autopilot` for Phase 1 only.
