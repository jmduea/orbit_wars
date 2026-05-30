# Deep Dive Spec: Feature Encoding v2 Greenfield

**Status:** Approved (interview) → ralplan  
**Trace:** `.omg/specs/deep-dive-trace-feature-encoding.md`  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**Design:** `docs/feature-encoding-v2-design.md`, `docs/feature-encoding-v2-pointer.md`

## Goal

Greenfield **v2 feature encoding stack**: new schema + new policy input interface, built **side-by-side with v1** until ablation proves cutover. **JAX-only** canonical encoder for training and submission when v2 wins. Comprehensive documentation required.

## Decisions (Interview-Locked)

| Topic | Decision |
|-------|----------|
| Scope | Full-stack greenfield v2 + side-by-side v1 |
| Encoder | Planet-centric tensor + hybrid edge features |
| Models | GNN pointer + Transformer on same encoder (ralplan: **GNN first** in v2 v1) |
| History | Planet deltas + **global-only stack** when H>1 |
| Action space | **Joint pointer** over `(source, target)` + ship bucket |
| Relational signals | Planet tensor + edge tensor (owned×active) |
| v1 fate | Functional until Phase 5 cutover after ablation |

## v2 Architecture (Target)

```
planet_features:  (MAX_PLANETS, P)
planet_mask:      (MAX_PLANETS,)
edge_features:    representation TBD in Phase 0 (sparse/top-K per source recommended)
edge_mask:        valid (source, target) pairs
global_features:  (G,) + optional (H * G,) history stack
```

Policy v2 v1: **`gnn_pointer_v2`** primary; transformer adapter deferred.

## Non-Goals (v2 v1)

- Removing v1 before ablation evidence
- Python encoder removal before submission migrates to JAX v2
- Transformer v2 policy in first tranche
- v1 incremental dedup (separate optional work; superseded by v2 greenfield)

## Success Criteria

See ralplan for phased exits and numeric cutover gates.

## Interview Transcript

| Round | Answer |
|-------|--------|
| Primary goal | All three trace themes (balanced) |
| Constraint | Full rework + docs; JAX-only at cutover |
| Greenfield scope | Full stack + side-by-side v2 |
| Model target | B + C unified (planet tensor) |
| History | Global-stack + deltas |
| Candidates | Pointer → refined to **joint (source, target)** |
| Pointer relations | Hybrid edges |
| Next step | ralplan |
