# Ralplan: PPO GAE + Gradient Checkpointing

Spec: `.omg/specs/deep-interview-ppo-gae-gradient-checkpoint.md`  
Parent audit: `.omg/specs/deep-dive-ppo-encoder-decoder-audit.md`  
Status: planned  
Iterations: 1 (consensus on first pass)

## RALPLAN-DR

### Principles

1. **Backward-compatible defaults** — existing Hydra runs should behave identically unless `gae_lambda` is explicitly lowered from 1.0.
2. **Rollout owns advantage math** — GAE belongs in `collect_rollout_jax` beside `discounted_returns`; PPO update stays agnostic.
3. **Encoder-only remat first** — memory win is dominated by planet-edge encoder; avoid decoder autoregressive remat complexity in v1.
4. **Prove with unit tests, not slow smokes** — GAE and remat toggles get CPU/light-JAX tests; no new mandatory rollout compile tests.
5. **Single config surface** — both knobs live under `training.*` and compose through existing Hydra groups.

### Decision Drivers

1. Turn-level value is already stored per rollout step; K-step launch logits share one critic output — GAE must run on `(rollout_steps, num_envs)` then broadcast to K.
2. `conf/training/default.yaml` sets `enable_gradient_checkpointing: true` today but it is a no-op — users assume it works.
3. JIT shape stability: remat must not change parameter keys or output shapes (checkpoint compatibility).
4. WSL2 agent workflow: verification tier is `make test-fast` + targeted `test_ppo_update.py` JAX tests.

### Viable Options

#### GAE wiring

| Option | Pros | Cons |
|--------|------|------|
| **A: Add `gae_lambda`, default `1.0` (chosen)** | Zero behavior change for existing configs; opt-in bias-variance tradeoff via YAML | Users must actively tune below 1.0 for standard PPO GAE |
| **B: Add `gae_lambda`, default `0.95` (chosen — user)** | Matches canonical PPO; configurable via Hydra | Changes advantage bias vs prior MC-only runs |
| **C: Document MC-only, no GAE** | No code churn | Leaves audit follow-up unresolved |

#### Gradient checkpointing wiring

| Option | Pros | Cons |
|--------|------|------|
| **A: Encoder layer remat via shared helper (chosen)** | Targets memory hotspot; same param tree; works for GNN + transformer backbones | Does not reduce decoder memory |
| **B: Full policy remat** | Maximum memory savings | Riskier JIT/compile interactions; harder to debug |
| **C: Remove config flag** | Eliminates dead config | Contradicts baseline sweep guidance and default YAML |

**Chosen:** GAE Option B (default λ=0.95, Hydra-configurable) + Remat Option A.

## User Decision (ralplan hooks)

- **GAE:** `training.gae_lambda` default **0.95**, overridable via Hydra (`training.gae_lambda=1.0` restores MC-like behavior).
- **Remat:** encoder-only `nn.remat` wired to `enable_gradient_checkpointing`.
- **Execution:** `/ralph` sequential loop with verification.

## ADR

**Decision:** Add `training.gae_lambda: float = 0.95`, implement `gae_returns_and_advantages()` in `src/jax/ppo_update.py`, call from `collect_rollout_jax`, and wire `training.enable_gradient_checkpointing` to encoder layer blocks using a shared `maybe_remat_layer()` helper passed into `PlanetEdgeBackboneEncoder` and `PlanetGraphTransformerEncoder`.

**Drivers:** Audit follow-up; user chose canonical PPO default λ=0.95; default YAML already enables checkpointing.

**Alternatives rejected:**
- Default λ=1.0 — user preferred 0.95 with Hydra override path
- Full-policy remat — scope/compile risk vs encoder-only win
- Removing checkpoint flag — breaks documented baseline comfort settings

**Consequences:**
- Slightly more rollout compute when λ<1 (negligible vs env step)
- Remat increases backward recompute time but lowers activation memory — net effect depends on GPU; flag remains tunable
- Architecture doc `docs/architecture/jax-policy-encoder.md` gets a short remat note

## Architect Review

**Approved with notes:**

- **GAE bootstrap:** At final rollout step, bootstrap `next_value=0` when `done`, else use `values[t+1]` from same rollout (standard on-policy). Document that we do not re-forward the policy for terminal bootstrap in v1.
- **λ=1 equivalence:** With `gae_lambda=1.0`, `gae_returns` should match `discounted_returns` and advantages should match `returns - values` on toy tensors (test this explicitly).
- **Broadcast contract:** After GAE, `returns` and `advantages` still broadcast to `(rollout_steps, num_envs, max_moves_k)`; factorized `step_mask` unchanged.
- **Remat placement:** Wrap each GNN message-passing block and each transformer sub-block (attn+ffn) — not individual Dense kernels.
- **Config threading:** Pass `gradient_checkpointing: bool` from `build_jax_policy` → encoder constructors; do not thread through decoder modules yet.

**Risk:** Flax `nn.remat` + `scan` in autoregressive decoder could interact badly — mitigated by encoder-only scope.

## Critic Review

**Approved** with test checklist:

| # | Check | Command / location |
|---|-------|-------------------|
| 1 | GAE toy tensor: λ=1 matches MC | `tests/test_ppo_update.py` |
| 2 | GAE toy tensor: λ=0.95 differs from MC | same |
| 3 | Hydra compose `training.gae_lambda=0.9` | parametrized config test |
| 4 | Remat on/off policy init+apply smoke | `tests/test_ppo_update.py` or `test_jax_policy_encoder.py` |
| 5 | Factorized + joint PPO update with remat on | light JAX test |
| 6 | No new slow-tier requirement | `make test-fast` |

**Non-blocking follow-ups:** profile remat wall-time on WSL2/CUDA before changing default YAML; optional `docs/experiments.md` note on λ tuning.

## Implementation Phases

| Phase | Deliverable | Verify |
|-------|-------------|--------|
| 0 | Manifest + this plan | — |
| 1 | `TrainingConfig.gae_lambda`, `conf/training/*.yaml`, runtime validation (`0 <= λ <= 1`) | `test_ppo_update` Hydra case |
| 2 | `gae_returns_and_advantages()` + rollout wiring | GAE unit tests |
| 3 | `maybe_remat()` helper + encoder wiring | init/apply smoke both architectures |
| 4 | Extend `tests/test_ppo_update.py` | `make test-fast` + targeted JAX |
| 5 | Update `docs/architecture/jax-policy-encoder.md` | doc sanity |

## Files Touched (expected)

| File | Change |
|------|--------|
| `src/config/schema.py` | `gae_lambda: float = 1.0` |
| `conf/training/default.yaml`, `ablation_m2.yaml` | optional explicit `gae_lambda` |
| `src/config/runtime.py` | validate λ range |
| `src/jax/ppo_update.py` | `gae_returns_and_advantages()` |
| `src/jax/rollout/collect.py` | use GAE when building transitions |
| `src/jax/encoders/planet_encoder_common.py` or new `remat.py` | `maybe_remat_call()` |
| `src/jax/policy.py` | GNN layer remat |
| `src/jax/encoders/planet_graph_transformer.py` | transformer layer remat |
| `src/jax/policy.py` (`build_*`) | pass checkpoint flag |
| `tests/test_ppo_update.py` | GAE + remat tests |
| `docs/architecture/jax-policy-encoder.md` | remat note |

## Workflow Gates

- [x] Planner draft
- [x] Architect approval
- [x] Critic approval
- [ ] User final approval (execution bridge)
