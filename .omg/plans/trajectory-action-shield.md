# Ralplan: Trajectory Action Shield

Date: 2026-05-21
Status: Revised after critic rejection, pending final approval
Spec: `.omg/specs/deep-interview-trajectory-action-shield.md`

## Mode

Deliberate. This crosses PPO sampling semantics, generated submission latency, JAX rollout throughput, config compatibility, diagnostics, and multiple action emitters.

## Principles

- The shield must affect the action actually stored for PPO, not only the emitted move.
- Legality is bucket-conditioned: target plus concrete ship count plus resulting fleet speed.
- No repo-owned emitter may bypass the same safety semantics.
- Full remaining horizon is the default, with conservative cap behavior available.
- Diagnostics must make shield pressure visible during training and validation.

## Decision Drivers

1. PPO correctness: sampled `target_index`, `ship_bucket`, `log_prob`, and emitted `JaxAction` must describe the same legal action.
2. Runtime safety: replay, packaged Kaggle `main.py`, self-play, scripted/random opponents, and Python/JAX env stepping must all filter unsafe launches.
3. Performance: full-horizon forecasting must stay inside JAX rollout and Kaggle 1 second action budgets.

## Options Considered

### Emitter-Only Post-Filtering

Rejected. It hides unsafe sampled actions from PPO storage and leaves `log_prob` inconsistent with executed behavior.

### Target-Only Candidate Masking

Rejected. A target can be safe for one ship bucket and unsafe for another because ship count changes fleet speed.

### Bucket-Conditioned Shield Before Sampling

Chosen. It preserves PPO/action consistency and gives diagnostics a single source of truth.

## Decision

Implement a bucket-conditioned trajectory action shield before PPO sampling and log-prob storage, then reuse the same semantics across all repo-owned action emitters. JAX semantics are authoritative for training; Python mirrors those semantics for runtime emitters and packaged submissions, with parity tests as the contract.

## Revised Implementation Plan

### 1. Add Shield Config And Observation Physics Inputs

- Add shield fields to `src/conf_schema.py`, `conf/config.yaml`, and validation in `src/config.py`:
  - `trajectory_shield_enabled`
  - `trajectory_shield_hit_mode`: `selected_target` or `non_friendly`
  - `trajectory_shield_horizon`: full remaining horizon by default, with configurable cap
  - `trajectory_shield_epsilon`
- Extend `src/game_types.py` parsing to retain `angular_velocity` and `initial_planets` when present, with safe defaults for older observations.
- Update generated submission `CONFIG_TEMPLATE`, fake config tests, and old checkpoint/config tolerance.
- Regenerate or check `default_cfg.yaml` with `scripts/generate_default_cfg.py`.

Acceptance:

- Old checkpoints/config payloads missing shield fields load with defaults.
- Generated runtime template compiles.
- Default config check passes.

### 2. Implement JAX Shield Semantics And Python Mirror

- Implement JAX-native vectorized shield using launch offset, `fleet_speed`, swept moving-planet collision, first sun intersection time, first bounds crossing time, and rotating positions from `initial_planets`, `angular_velocity`, and `step`.
- Implement Python mirror for replay, local opponents, Python env last-mile filtering, and generated Kaggle `main.py`.
- Event ordering rule: compute first event time within each simulated step. Acceptable planet hit is safe only when it occurs strictly before sun/bounds and before any unacceptable planet hit, within epsilon. Same-time conflicts are blocked conservatively.
- Non-friendly mode: neutral owner `-1` and enemy owners count as acceptable; friendly owner equal to acting player is blocked. Selected-target mode only accepts the selected target.

Acceptance:

- Python and JAX predicates agree on static, rotating, sun, bounds, intended-hit, unintended-hit, no-hit-through-horizon, and same-step conflict cases.
- Direct sun-crossing behavior remains blocked.

### 3. Make PPO Sampling Bucket-Conditioned Before Storage

- Add target-by-ship-bucket legality before learner rollout sampling in `src/jax_ppo.py`.
- Replace learner rollout sampling with shielded sequential sampling:
  - target mask is `candidate_mask & any_legal_bucket`;
  - selected bucket mask is legal buckets for the sampled target;
  - canonical no-op is `(target=0, bucket=0)`;
  - if no legal non-noop remains, sampled action, stored transition, log-prob, and emitted action all become canonical no-op.
- Store enough legality masks in `JaxTransitionBatch` for PPO update log-prob recomputation, so update-time probability matches rollout-time shielded distribution.
- For pointer models, use a scan over K sequence steps because later-step legality depends on ships consumed by earlier sampled buckets.

Acceptance:

- Rollout transition `target_index`, `ship_bucket`, `log_prob`, and `build_action_from_batch` output always match the shielded sampled action.
- PPO update recomputes log-prob against the same shielded distribution used during rollout.
- No post-hoc action drop can create policy/action drift.

### 4. Cover Every Repo-Owned Action Emitter

- `src/replay.py`: apply deterministic shielded target/bucket selection before move construction.
- `scripts/validate_kaggle_docker_submission.py`: embed config and shield helpers in generated `main.py`; shield before moves are appended.
- `src/opponents.py`: shield `SelfPlayOpponent.act`, `SniperOpponent.act`, and post-filter `KaggleRandomOpponent.act` returned moves.
- `src/jax_ppo.py`: shield `_sample_policy_action_with_params`, `_sample_policy_action`, `build_random_action_from_batch`, `build_sniper_action_from_batch`, and latest/historical/random/scripted/noop branches.
- `src/env.py`: last-mile shield learner and active opponent moves before `env.step` where the repo controls those moves.
- Out of scope: modifying Kaggle external built-in random agent internals. Returned moves from that agent remain in scope for post-filtering before use.

Acceptance:

- Replay, self-play, scripted, random, generated submission, and JAX action builders do not emit shield-invalid moves.
- All unsafe non-noops fall back to no emitted move/no-op.

### 5. Add Diagnostics Plumbing

- Add rollout metrics in `src/jax_ppo.py`:
  - `trajectory_shield_blocked_count`
  - `trajectory_shield_blocked_sun_count`
  - `trajectory_shield_blocked_bounds_count`
  - `trajectory_shield_blocked_unintended_hit_count`
  - `trajectory_shield_blocked_horizon_count`
  - `trajectory_shield_fallback_noop_count`
  - shielded legal non-noop rate
- Thread metric keys through `src/jax_train.py` scalar keys, JSONL records, and telemetry records in `src/telemetry.py` where applicable.

Acceptance:

- Training logs expose blocked action totals, blocked reason counts, fallback no-op counts, and legal non-noop rate.
- Tests fail if a new rollout diagnostic is computed but not logged where expected.

### 6. Add Focused Tests And Benchmarks

Unit tests:

- direct sun crossing blocked;
- moving target miss followed by sun hit blocked;
- out-of-bounds before acceptable hit blocked;
- selected-target mode blocks unintended hits;
- non-friendly mode permits neutral/enemy hits and blocks friendly hits;
- all unsafe non-noops fall back to no-op;
- same-step planet-before-sun allowed, sun-before-planet blocked, bounds-before-hit blocked, and tie conflicts blocked.

Parity tests:

- Python vs JAX legality agreement across static, rotating, sun, bounds, selected-target, non-friendly, and horizon cases.

PPO tests:

- bucket-conditioned sampling happens before `log_prob`;
- canonical no-op is stored when all non-noops are unsafe;
- PPO update recomputes log-prob with stored shield masks;
- K-step pointer legality accounts for remaining ships after earlier sequence steps.

Emitter tests:

- replay, generated `main.py`, self-play, random, sniper, JAX random/sniper/latest/historical/noop, and Python env last-mile filtering skip unsafe moves.

Config tests:

- Hydra composition;
- generated default config check;
- generated submission template compile;
- fake config update;
- old checkpoint/config tolerance.

Benchmark gates:

- JAX rollout: compile excluded, steady-state rollout with shield on must be no worse than 25 percent slower than same-commit shield-off baseline for the default mixed 2p/4p config; no new OOM or recompilation per update.
- JAX vectorization: fixed-shape `vmap` over env/source/target/bucket and `lax.scan` over capped horizon.
- Horizon cap fallback: default cap equals full remaining episode horizon. If user lowers cap and no acceptable hit occurs before cap, classify as `horizon_exhausted` and block conservatively.
- Generated Kaggle `main.py`: Docker validation must report `first_action_seconds <= 0.90s`, warmed p95 action latency `<= 0.20s`, and max action latency `<= 0.50s` for 2p and 4p smoke runs under the existing 1 second timeout.

## Architect Concerns To Preserve During Execution

- Treat bucket-conditioned legality before PPO sampling/log-prob/action storage as a hard requirement.
- K-step pointer sampling must be a sequence scan, because remaining ships affect later legality.
- Runtime emitters need final guards, but guards are defense-in-depth rather than the primary source of truth.
- Generated package file lists must include any new runtime shield module.
- Diagnostics must be added at all layers where rollout metrics are computed, transferred, and recorded.

## Consequences

- Transition storage grows to retain shield masks for PPO updates.
- Rollout sampling becomes more expensive.
- Generated submission gets more embedded logic or an additional packaged runtime helper.
- The benefit is that unsafe launches are blocked at the policy distribution level, final emitters are protected, and shield pressure is measured during training.

## Follow-Ups After Implementation

- Re-run architect/critic review on mask storage size, JAX compile behavior, generated template maintainability, and benchmark thresholds.
- If full remaining horizon is too expensive, keep conservative horizon-exhausted blocking while tuning caps with telemetry.
