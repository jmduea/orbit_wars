# Deep Interview Spec: Rollout Process Optimization Audit

## Goal

Produce a **prioritized audit** of the JAX training rollout path—bottlenecks, likely causes, and optimization ideas—without implementing changes yet. The north star is **closing the throughput gap between 4-player / mixed-format rollouts and 2-player-only rollouts** (user: “the closer we can get 4p/mixed_2p_4p performance to 2p_only performance the better”).

## Constraints

- **Throughput-first ranking** using existing telemetry: `rollout_env_steps_per_sec_*`, `rollout_samples_per_sec_*`, `rollout_seconds_*`, `update_time_rollout_fraction`.
- **Flexible on mechanism**; user is open to suggestions. Prefer changes that preserve training semantics where cheap; accept larger refactors if they unlock major 4p gains.
- **In scope formats**: `mix_2p_4p_8env` / `mix_2p_4p_16env` and 2p-only baselines (e.g. `2p_16env`, `2p_32env`, sweep `2p_only_throughput`).
- **Brownfield**: JAX path only (`src/jax/train.py`, `src/jax/ppo.py`, `src/jax/env.py`); Python env parity matters if rollout semantics change.

## Non-Goals

- Implement optimizations in this phase (audit only).
- Optimize PPO update (`ppo_update_jax`) unless rollout audit shows it dominates `update_time_rollout_fraction`.
- Redesign curriculum, reward, or model architecture (unless audit ties them to rollout cost).
- Prove final policy quality from throughput tweaks alone.

## Success Criteria (Audit Deliverable)

1. **End-to-end rollout map** from training loop → per-format JIT collector → `collect_rollout_jax` scan body → env step → metrics sync.
2. **Ranked findings** (P0/P1/P2) with estimated impact class: High / Medium / Low on 4p and mixed-format throughput.
3. **Per-finding notes**: what it costs today, why 4p/mixed hurts more than 2p, suggested direction, validation metric, risk to semantics/tests.
4. **Baseline comparison recipe**: how to A/B `format=mix_2p_4p_8env` vs `format=2p_16env` (or `2p_32env`) at fixed `rollout_steps`, `rollout_microbatch_envs`, model, and `total_updates` smoke length.
5. **Optional profiling checklist** (JAX profiler, split 2p vs 4p `rollout_seconds_*`) for confirmatory work before implementation.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| User wants code changes immediately | **No** — audit first |
| Success = win rate | **No** — throughput primary for this pass |
| Must preserve bitwise-identical rollouts | **Open** — user wants 4p≈2p throughput; parity-tested changes acceptable |
| Pain is only env physics | **Unlikely** — opponent/policy/encode paths dominate structurally |

## Ontology (Key Entities)

| Entity | Role |
|--------|------|
| `JaxRolloutGroup` | Per static format (2p/4p): env state, `jax.jit(collect_fn)` |
| `collect_rollout_jax` | `lax.scan` over `training.rollout_steps`; core hotspot |
| `_collect_rollout_microbatched` | Splits env axis; sequential chunks inside JIT wrapper |
| `batched_step` / `batched_step_multi_player` | 2p vs 4p env advance |
| `encode_turn` / `apply_trajectory_shield` | Per-step feature/shield work |
| Mixed opponent branches | Eager multi-action compute + mask select |
| Training loop (`train_jax`) | Sequential active groups; per-group `device_get` for timing |

## Prioritized Audit Findings (Initial)

### P0 — Structural 4p multiplier inside scan

**Finding:** Each rollout step for 4p rebuilds all player perspectives and runs a Python `for player_id in range(4)` with full self-play branching per player.

```1253:1272:src/jax/ppo.py
        elif cfg.task.player_count == 4:
            player_actions = []
            player_ids = jnp.arange(cfg.task.player_count, dtype=jnp.int32)
            player_games = jax.vmap(
                lambda player_id: state.game._replace(
                    player=jnp.full_like(state.game.step, player_id, dtype=jnp.int32)
                )
            )(player_ids)
            ...
            flat_player_batch = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                flat_player_games
            )
            ...
            for player_id in range(cfg.task.player_count):
```

**Why 4p ≫ 2p:** 2p uses one opponent view + `batched_step`; 4p does up to **4× encode_turn**, **4× policy/heuristic opponent paths**, and `batched_step_multi_player`.

**Directions:** Single fused 4p encode from shared state; `lax.scan`/`fori_loop` over players; branch on `slot_type` *before* computing unused action families; cache player batches across steps when game tree allows.

**Validate:** `rollout_env_steps_per_sec_4p` vs `_2p` at same total env slots; JAX op count / profiler.

**Risk:** Medium — opponent slot semantics must stay curriculum-correct.

---

### P0 — Mixed opponent “compute all, select one”

**Finding:** In mixed self-play, each step builds random, sniper, turtle, opportunistic, noop, historical, and policy actions, then `_select_env_action` by slot type.

**Why 4p ≫ 2p:** Cost scales with player loop (×4) and shield vmaps per player.

**Directions:** `switch`/`lax.switch` on family id; stage-view static specialization when `single_family`; compile separate jitted collectors per curriculum stage family mix.

**Validate:** Rollout seconds at fixed updates; opponent slot telemetry unchanged.

**Risk:** Medium–high if distributions shift.

---

### P1 — Per-step 2p opponent re-encode

**Finding:** After every 2p step, `encode_turn` vmapped over opponent-perspective games updates `next_opp_batch_cache`.

```1429:1435:src/jax/ppo.py
        if cfg.task.player_count == 2:
            next_opp_game = next_state.game._replace(
                player=(1 - next_state.learner_player).astype(jnp.int32)
            )
            next_opp_batch_cache = jax.vmap(lambda game: encode_turn(game, cfg.task))(
                next_opp_game
            )
```

**Directions:** Incremental feature delta from step result; reuse `result.batch` where opponent view aligns with learner batch layout.

**Validate:** `rollout_env_steps_per_sec_2p`; parity tests.

**Risk:** High if feature semantics depend on full re-encode.

---

### P1 — Mixed format: sequential rollout groups per update

**Finding:** Active 2p and 4p groups run **sequentially** in Python; each has separate `jax.jit` collector and timing `device_get`.

```1069:1103:src/jax/train.py
            for group_idx, rollout_key in zip(active_indices, rollout_keys, strict=True):
                group = rollout_groups[group_idx]
                ...
                ) = group.collect_fn(...)
                group_env_steps, group_samples = jax.device_get(...)
```

**Why mixed ≈ 2p + 4p (not half each):** Both formats pay full collect cost when curriculum activates both; no device-side overlap.

**Directions:** `pmap`/`shard_map` across formats if memory allows; alternate updates by format weight; increase 4p envs only when 4p weight high (config, not code).

**Validate:** `update_time_rollout_fraction`; sum of `rollout_seconds_2p` + `rollout_seconds_4p`.

**Risk:** Low for overlap experiments; medium for schedule changes (sample mix).

---

### P1 — Microbatch env chunking

**Finding:** `rollout_microbatch_envs` runs **sequential** `collect_rollout_jax` chunks (slice/concat), trading memory for repeated scan overhead.

**Directions:** Profile default `microbatch=4` vs full width; tune per format; fuse chunks with `lax.scan` over chunks instead of Python loop inside trace.

**Validate:** `samples_per_sec` vs peak HBM; comfort from baseline sweep.

**Risk:** Low.

---

### P2 — Reset `lax.cond(any(done))` every step

**Finding:** Full batched reset path traced whenever any env finishes episode.

**Directions:** Separate reset handling; delayed reset buffer; mask resets without full tree reset (if semantics allow).

**Validate:** Rollout time vs `episode_done` rate.

**Risk:** Medium.

---

### P2 — Heavy per-step transition / diagnostics in scan

**Finding:** Large `transition` dict per scan step; `_rollout_diagnostics` post-scan with schema-driven reductions.

**Directions:** Optional lean rollout mode for training; defer non-plateau metrics to subsampled steps.

**Validate:** Rollout fraction; W&B metric completeness.

**Risk:** Low for optional telemetry strip.

---

### P2 — Duplicate JIT collectors & `deepcopy` per group

**Finding:** Each `JaxRolloutGroup` deep-copies config and compiles separate `collect_fn`.

**Directions:** Shared static args; one collector with `player_count` as static axis if shapes align.

**Validate:** Compile time, memory at init.

**Risk:** Low–medium (shape/static specialization).

## Interview Transcript

| Round | Question | Answer |
|-------|----------|--------|
| 1 | Primary outcome? | Prioritized audit, no code yet |
| 2 | Ranking metric? | Throughput first (`env_steps/sec`, `samples/sec`) |
| 3 | Constraints? | Open; **north star: 4p/mixed ≈ 2p-only throughput** |

**Final ambiguity:** ~18% (Goal 10%, Constraints 25%, Success 15%, Context 25%)

## Acceptance Criteria (Interview Complete)

- [ ] User approves this spec or requests revisions
- [ ] Execution path chosen (ralplan / autopilot / profiling-only follow-up)
- [ ] Audit document expanded with measured A/B gaps when user runs baseline recipe

## Suggested Baseline Recipe (Post-Approval)

```bash
# Smoke compare — disable artifacts for timing clarity
uv run python -m src.train model=attention format=mix_2p_4p_8env \
  training.total_updates=10 training.rollout_steps=32 \
  training.rollout_microbatch_envs=4 \
  artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false \
  telemetry.wandb.enabled=false

uv run python -m src.train model=attention format=2p_16env \
  training.total_updates=10 training.rollout_steps=32 \
  training.rollout_microbatch_envs=4 \
  artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false \
  telemetry.wandb.enabled=false
```

Compare logged `rollout_env_steps_per_sec_2p`, `rollout_env_steps_per_sec_4p`, and total `update_seconds`.
