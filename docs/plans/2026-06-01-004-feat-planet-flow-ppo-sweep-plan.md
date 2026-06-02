---
title: "feat: Add Planet Flow PPO signal sweep"
date: 2026-06-01
status: active
type: feat
origin: ".cursor/plans/planet-flow-ppo-sweep_fd39918d.plan.md"
---

# feat: Add Planet Flow PPO Signal Sweep

## Summary

Add a config-only W&B sweep for tuning PPO parameters specifically for the Planet Flow decoder. The sweep uses the existing factorized PPO stability sweep as a frame of reference, but optimizes `overall_win_rate` instead of KL because Planet Flow has already shown stable late-run KL while failing the learning-trend proof.

The first pass intentionally avoids adding a new telemetry objective. If optimizing `overall_win_rate` is too noisy, a follow-up can add a dedicated Planet Flow sweep score.

---

## Problem Frame

Planet Flow passed the initial throughput proof on `training=2p4p_16_split`, but the `beat_noop` learn-proof run returned `NOT_VERIFIED` because the last-window win-rate trend was negative. Late-run PPO stability metrics were small, so a sweep that minimizes `approx_kl_v2` alone is unlikely to select for the missing behavior.

The tuning surface should therefore search PPO knobs while keeping the objective aligned with learning progress and preserving guardrail metrics for KL, entropy, throughput, and compiler-control attribution.

---

## Requirements

- Add a W&B sweep recipe for `model=planet_flow_target_heatmap`.
- Keep the sweep config-only: no new training-loop metric or custom W&B objective in this pass.
- Optimize an existing logged metric, `overall_win_rate`, to prioritize proof-surface behavior.
- Keep Planet Flow guardrails visible in W&B: action-decision telemetry, compiler-control deltas, PPO KL, entropy, and throughput.
- Keep artifacts disabled and curriculum off for P0 proof/tuning runs.
- Verify the new sweep composes and generates a W&B sweep YAML.

---

## Key Technical Decisions

- **Use `overall_win_rate` as the W&B objective.** It is already logged, registered as a known sweep objective, and directly matches the failed learn-proof surface.
- **Keep KL as a guardrail, not the optimizer.** Current Planet Flow logs show low late-run `approx_kl_v2`, so KL stability alone is not the bottleneck.
- **Use 2-player noop tuning first.** The failed gate is `beat_noop`, and the fastest signal loop is `training=2p_16` with `opponents=noop_only`.
- **Lower entropy pressure relative to factorized tuning.** Planet Flow has one categorical demand head per active planet, and observed entropy stayed high, so the sweep should explore lower `training.ent_coef` values than the factorized stability sweep.
- **Preserve compiler-control telemetry.** `telemetry.metric_groups.action_decision=true` is required so Planet Flow learned-vs-control metrics remain visible during candidate review.

---

## Implementation Units

### U1. Add Planet Flow Sweep Recipe

**Goal:** Add the top-level W&B sweep recipe that composes the method, metric, fixed axes, and search space for Planet Flow PPO tuning.

**Requirements:** Adds the sweep entry point and uses `overall_win_rate` as the objective.

**Dependencies:** None.

**Files:**
- Create: `conf/wandb_sweep/planet_flow_ppo_signal.yaml`
- Test: `tests/test_config_consolidation.py`

**Approach:** Mirror the structure of `conf/wandb_sweep/ppo_stability_kl.yaml`, but select `metric: overall_win_rate`, `fixed: planet_flow_ppo_signal`, and `space: planet_flow_ppo_signal`. Keep the initial run cap modest so the sweep is cheap enough to run after the branch lands.

**Patterns to follow:** `conf/wandb_sweep/ppo_stability_kl.yaml`.

**Test scenarios:**
- Compose the sweep recipe through the existing sweep YAML smoke test.
- Generate the sweep YAML and confirm the metric is `overall_win_rate`.

**Verification:** The new sweep recipe is discoverable by the existing W&B sweep compose tests.

### U2. Add Fixed Planet Flow PPO Profile

**Goal:** Define fixed axes for short Planet Flow PPO tuning runs.

**Requirements:** Uses Planet Flow, action-decision telemetry, disabled artifacts, disabled curriculum, and a proof-like 2-player noop profile.

**Dependencies:** U1.

**Files:**
- Create: `conf/wandb_sweep/fixed/planet_flow_ppo_signal.yaml`
- Test: `tests/test_config_consolidation.py`

**Approach:** Mirror `conf/wandb_sweep/fixed/ppo_stability_kl.yaml` while changing the model/profile to Planet Flow P0:
- `model=planet_flow_target_heatmap`
- `training=2p_16`
- `training.total_updates=300`
- `opponents=noop_only`
- `curriculum=off`
- `artifacts=disabled`
- `telemetry.metric_groups.action_decision=true`
- W&B enabled with a Planet Flow-specific group/tag set.

**Patterns to follow:** `conf/wandb_sweep/fixed/ppo_stability_kl.yaml`; Planet Flow P0 overrides in `src/jax/training_benchmark.py`.

**Test scenarios:**
- Compose sampled sweep configs and verify runtime guards accept artifacts/curriculum settings.
- Confirm action-decision telemetry is enabled in the composed fixed profile.

**Verification:** The fixed sweep block composes without enabling unsupported Planet Flow artifact/eval paths.

### U3. Add Planet Flow PPO Search Space

**Goal:** Define the PPO hyperparameter search space for Planet Flow.

**Requirements:** Search PPO knobs likely to affect Planet Flow learning while keeping throughput and stability guardrails reviewable.

**Dependencies:** U1, U2.

**Files:**
- Create: `conf/wandb_sweep/space/planet_flow_ppo_signal.yaml`
- Test: `tests/test_config_consolidation.py`

**Approach:** Start from `conf/wandb_sweep/space/ppo_stability_kl.yaml` and adjust ranges for Planet Flow:
- lower `training.lr` floor to include more conservative updates
- lower `training.ent_coef` range to reduce persistent high-entropy pressure fields
- keep `training.clip_coef`, `training.epochs`, `training.vf_coef`, and `training.max_grad_norm`
- include a small set of `training.update_chunk_rows` values if composition supports it, to test PPO minibatch pressure without changing rollout shape.

**Patterns to follow:** `conf/wandb_sweep/space/ppo_stability_kl.yaml`.

**Test scenarios:**
- Compose bounded sweep samples from the new search space.
- Confirm sampled hyperparameters stay in valid ranges accepted by the training config.

**Verification:** Generated sweep YAML contains the Planet Flow PPO hyperparameter ranges and no unsupported artifact/curriculum overrides.

### U4. Verify Generated Sweep

**Goal:** Prove the new sweep can be generated and checked by the existing config test harness.

**Requirements:** Generated YAML composes and can be registered with W&B after manual review.

**Dependencies:** U1, U2, U3.

**Files:**
- Test: `tests/test_config_consolidation.py`
- Test: `tests/test_metric_registry.py`

**Approach:** Run targeted config tests and generate the sweep YAML through the repo’s sweep generator. Do not launch W&B agents as part of this implementation.

**Patterns to follow:** Existing sweep verification in `tests/test_config_consolidation.py`.

**Test scenarios:**
- The sweep YAML smoke compose test includes the new recipe.
- The known sweep metric registry test recognizes `overall_win_rate`.
- `uv run ow make wandb_sweep=planet_flow_ppo_signal` writes a generated sweep file.

**Verification:** Targeted config tests pass and the generated sweep artifact is present under `outputs/_meta/sweeps/`.

---

## Scope Boundaries

### In Scope

- W&B sweep config files for Planet Flow PPO tuning.
- Config and metric-registry verification.
- Comments documenting objective and manual guardrails.

### Out of Scope

- Running W&B agents.
- Adding a custom Planet Flow sweep objective metric.
- Changing PPO implementation behavior.
- Changing Planet Flow policy or compiler behavior.
- Updating persistent calibration thresholds from this sweep.

### Deferred to Follow-Up Work

- Add a dedicated `planet_flow_sweep_score` telemetry metric if `overall_win_rate` is too noisy.
- Run top sweep candidates through `ow benchmark learn-proof` with exact selected overrides.

---

## Risks And Guardrails

- **Noisy objective:** `overall_win_rate` can be volatile over short runs. Keep candidate review anchored by last-window trend, entropy, and compiler-control deltas.
- **Overfitting to noop:** A noop-first sweep may find parameters that do not transfer to random or mixed-format settings. Confirm winners on `beat_random` before promoting.
- **High entropy plateau:** Planet Flow may continue to behave near random pressure fields if `ent_coef` remains too high. Include lower entropy coefficients and review entropy trends.
- **Throughput regression:** PPO settings should not change rollout compiler throughput, but update settings can affect wall-clock. Keep `samples_per_sec` and `env_steps_per_sec` visible.

---

## Operational Notes

After implementation, the intended manual flow is:

1. Generate the sweep YAML.
2. Register it with W&B.
3. Run W&B agents on the current GPU host.
4. Select top candidates by `overall_win_rate` subject to KL, entropy, throughput, and compiler-control guardrails.
5. Re-run `learn-proof` for the selected overrides.

---

## Acceptance Criteria

- `conf/wandb_sweep/planet_flow_ppo_signal.yaml` exists and composes.
- `conf/wandb_sweep/fixed/planet_flow_ppo_signal.yaml` fixes Planet Flow P0-safe axes.
- `conf/wandb_sweep/space/planet_flow_ppo_signal.yaml` defines the PPO search space.
- Targeted config/metric tests pass.
- The sweep generator emits `outputs/_meta/sweeps/planet_flow_ppo_signal.yaml`.
