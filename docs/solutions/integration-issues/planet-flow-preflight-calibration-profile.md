---
title: Planet Flow preflight calibration broke on missing export and wrong training profile
date: 2026-06-02
last_updated: 2026-06-02
category: integration-issues
module: jax-preflight
problem_type: integration_issue
component: testing_framework
symptoms:
  - "ImportError: cannot import name window_mean_from_metric_rows from src.jax.preflight_calibration"
  - "calibration_train_overrides for planet_flow_target_heatmap used training=2p4p_16_split and training.rollout_steps=128"
  - "Planet Flow shortlist/noop-smoke gates could not score approx_kl and entropy window means from JAX logs"
root_cause: config_error
resolution_type: code_fix
severity: medium
tags:
  - preflight
  - planet-flow
  - calibration
  - hydra
  - window-mean
  - shortlist
related_components:
  - src/jax/preflight_calibration.py
  - src/jax/planet_flow_shortlist.py
  - conf/training/planet_flow.yaml
---

# Planet Flow preflight calibration broke on missing export and wrong training profile

## Problem

After merging Planet Flow proof work with preflight training profiles (PR #173), Planet Flow benchmark and calibration paths failed to import shared metric helpers, and `calibration_train_overrides()` composed the wrong Hydra training profile for `planet_flow_target_heatmap`. Calibration and shortlist tooling then either crashed at import time or trained with rollout geometry that did not match the Planet Flow proof path.

## Symptoms

- `ImportError` when `src/jax/planet_flow_shortlist.py` imports `window_mean_from_metric_rows` from `preflight_calibration` (helper existed only as private `_window_mean`).
- Resolved `ow train` overrides for Planet Flow calibration used `training=2p4p_16_split` plus `training.rollout_steps=128` instead of `training=planet_flow` (512 rollout steps, 2048 chunk rows per `conf/training/planet_flow.yaml`).
- `ow benchmark shortlist-planet-flow-sweep` and related noop-smoke scoring could not compute last-window `approx_kl` / `entropy` aligned with Gates 2–3.

## What Didn't Work

- **Reusing factorized preflight defaults for Planet Flow.** Mapping `planet_flow_target_heatmap` to `2p4p_16_split` and forcing `rollout_steps=128` matched legacy factorized calibration but ignored the dedicated `planet_flow` Hydra group (session history: merge integration on `merge-sim/planet-flow-preflight`).
- **Duplicating window-mean logic in shortlist only.** Keeping `_window_mean` private in `preflight_calibration.py` while shortlist imported a public name caused an immediate import failure once shortlist landed.
- **Extra telemetry override on the proof path.** `telemetry.metric_groups.action_decision=true` was dropped from the Planet Flow calibration tuple because `PREFLIGHT_TRAIN_BASE` already enables it for all calibration arms.

## Solution

Commit `f0ba5b0` (merged via PR #173):

### 1. Export shared window-mean helper

```python
def window_mean_from_metric_rows(
    records: list[dict[str, object]], key: str, *, last_n: int
) -> float | None:
    """Mean of ``key`` over the last ``last_n`` metric rows (shared with preflight gates)."""
    return _window_mean(records, key, last_n=last_n)
```

`planet_flow_shortlist.py` imports this for sweep eligibility scoring consistent with `src/jax/preflight.py` Gates 2–3.

### 2. Use `training=planet_flow` without overriding rollout steps

```python
is_planet_flow = model == "planet_flow_target_heatmap"
training_profile = "planet_flow" if is_planet_flow else "2p_16"
rollout_steps = () if is_planet_flow else ("training.rollout_steps=128",)
return (
    f"model={model}",
    f"training={training_profile}",
    *rollout_steps,
    ...
)
```

Planet Flow arms still append `artifacts=planet_flow_proof` and `artifacts.artifact_pipeline.enabled=true` (with explicit pipeline re-enable per `PREFLIGHT_TRAIN_BASE` — see `docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md`).

### 3. Align factored sampler regression test (merge fallout)

Commit `1eab574` updates `tests/test_factored_step_vmap.py` to match `_sample_factored_step_from_logits` (single `deterministic` flag) so pre-merge JAX tests stay green after sampler signature changes.

## Why This Works

Planet Flow proof and preflight share one calibration entrypoint (`calibration_train_overrides` → `run_ow_train`). Shortlist and gates need the same windowed metric semantics over `logs/*_jax.jsonl`; exporting `window_mean_from_metric_rows` makes that contract explicit. The `planet_flow` Hydra profile owns rollout geometry (`rollout_steps: 512`, `update_chunk_rows: 2048`); omitting a hardcoded `training.rollout_steps=128` override lets composed config match learn-proof and W&B sweep guidance documented in `planet-flow-sweep-gameable-objective.md`.

## Prevention

- When a module imports a helper from `preflight_calibration.py`, export it (or move shared metrics helpers to a neutral module) instead of exposing only `_private` names.
- For model-specific calibration, select the **named training profile** (`training=planet_flow`) rather than a nearby default plus ad hoc overrides.
- After changing sampler signatures, run `uv run pytest tests/test_factored_step_vmap.py tests/test_planet_flow_shortlist.py -q` in the fast tier.
- Verify composed overrides before long GPU sweeps: `uv run ow train print_resolved_config=true model=planet_flow_target_heatmap training=planet_flow opponents=random_only curriculum=off`.

## Related Issues

- [Planet Flow sweep objective and proof artifacts](../logic-errors/planet-flow-sweep-gameable-objective.md) — `training=planet_flow`, `artifacts=planet_flow_proof`, pipeline re-enable pattern, shortlist/noop-smoke operator flow.
- [Benchmark subprocess observability](../developer-experience/benchmark-subprocess-training-observability.md) — `run_ow_train()` streaming used by calibration sweeps.
- [Planet Flow catalog reachability mismatch](../logic-errors/planet-flow-catalog-reachability-mismatch.md) — separate compiler/telemetry failure mode at ~u150.
- Operator flow: `docs/benchmarks/preflight-calibration.md`, `make preflight-calibrate`.
- Merge playbook: [multi-branch agent merge orchestration](../workflow-issues/multi-branch-agent-merge-orchestration.md)
- Merge context: PR [#173](https://github.com/jmduea/orbit_wars/pull/173), `docs/solutions/workflow-issues/multi-branch-agent-merge-orchestration.md`.
