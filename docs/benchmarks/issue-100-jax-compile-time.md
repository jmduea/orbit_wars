# Issue #100 — JAX compile time vs expected bounds

Research spike closing [#100](https://github.com/jmduea/orbit_wars/issues/100).

## Canonical metric

**`compile_seconds_to_update_3`** — wall-clock seconds from process start until global update **3** completes (inclusive of all JIT compilation through that point).

| Property | Value |
|----------|-------|
| Script | `scripts/issues_jax_30update_benchmark.py` |
| Warmup | 2 updates (compile still counted through update 3) |
| Steady-state timing | `seconds_per_update_mean` over measured updates after warmup |
| Profile | `--preset validation` → `WORKSTATION_VALIDATION_OVERRIDES` |

Why update 3: updates 1–2 are warmup iterations; by update 3 both per-format `collect_fn` jits and the shared `update_fn` jit have executed at least once on the production path. This matches the metric used in all prior workstation benchmark artifacts.

```bash
uv run python scripts/issues_jax_30update_benchmark.py \
  --label issue-100-current-30u \
  --tier workstation \
  --preset validation \
  --updates 30 \
  --out docs/benchmarks/issue-100-current-30u.json
```

## Reference environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5080 (WSL2, CUDA UMD 13.3) |
| JAX | `0.10.0` with `jax[cuda13]` |
| Default profile | `format=2p_4p_16env`, `training=workstation`, `model=transformer_factorized`, `opponents=self_play_only`, `curriculum=off` |
| Parallelism | 32 envs (16×2p + 16×4p), `rollout_steps=128`, `rollout_microbatch_envs=16` |

## Prior runs (reference band)

Compile times from committed JSON under `docs/benchmarks/` on the same hardware class:

| Artifact | Commit | Profile notes | compile→u3 (s) | steady env_steps/s |
|----------|--------|---------------|----------------|-------------------|
| `workstation-30update.json` | `dcafdc8` | `self_play_curriculum` + staged curriculum | **221** | 5826 |
| `validation-500u.json` | `dcafdc8` | validation preset, seed 42 | **310** | 3860 |
| `validation-seed-{44,45,46}-500u.json` | `dcafdc8` | validation preset | **235–237** | ~4700 |
| `validation-seed-43-500u.json` | `dcafdc8` | validation preset (cold-start outlier) | **415** | 949 |
| `validation-{noop,random}_only-*.json` | `dcafdc8` | opponent contrast, 100u | **207–217** | ~2100 |
| `terminal-reward-*.json` | later | reward ablation, same format | **226–236** | — |

**Aggregate (validation preset, self_play_only, seeds 42–46):** mean **287 ± 78 s**; excluding seed-43 outlier **255 ± 35 s** (`validation-seed-sweep-summary.json`).

## Current run (2026-05-30)

Artifact: `issue-100-current-30u.json` · commit `71c3e91`.

| Metric | Current | Prior typical (validation) |
|--------|---------|--------------------------|
| `compile_seconds_to_update_3` | **507 s** | 235–310 s (255 ± 35 excl. outlier) |
| `seconds_per_update_mean` | **0.44 s** | 0.70 s (`workstation-30update`) / ~0.45 s implied at 4.5k env_steps/s |
| `env_steps_per_sec` | **9243** | 3800–5826 (profile-dependent) |

The compile measurement is in the **upper tail** of prior observations (~2× the 255 s central band) but the same order of magnitude as the seed-43 cold-start outlier (415 s). Steady-state throughput after compile is **at or above** historical validation runs, which argues against a functional regression in the compiled graph — variance is dominated by first-run XLA compilation and host/GPU scheduling, not broken training logic.

## Expected bounds (JAX context)

What gets compiled on the production path:

1. One jitted `collect_fn` per rollout group (2p and 4p each own env state + collector).
2. Shared jitted `update_fn` wrapping `ppo_update_jax`.
3. Static shapes from `transformer_factorized`, 128-step rollouts, and opponent/curriculum stage views baked into traces.

For this stack size, **multi-minute first-run compile on GPU is normal** for JAX/XLA. Steady-state update times (~0.4–0.7 s) dominate long runs; compile is a one-time (per process / per trace change) cost.

| Bound | Interpretation |
|-------|----------------|
| **< 120 s** | Unusually fast; suspect cached artifacts or reduced profile |
| **200–350 s** | Typical cold compile on RTX 5080, validation preset |
| **350–600 s** | Elevated but observed (cold start, contention, trace invalidation) — monitor, do not auto-escalate |
| **> 600 s** | Investigate: wrong backend (CPU), profile mismatch, or compile regression |

## Conclusion

**Status: OK — no compile optimization escalation.**

1. **Metric defined:** `compile_seconds_to_update_3` via `issues_jax_30update_benchmark.py` is the canonical compile baseline; it is already emitted in all workstation benchmark JSON.
2. **Within expected bounds:** Current 507 s is high-variance but same order of magnitude as the 207–415 s historical band on identical hardware. Steady-state throughput is healthy.
3. **Action:** Treat compile and steady-state as separate gates. Use `--preset validation` + 30 updates for compile spot-checks before long runs. Re-run once if compile exceeds 400 s before opening an optimization issue.

**Do not escalate** unless compile routinely exceeds **600 s** on a warm, idle workstation or blocks iteration velocity (e.g. frequent config changes during development). Future optional work (not #100 scope): XLA persistent cache, compile-time logging in `run_jax_training`, or a CI smoke that only checks steady-state after a cached warm start.

## Related docs

- `docs/benchmarks/issues-jax-fix-30update.md` — protocol and workstation production profile
- `docs/benchmarks/validation-seed-sweep.md` — multi-seed stability + compile variance
- `scripts/summarize_validation_seed_sweep.py` — aggregate compile stats from JSON artifacts
