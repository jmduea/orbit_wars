# Operator runbook

Human verification paths for throughput gates and preflight learning proof. Agents should prefer primitives in `docs/AGENT_CAPABILITIES.md`; this doc is the consolidated operator reference.

## Launch hygiene throughput

| Tier | Command | Proves | GPU |
|------|---------|--------|-----|
| **1** (sampler microbench) | `make test-launch-hygiene-throughput` | Factorized K=5 decoder ≤ 3.22 ms (isolated process) | Optional |
| **2** (production e2e) | `make test-launch-hygiene-e2e-throughput` | Full rollout + PPO path within 10% of pre-hygiene baseline | **Required** |

**Tier-1 pass does not imply tier-2 pass.** Merge hygiene/throughput changes only after tier-2 on the **same GPU machine** used for baseline capture.

### Baseline artifact

- Path: `docs/benchmarks/launch-hygiene-e2e-baseline.json`
- Pre-hygiene SHA: first parent of PR #163 merge (`79162a2088160b8ed05c3e3a050e064c7f6c9556`)
- Pass band: derived from baseline aggregates with `--assert-within-pct 10` (do not invent thresholds)

### Capture (one-time or after metric formula change)

```bash
git worktree add ../orbit_wars-pre-hygiene 79162a2088160b8ed05c3e3a050e064c7f6c9556
cd ../orbit_wars-pre-hygiene && uv sync --group dev
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark training --preset primary --label pre_hygiene_capture \
  --repeats 3 --updates 20 --warmup 2 \
  --out docs/benchmarks/launch-hygiene-e2e-baseline.json
```

Copy the artifact to `main` only after N≥3 runs on the canonical GPU host.

### Gate (before merge on hygiene/throughput paths)

```bash
# Check no other GPU job is active (terminals / ow train)
make test-launch-hygiene-e2e-throughput
```

Equivalent CLI:

```bash
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark training --preset primary --label launch_hygiene_e2e_gate \
  --updates 20 --warmup 2 --out /tmp/launch_hygiene_e2e_gate.json \
  --baseline docs/benchmarks/launch-hygiene-e2e-baseline.json \
  --assert-within-pct 10
```

**Acceptance:** exit code 0; all three metrics (`env_steps_per_sec`, `samples_per_sec`, `seconds_per_update_mean`) within derived band.

If gate fails after hot-path options are exhausted, run the **learner ablation** (A pre-hygiene SHA vs B launch-hygiene `main`) — winner is learn-proof / gate trends, not throughput. See `docs/benchmarks/launch-hygiene-ablation.json` and `docs/solutions/developer-experience/production-training-throughput-profiling.md`. Phase B (U7) is cancelled unless a new rollout sampling design lands.

## Learner ablation (when tier-2 fails)

Compare arms with the same preflight recipe (`transformer_factorized_small`, `--through beat_random`):

| Arm | Checkout | Command |
|-----|----------|---------|
| A (pre-hygiene) | `79162a2088160b8ed05c3e3a050e064c7f6c9556` | `uv run ow benchmark learn-proof --model transformer_factorized_small --through beat_random --out outputs/preflight/ablation_arm_a_pre_hygiene.json` |
| B (launch-hygiene) | `main` | `make preflight-learn-proof` |

Artifact: `docs/benchmarks/launch-hygiene-ablation.json`. Thresholds: `docs/benchmarks/preflight-calibration.json`.

## Preflight learning proof

| Step | Command | Notes |
|------|---------|-------|
| Wiring | `make test-fast` | CPU; safe default |
| Gate 1 | `make preflight-sanity` | Reproducibility |
| Gate 2–3 dry-run | `uv run ow benchmark gate run beat_noop --dry-run` | Verify overrides |
| Gate 2–5 ladder | `make preflight-learn-proof` | GPU; ~minutes per gate |
| Threshold refresh | `make preflight-calibrate` | After calibration campaigns |

**Acceptance (Gates 2–3):** report JSON vs thresholds in `docs/benchmarks/preflight-calibration.json` (synced to `AGENTS.md`). Do not relax thresholds until a run passes under the same recipe.

Primitive sequence behind `make preflight-learn-proof`:

```bash
uv run ow benchmark learn-proof --through beat_random \
  --out outputs/preflight/learn_proof_report.json
```

Gate 5 (tournament win proof) uses `ow benchmark tournament-proof` or full learn-proof with `--eval-checkpoint`.

Full ladder table: `docs/benchmarks/preflight-calibration.md`.

## Environment hygiene

- One GPU: do not run tier-2 e2e, learn-proof, or training smokes in parallel.
- JAX cache for gates: `env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0`
- WSL2 first compile can take several minutes; see `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`

## Related docs

- `AGENTS.md` — agent defaults and threshold excerpts
- `docs/solutions/developer-experience/production-training-throughput-profiling.md` — rollout throughput design and e2e gate context
- `docs/solutions/performance-issues/launch-hygiene-incremental-carry-throughput.md` — tier-1 context
