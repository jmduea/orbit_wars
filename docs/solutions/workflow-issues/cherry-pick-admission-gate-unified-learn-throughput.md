---
title: Cherry-pick admission gate — unified learning + throughput on one recipe
date: 2026-06-06
category: workflow-issues
module: benchmark
problem_type: workflow_issue
component: tooling
severity: high
applies_when:
  - "Blocking cherry-pick admission with make gate-admission or ow benchmark gate run admission"
  - "Learning proof and throughput must share one training run on the locked admission recipe"
  - "Gate stderr shows misleading raw Hydra overrides or suspicious approx_kl vs calibrated floors"
  - "Cherry-pick trials use --repo-root so training conf resolves in the worktree, not main"
symptoms:
  - "approx_kl far above 0.15 on beat_noop despite healthy win-rate trend"
  - "Gate dry-run lists training.lr=0.0003 and clip_coef=0.2 from preflight-profiles.json instead of conf/training/base.yaml"
  - "Edits to conf/training/base.yaml on main have no effect on anchor gate runs"
  - "Raw override dump shows both training=2p_16 and training=2p4p_32_split in one line"
root_cause: config_error
resolution_type: config_change
tags:
  - admission-gate
  - beat-noop
  - apply-ppo-profile
  - preflight-profiles
  - approx-kl
  - throughput-extract
  - resolved-config
  - cherry-pick-admission
  - repo-root
related_components:
  - conf/benchmark/gates/admission.yaml
  - conf/benchmark/gates/beat_noop.yaml
  - src/cli/benchmark_gates.py
  - src/jax/preflight_gate_loader.py
  - src/jax/preflight_config_summary.py
  - docs/benchmarks/cherry-pick-manifest.json
---

# Cherry-pick admission gate — unified learning + throughput on one recipe

## Context

Cherry-pick admission must prove **learning signal** and **throughput** on the **same** operator-locked training recipe (mixed 2p/4p, 32 envs, 256 rollout steps, noop opponents). The throughput-anchor worktree holds candidate training code; `main` holds the gate harness, thresholds, and combined `admission` gate.

Early runs failed with `approx_kl` ~2.34 and borderline win-rate delta while throughput passed — traced to stale PPO pins from `preflight-profiles.json`, not geometry. A separate `beat_noop` run plus post-hoc `admission-throughput` was easy to mis-run; operators needed one gate, readable resolved config, and clear worktree conf boundaries.

## Guidance

### Unified admission gate (preferred)

One command runs learning proof (`beat_noop`) then extracts throughput from the same JSONL (updates 3–20) against `docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json` (±10%).

```bash
cd ~/projects/orbit_wars

make gate-admission \
  REPO_ROOT=~/projects/orbit_wars-throughput-anchor \
  ADMISSION_OUT=~/projects/orbit_wars-throughput-anchor/outputs/benchmarks/admission/gate.json
```

CLI equivalent:

```bash
uv run ow benchmark gate run admission \
  --repo-root ~/projects/orbit_wars-throughput-anchor \
  --output-root ~/projects/orbit_wars-throughput-anchor/outputs \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3 \
    telemetry.wandb.enabled=true telemetry.wandb.group=preflight artifacts.replay.enabled=false \
  --out ~/projects/orbit_wars-throughput-anchor/outputs/benchmarks/admission/gate.json
```

**JSON fields:** `admission_passed` (both must pass), `verdict` (learning), `throughput_verdict`, `throughput` extract block.

**Multi-seed sweep:** `make sweep-ppo-admission REPO_ROOT=~/projects/orbit_wars-throughput-anchor`

Legacy path `beat_noop` + `--also-throughput` still works; prefer `admission`.

### Worktree harness (`--repo-root`)

| Piece | Checkout |
|-------|----------|
| `ow benchmark gate run`, gate YAML, thresholds | **main** |
| `ow train` / Hydra `conf/` (PPO, model, training groups) | **worktree** |
| Gate artifacts under `outputs/campaigns/preflight_*` | **worktree** |

Do **not** run `ow benchmark gate` inside a worktree that predates the gate subcommand.

**Hydra fix:** pass `output.root=outputs` (repo-relative) when `--repo-root` is set — absolute `output.root` fails `_orbit_safe_rel`. See [worktree-preflight-gate-repo-root.md](worktree-preflight-gate-repo-root.md) for the full `--repo-root` template.

### PPO source of truth (KL stability)

`beat_noop` / `beat_random` previously set `apply_ppo_profile: true`, appending PPO overrides from `docs/benchmarks/preflight-profiles.json` **after** gate envelope overrides — overriding `conf/training/base.yaml`:

| Parameter | Stale profile pin | Production base.yaml (example) |
|-----------|-------------------|----------------------------------|
| lr | 0.0003 | 0.00006 |
| clip_coef | 0.2 | 0.15 |
| epochs | 2 | 1 |
| reseed_every_updates | 0 | 50 |

**Fix:** remove `apply_ppo_profile` from gate YAML so gate runs use Hydra-composed `training/base.yaml` unless you pass explicit PPO in `--train-overrides`.

**Worktree rule:** with `--repo-root`, only the **worktree** `conf/training/base.yaml` matters. Main-only edits do not affect gate training.

**Calibration note:** `ow benchmark calibrate` still uses profile PPO pins by design (thresholds measured under that bundle). Gate admission uses production `base.yaml`.

### Resolved config on stderr

Gate progress used to print the full raw override list (gate defaults + CLI overrides), making it look like old geometry (`training=2p_16`) was active when later overrides won.

**Now:** stderr prints a **Resolved gate training config** block (geometry + PPO) before train; use `--verbose` for the full override list.

Dry-run check:

```bash
uv run ow benchmark gate run admission --dry-run --verbose \
  --repo-root ~/projects/orbit_wars-throughput-anchor \
  --output-root ~/projects/orbit_wars-throughput-anchor/outputs \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3
```

### Threshold geometry caveat

`beat_noop` floors in `preflight-calibration.json` were calibrated on 2p-only / 16 env / 128 steps. The locked admission recipe is mixed 2p/4p / 32 / 256. Borderline win-rate delta with high KL may be geometry mismatch **or** PPO mismatch — fix PPO source first, then recalibrate if still borderline.

## Why This Matters

Split gates and profile PPO pins produced false KL failures and wasted GPU time. Unified `admission` + worktree-aware conf + resolved config display give one auditable artifact (`admission_passed`) before Phase 2 env-parity cherry-picks.

## When to Apply

- Nuclear cherry-pick manifest: anchor admission on `throughput-baseline` worktree.
- Re-gating after PPO or training conf changes in the worktree.
- Any trial where main harness must gate worktree training code.

## Examples

**Inspect result:**

```bash
jq '{admission_passed, verdict, throughput_verdict, stage: .stage.verdict}' \
  ~/projects/orbit_wars-throughput-anchor/outputs/benchmarks/admission/gate.json
```

**Phase 2 after admission passes:** granular file/hunk env-parity picks with fast gates (`make test-kaggle-parity` + short validation smoke) — not tier-2 e2e per hunk. See [nuclear-cherry-pick-manifest-baseline-integration.md](nuclear-cherry-pick-manifest-baseline-integration.md).

## Related

- [worktree-preflight-gate-repo-root.md](worktree-preflight-gate-repo-root.md) — `--repo-root` and `output.root` details
- [nuclear-cherry-pick-manifest-baseline-integration.md](nuclear-cherry-pick-manifest-baseline-integration.md) — Phase 2/3 strategy
- [planet-flow-preflight-calibration-profile.md](../integration-issues/planet-flow-preflight-calibration-profile.md) — profile override anti-pattern (model-specific)
- `docs/benchmarks/preflight-calibration.md` — calibration vs production PPO split
- `docs/benchmarks/cherry-pick-manifest.json` — `admission_profile`, `baseline_gates`
