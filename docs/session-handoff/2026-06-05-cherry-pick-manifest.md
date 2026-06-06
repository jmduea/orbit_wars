# Session handoff: nuclear cherry-pick manifest (2026-06-05)

Handoff for continuing Orbit Wars cherry-pick manifest work.

## Decision locked this session

**Learning proof comes first; one training recipe gates both checks.**

The prior assistant recommendation used **two different training setups** (self-play for speed, noop for learning). **You rejected that.** Both gates must use the **same core training config**.

**New speed check:** After the 200-update learning run finishes, read training speed from the same log file — updates 3 through 20 only (skip the first two updates while JAX compiles). No second GPU job for a short speed-only run during cherry-pick admission.

The old speed baseline (`launch-hygiene-e2e-baseline.json`, self-play / full model) is **superseded**. Capture a **new baseline** from the locked admission recipe below.

## Operator-locked admission recipe (2026-06-05)

Locked from picker preset **Beat noop learning proof (core recipe)** with these edits:

| Setting | Default beat_noop | Locked |
|---------|-------------------|--------|
| Game formats | 2-player only | 50% 2-player, 50% 4-player |
| Parallel envs | 16 | 32 |
| Rollout steps | 128 | 256 |
| Planet candidates | 6 | 3 |
| Weights & Biases | off | on (`group=preflight`) |
| Replay artifacts | on (default) | off |

**Microbatch note:** The picker diff showed `rollout_microbatch_envs=32`. With mixed format and 32 total envs, each format group gets 16 envs, so the resolved microbatch is **16** (32 would fail validation).

### Full Hydra overrides (train / verify)

Append PPO lines when not using the gate (gate adds them automatically):

```text
model=transformer_factorized_small
task=shield_cheap
training=2p4p_32_split
training.rollout_steps=256
task.candidate_count=3
opponents=noop_only
curriculum=off
telemetry.wandb.enabled=true
telemetry.wandb.group=preflight
artifacts.artifact_pipeline.enabled=false
artifacts.replay.enabled=false
telemetry.metric_groups.action_decision=true
seed=42
training.log_every=1
training.lr=0.0003
training.clip_coef=0.2
training.ent_coef=0.005
training.vf_coef=0.5
training.max_grad_norm=0.5
training.epochs=2
training.update_chunk_rows=1024
training.reseed_every_updates=0
```

Verify (expect 32 envs, 256 steps, format_weights 0.5/0.5, candidate_count 3, wandb on):

```bash
cd /home/jmduea/projects/orbit_wars
uv run ow train print_resolved_config=true \
  model=transformer_factorized_small task=shield_cheap training=2p4p_32_split \
  training.rollout_steps=256 task.candidate_count=3 opponents=noop_only curriculum=off \
  telemetry.wandb.enabled=true telemetry.wandb.group=preflight \
  artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false \
  telemetry.metric_groups.action_decision=true seed=42 training.log_every=1 \
  training.lr=0.0003 training.clip_coef=0.2 training.ent_coef=0.005 \
  training.vf_coef=0.5 training.max_grad_norm=0.5 training.epochs=2 \
  training.update_chunk_rows=1024 training.reseed_every_updates=0
```

Picker preset `admission_locked` in `scripts/build_config_frozen_defaults_picker.py` matches this bundle (regenerate HTML after pulling).

### Threshold / baseline warnings

- **Learning gate (`beat_noop`):** Floors in `preflight-calibration.json` were measured on **2-player-only**, 16 envs, 128 steps, 6 candidates. Mixed format and longer rollouts change the learning signal — use existing thresholds **provisionally**; run `ow benchmark calibrate` if results are borderline or failures look like geometry mismatch, not regressions.
- **Throughput baseline:** Required on this recipe before ±10% gating. ~4× more env-steps per update than the old 2p_16×128 proposal; do not compare to `launch-hygiene-e2e-baseline.json` or uncaptured 2p-only baselines.

## Step-by-step commands (operator)

### 1. Verify config (already locked — re-run if conf/ changed)

```bash
cd /home/jmduea/projects/orbit_wars
uv run python scripts/build_config_frozen_defaults_picker.py   # optional: refresh HTML + admission_locked preset
# verify command above
```

### 2. Capture speed baseline (GPU — one-time, check terminals first)

Gate dry-run shows full override list including PPO. **Do not pipe to tail.**

```bash
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark training \
  --overrides model=transformer_factorized_small task=shield_cheap training=2p4p_32_split \
    training.rollout_steps=256 task.candidate_count=3 opponents=noop_only curriculum=off \
    telemetry.wandb.enabled=true telemetry.wandb.group=preflight \
    artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false \
    telemetry.metric_groups.action_decision=true seed=42 training.log_every=1 \
    training.lr=0.0003 training.clip_coef=0.2 training.ent_coef=0.005 training.vf_coef=0.5 \
    training.max_grad_norm=0.5 training.epochs=2 training.update_chunk_rows=1024 \
    training.reseed_every_updates=0 \
  --label learning_first_capture \
  --updates 20 --warmup 2 --repeats 3 --detailed-timing \
  --out docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json
```

After approval, point `Makefile` `test-launch-hygiene-e2e-throughput` at this JSON.

### 3. Learning proof + speed extract (one GPU run)

On the throughput comparison branch, append locked deltas after gate defaults:

```bash
cd /home/jmduea/projects/orbit_wars-throughput-anchor
uv run ow benchmark gate run beat_noop --dry-run --verbose \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3 \
  telemetry.wandb.enabled=true telemetry.wandb.group=preflight artifacts.replay.enabled=false

uv run ow benchmark gate run beat_noop \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3 \
  telemetry.wandb.enabled=true telemetry.wandb.group=preflight artifacts.replay.enabled=false \
  --out outputs/benchmarks/cherry-pick/anchor_learn_proof.json
```

Then extract speed from the same run:

```bash
uv run ow benchmark admission-throughput \
  outputs/benchmarks/cherry-pick/anchor_learn_proof.json \
  --baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json \
  --assert-within-pct 10
```

**One-shot alternative:**

```bash
uv run ow benchmark gate run beat_noop \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3 \
  telemetry.wandb.enabled=true telemetry.wandb.group=preflight artifacts.replay.enabled=false \
  --out outputs/benchmarks/cherry-pick/anchor_learn_proof.json \
  --also-throughput \
  --throughput-baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json \
  --throughput-within-pct 10
```

### 4. Kaggle parity (before env-parity cherry-picks)

```bash
make test-kaggle-parity
```

### 5. Then env-parity cherry-picks

Only after unified speed baseline captured, learning proof VERIFIED (or recalibrated), throughput within ±10% of baseline, and parity green — cherry-pick env-parity commits onto `throughput-baseline-integration`.

## Git state (unchanged infrastructure)

| Location | Branch | Notes |
|----------|--------|-------|
| Main repo | `main` | Manifest + handoff live here |
| Throughput comparison worktree | `throughput-baseline` @ `52dfdb0` | `/home/jmduea/projects/orbit_wars-throughput-anchor` |
| Integration branch | `throughput-baseline-integration` @ pre-hygiene anchor | Env-parity integration target |

## Blockers

1. **GPU time** — baseline capture (3×20 updates) + one learning proof (200 updates) + parity tests; one heavy GPU job at a time.
2. **Threshold recalibration** — mixed-format learning proof may need `ow benchmark calibrate` before hard gating.
3. **Makefile / manifest wiring** — after baseline JSON lands, update `test-launch-hygiene-e2e-throughput` to use locked overrides + new baseline path.

## Key paths

| Artifact | Path |
|----------|------|
| Manifest | `docs/benchmarks/cherry-pick-manifest.json` |
| Config picker | `docs/tools/config-frozen-defaults-picker.html` |
| Old speed baseline (superseded framing) | `docs/benchmarks/launch-hygiene-e2e-baseline.json` |
| New baseline (pending capture) | `docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json` |
| Learning thresholds | `docs/benchmarks/preflight-calibration.json` |
| Throughput extract CLI | `uv run ow benchmark admission-throughput` |

## Next session starter

```
Continue cherry-pick manifest from docs/session-handoff/2026-06-05-cherry-pick-manifest.md.
Admission recipe locked 2026-06-05 (mixed 2p/4p, 32×256, candidate_count=3, wandb preflight).
Run: capture launch-hygiene-e2e-baseline-learning-first.json → beat_noop --train-overrides → admission-throughput → parity → env-parity cherry-picks.
```
