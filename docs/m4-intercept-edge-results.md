# M4 Intercept Edge Features — Ablation Results

**Milestone:** `intercept-edge-features` (schema v4, E=18)  
**Status:** Implementation complete; **reward gate pending user-approved training runs**

## Implementation summary

- Edge geometry replaced with two-anchor intercept features (`intercept_anchors: [1.0, 6.0]`).
- `crosses_now` retained as legality-aligned snapshot sun-crossing; per-anchor `sun_cross_at_intercept_*` is predictive.
- Dynamic trajectory shield unchanged (thin-shield work deferred to `thin-trajectory-shield`).
- CPU verification: `make test-domain-features` (16/16), `make test-domain-artifacts` (36/36).

## Gate table

| ID | Metric | Target | Baseline | M4 | Pass? |
|----|--------|--------|----------|-----|-------|
| W1 | `episode_reward_mean` (updates 450–500), 2p | ≥ +2% vs baseline | — | — | pending |
| W1 | `episode_reward_mean` (updates 450–500), 4p | ≥ +2% vs baseline | — | — | pending |
| H1 | Submission validator (100 games × 2p/4p) | zero rejections | — | — | pending |
| H2 | `env_steps_per_sec` median | ≥ 0.95× baseline | — | — | pending |
| H3 | Training stability | no NaN/inf, entropy floor | — | — | pending |

## Variance characterization (pre-A/B)

Run before the full paired A/B to decide 3 vs 6 seeds:

```bash
for seed in 11 13 17; do
  uv run python -m src.train \
    model=gnn_pointer format=mix_2p_4p_8env \
    training.total_updates=100 training.rollout_steps=64 \
    training.minibatch_size=256 training.rollout_microbatch_envs=8 \
    seed=$seed telemetry.wandb.enabled=false \
    artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false
done
```

Record σ of `episode_reward_mean` at update 100 → `artifacts/m4/variance_characterization.json`.

## Paired A/B commands (500 updates × 3 seeds)

**Requires user approval** (~3 hours wall-clock on WSL2/CUDA host).

Shared anchor:

```bash
format=mix_2p_4p_8env
training.rollout_steps=64
training.minibatch_size=256
training.rollout_microbatch_envs=8
training.total_updates=500
task.candidate_count=4
task.intercept_anchors=[1.0,6.0]
```

Baseline arm (schema v3 checkpoint invalid — retrain from scratch on pinned pre-M4 commit or current main without intercept features is not possible once merged; both arms retrain on current code, baseline uses pre-merge commit checkout for fair comparison):

```bash
# Baseline: checkout pre-M4 commit, retrain 3 seeds
git checkout e4bb3735a809327cef628d53bf830791ccee6fd1
uv run python -m src.train model=gnn_pointer format=mix_2p_4p_8env \
  training.total_updates=500 ... seed=101
```

M4 arm (current main):

```bash
uv run python -m src.train model=gnn_pointer format=mix_2p_4p_8env \
  training.total_updates=500 task.intercept_anchors=[1.0,6.0] ... seed=101
```

Repeat seeds `101`, `202`, `303` for both arms. Run 2p-only and 4p-only format groups per plan if mixed format confounds win-rate.

## Throughput benchmark

```bash
uv run python scripts/benchmark_jax_rl.py \
  --overrides model=gnn_pointer format=mix_2p_4p_8env \
  training.rollout_steps=64 training.minibatch_size=256 \
  training.rollout_microbatch_envs=8 \
  --warmup 2 --updates 20
```

Run ×3 reps on baseline commit and M4 commit; compare median `env_steps_per_sec`.
