# Feature Encoding v2 Ablation Runbook

**Status:** Informational (non-blocking for Phase 5 cutover)  
**Related:** [feature-encoding-v2.md](./feature-encoding-v2.md), `.omg/plans/ralplan-feature-encoding-v2.md`

This runbook documents matched v1 vs v2 Hydra training commands for ablation comparison. Phase 5 hard cutover proceeds regardless of ablation outcomes; use these runs to capture evidence for future tuning.

## Shared hyperparameters

Keep these identical across v1 and v2 arms unless noted:

- `training.total_updates=2000`
- `training.num_envs=8`
- `training.rollout_steps=32`
- `model.hidden_size=128`
- `model.gnn_k_neighbors=5` (v2 only; v1 uses same GNN preset when applicable)
- `curriculum=self_play_staged`
- `format.rollout_groups` — run **2p-only** and **4p-only** separately before mixed format

## v1 baseline (2p-only)

```bash
uv run python -m src.train \
  model=gnn_pointer \
  task.encoding_version=v1 \
  curriculum=self_play_staged \
  format.rollout_groups='[{name: two_player, player_count: 2, num_envs: 8},{name: four_player, player_count: 4, num_envs: 0}]' \
  training.total_updates=2000 \
  run_name=ablation_v1_2p
```

## v2 candidate (2p-only)

```bash
uv run python -m src.train \
  model=gnn_pointer_v2 \
  task.encoding_version=v2 \
  curriculum=self_play_staged \
  format.rollout_groups='[{name: two_player, player_count: 2, num_envs: 8},{name: four_player, player_count: 4, num_envs: 0}]' \
  training.total_updates=2000 \
  run_name=ablation_v2_2p
```

## v1 baseline (4p-only)

```bash
uv run python -m src.train \
  model=gnn_pointer \
  task.encoding_version=v1 \
  task.player_count=4 \
  curriculum=self_play_staged \
  format.rollout_groups='[{name: two_player, player_count: 2, num_envs: 0},{name: four_player, player_count: 4, num_envs: 8}]' \
  training.total_updates=2000 \
  run_name=ablation_v1_4p
```

## v2 candidate (4p-only)

```bash
uv run python -m src.train \
  model=gnn_pointer_v2 \
  task.encoding_version=v2 \
  task.player_count=4 \
  curriculum=self_play_staged \
  format.rollout_groups='[{name: two_player, player_count: 2, num_envs: 0},{name: four_player, player_count: 4, num_envs: 8}]' \
  training.total_updates=2000 \
  run_name=ablation_v2_4p
```

## Metrics to extract

From local JSONL logs (`outputs/.../logs/*_jax.jsonl`) or W&B when enabled:

| Metric | Notes |
|--------|-------|
| `overall_win_rate` | Primary win-rate comparison |
| `rollout_samples_per_sec_2p` / `_4p` | Throughput by format |
| `trajectory_shield_legal_non_noop_rate` | Shield diagnostic (informational) |
| `curriculum_phase_id` | Stage progression sanity |

## Evidence table template

| Seed | Format | Encoding | Win rate | Rollout SPS | Shield legal non-noop | Notes |
|------|--------|----------|----------|-------------|----------------------|-------|
| 0 | 2p | v1 | | | | |
| 0 | 2p | v2 | | | | |
| 0 | 4p | v1 | | | | |
| 0 | 4p | v2 | | | | |

Recommend ≥3 seeds per cell when time permits. **Does not gate Phase 5 cutover** per interview override.

## M4 intercept-edge ablation (schema v4)

**Prerequisite:** M4 implementation merged (`schema_version=4`, `edge_feature_dim=18`, `task.intercept_anchors=[1.0,6.0]`).

**Baseline pin:** `artifacts/m4/baseline_pin.json` (pre-M4 commit SHA).

Shared anchor (match Phase 0 throughput docs):

```bash
model=gnn_pointer
format=mix_2p_4p_8env
training.rollout_steps=64
training.minibatch_size=256
training.rollout_microbatch_envs=8
training.total_updates=500
task.candidate_count=4
task.intercept_anchors=[1.0,6.0]
```

**Gates (3 paired seeds × 500 updates):**

| ID | Metric | Pass |
|----|--------|------|
| W1 | `episode_reward_mean` (updates 450–500) | ≥ +2% vs pre-M4 baseline, 2p and 4p |
| H2 | `rollout_env_steps_per_sec` median | ≥ 0.95× baseline |
| H1 | Submission validator | zero illegal actions |
| H3 | No NaN/inf losses | stable entropy |

Results doc: [m4-intercept-edge-results.md](./m4-intercept-edge-results.md).

## Cutover recommendation field

After collecting evidence, record observed deltas in the plan appendix. Default recommendation post-Phase 5: **production default is v2** (`conf/task/default.yaml` → `encoding_version: v2`); use ablation numbers to prioritize follow-up tuning, not to block deployment.
