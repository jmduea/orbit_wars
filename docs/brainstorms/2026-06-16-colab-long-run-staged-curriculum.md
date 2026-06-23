# Colab long-run staged training curriculum (≥1000 updates)

**Date:** 2026-06-16  
**Status:** Proposal — not yet a Hydra profile  
**Audience:** Operators launching Colab / local runs ≥1000 updates after noop-recovery preflight

## Problem statement

Long runs need a curriculum that:

1. Builds **scripted-opponent competence** before self-play dominates.
2. Avoids **policy collapse** (entropy collapse, launch abstention, PPO loss spikes).
3. Leaves headroom for **continued improvement** after scripted beats — not a terminal local optimum.
4. Fits **T4 credit budgets** and known-good Colab geometry from June 2026 pilots.

## Evidence base (repo)

| Source | Finding |
|--------|---------|
| `conf/curriculum/self_play_staged.yaml` | Multi-stage design exists but **starts at 80% latest self-play** (`soft_start`) — caused `policy_loss` spike ~291k at u500 while still in `soft_start` (`docs/benchmarks/issues-jax-validation-500u.md`). |
| `conf/curriculum/production_mix.yaml` | 30% latest self-play — Colab pilots **learning FAIL** (`docs/solutions/workflow-issues/colab-long-run-monitor-sync-recovery.md`). |
| `conf/curriculum/scripted_heavy.yaml` | Static 40/40/20 random/noop/sniper — **known-good** fixed-path pilot; strong noop/random eval by ~u96. |
| Preflight gates | Gates 2–3 require **win-rate delta trend** on noop/random, not absolute `overall_win_rate` (`docs/benchmarks/preflight-calibration.json`). |
| Curriculum controller | Promotion uses rolling mean of `promote_if.metric` after `min_updates` dwell (`src/opponents/curriculum.py`). |
| Collapse guardrails | `approx_kl ≤ 0.15`, `entropy ≥ 1e-4` (calibrated); launch-collapse floors in sweep scoring (`docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md`). |

## Design principles

### 1. Opponent ordering (non-negotiable)

```
noop/random foundations → scripted exploiters → mixed scripted → 4p format → self-play ramp → self-play + scripted anchor
```

Never start long runs with `latest` or `historical` >20% until scripted tournament eval (noop ≥0.7, random ≥0.58) passes on synced checkpoints.

### 2. Metrics for promotion (avoid self-play traps)

| Metric | Use when | Avoid when |
|--------|----------|------------|
| `overall_win_rate` | Stages dominated by noop/random/scripted (<50% self-play) | Self-play-heavy stages (>40% latest+historical) — ~50% is not progress |
| `win_rate_2p` | 2p-only stages | 4p-only stages |
| `first_place_rate_4p` | 4p-only stages | 2p-only stages |
| Manual gate (monitor eval) | Before entering self-play phase | — |

**Operator rule:** read `curriculum_family_prob_*` in JSONL, not `overall_win_rate` alone.

### 3. Stability guardrails (hold between stages)

The curriculum controller supports **one** `promote_if` metric per stage. Stability floors are **operator-enforced** until multi-metric promotion exists.

Between promotions, require a rolling 10-update mean computed from JSONL (or W&B) using **per-update scalars** — long Colab runs must **not** use the `preflight` W&B tag, so `*_window_mean` fields are absent unless planet-flow model or preflight sweep mode is active (`src/jax/train/loop.py`):

| Guardrail | JSONL fields (default telemetry) | Floor |
|-----------|----------------------------------|-------|
| KL stable | mean of last 10 `approx_kl` | ≤ 0.15 |
| Entropy retained | mean of last 10 `entropy` | ≥ 0.0001 |
| No launch collapse | mean of last 10 `mean_active_launches_per_turn` | > 0 |

If guardrails fail: **extend dwell** (do not promote). Consider `training.reseed_every_updates=50` (default) or `100` on long 2p-only phases.

Optional: enable `telemetry.metric_groups.action_decision=true` (default) and post-process JSONL with the same window logic as `src/jax/preflight.py` Gates 2–3.

### 4. Training geometry (single continuous Colab session)

**Problem:** `training=2p_32` has no 4p rollout groups — Phases D–F cannot run in one session without changing geometry mid-run.

**Solution (recommended):** use **`training=2p4p_16_rotate`** for the entire run. Curriculum `format_weights` per stage select which format collects each update (`active_group_indices` in `src/jax/train/rollout_groups.py`):

- Phases A–C: `format_weights: {2: 1.0, 4: 0.0}` → only 2p envs roll (16 envs/update, same memory as `2p_16` rotate).
- Phase D sub-stages: `{4: 1.0, 2: 0.0}` or `{2: 0.5, 4: 0.5}` → 4p or mixed without relaunching.
- Phases E–F: `{2: 0.5, 4: 0.5}` → long-run mix tracks competition formats.

| Phase | Updates (2000u) | Active format (via curriculum) | `training` profile |
|-------|-----------------|--------------------------------|--------------------|
| A–C | 0–900 | 2p only | `2p4p_16_rotate` |
| D | 900–1150 | 4p then mixed | same |
| E–F | 1150–2000 | 50/50 rotate | same |

Set `rollout_microbatch_envs: 16` (matches rotate preset). `task=shield_cheap` unless shortlist selects tiered shield.

**Alternative (two-run, max early throughput):** Run 1 uses `2p_32` for Phases A–C (~900u), sync checkpoint; Run 2 `resume_checkpoint=…` with `training=2p4p_16_split` for Phases D–F. Colab v1 cannot upload local ckpt to a fresh VM — **resume Run 2 locally** or accept two Colab sessions only if remote resume is added later.

**Map pool:** all standard task profiles inherit `map_pool_path: data/jax_map_pool/default_v1.npz` from `conf/task/base.yaml` — training uses pre-baked gather reset, not the legacy stub. `task=map_pool` is an explicit alias for the same default pool, not an on/off switch. To use a different bake, override `task.map_pool_path=` (and verify `map_pool_sha256`). The only path that hits `_reset_legacy` is absent/null `map_pool_path` (test/stub boards).

### 5. Snapshot pool (self-play phases only)

Enable when stage first includes `historical > 0`:

```yaml
snapshot:
  pool_size: 4          # 2000u: 2 is minimal; 4–8 for ≥3000u
  interval_updates: 25  # add snapshot every 25 updates once in self-play
  selection: recent_biased
  fallback: latest
```

Start snapshots only at **Phase E** — earlier snapshots of a weak policy add noise.

## Proposed stage ladder (`curriculum=colab_long_staged` — to implement)

Target: **2000 updates** on T4. Scale `min_updates` proportionally for 1000u (×0.5) or 3000u (×1.5).

### Phase A — noop recovery (updates 1–150)

**Purpose:** Territory expansion; beat passive opponents. Aligns with Gate 2 / preflight shortlist.

```yaml
- id: noop_recovery_2p
  min_updates: 100
  cooldown_updates: 10
  format_weights: {2: 1.0, 4: 0.0}
  opponent_families:
    noop: 0.85
    random: 0.15
    latest: 0.0
    historical: 0.0
    nearest_sniper: 0.0
  promote_if:
    metric: overall_win_rate
    op: ">="
    value: 0.55        # rolling mean; not tournament proof yet
    window_updates: 10
```

**Skip if** resuming from preflight-validated checkpoint that already beats noop (Gate 2 passed locally).

### Phase B — random recovery (updates 150–350)

**Purpose:** Stochastic opponents; Gate 3 alignment.

```yaml
- id: random_recovery_2p
  min_updates: 150
  cooldown_updates: 10
  format_weights: {2: 1.0, 4: 0.0}
  opponent_families:
    random: 0.75
    noop: 0.15
    nearest_sniper: 0.10
    latest: 0.0
  promote_if:
    metric: overall_win_rate
    op: ">="
    value: 0.58
    window_updates: 10
```

### Phase C — scripted pressure (updates 350–900)

**Purpose:** Exploit turtle/noop cheese; learn sniper-shaped threats. Matches successful `scripted_heavy` mix, extended with turtle/opportunistic.

```yaml
- id: scripted_core_2p
  min_updates: 300
  cooldown_updates: 15
  format_weights: {2: 1.0, 4: 0.0}
  opponent_families:
    random: 0.35
    noop: 0.30
    nearest_sniper: 0.20
    turtle: 0.10
    opportunistic: 0.05
    latest: 0.0
  promote_if:
    metric: overall_win_rate
    op: ">="
    value: 0.72
    window_updates: 10
```

**Hard gate (operator / monitor eval):** synced checkpoint must reach tournament noop ≥0.70, random ≥0.58 before Phase D. Do not rely on rollout `overall_win_rate` alone.

### Phase D — 4p bootstrap (updates 900–1150)

**Purpose:** First multi-agent format exposure while opponents remain mostly scripted.

```yaml
- id: scripted_bootstrap_4p
  min_updates: 200
  cooldown_updates: 10
  format_weights: {2: 0.0, 4: 1.0}
  opponent_families:
    random: 0.50
    noop: 0.30
    nearest_sniper: 0.15
    turtle: 0.05
    latest: 0.0
  promote_if:
    metric: first_place_rate_4p
    op: ">="
    value: 0.35
    window_updates: 10
```

Then mixed format scripted bridge:

```yaml
- id: scripted_mixed_formats
  min_updates: 150
  cooldown_updates: 10
  format_weights: {2: 0.5, 4: 0.5}
  opponent_families:
    random: 0.40
    noop: 0.25
    nearest_sniper: 0.20
    turtle: 0.10
    opportunistic: 0.05
    latest: 0.0
  promote_if:
    metric: overall_win_rate
    op: ">="
    value: 0.65
    window_updates: 10
```

No training-profile change at Phase D — `format_weights` switch from `{2:1,4:0}` to `{4:1,2:0}` then `{2:0.5,4:0.5}` on the same `2p4p_16_rotate` run.

### Phase E — self-play ramp (updates 1150–1600)

**Purpose:** Introduce latest + historical without abandoning scripted anchor. **Enable snapshot pool here.**

```yaml
snapshot:
  pool_size: 4
  interval_updates: 25

- id: self_play_ramp
  min_updates: 250
  cooldown_updates: 15
  format_weights: {2: 0.5, 4: 0.5}
  opponent_families:
    latest: 0.25
    historical: 0.15
    random: 0.20
    nearest_sniper: 0.20
    noop: 0.10
    turtle: 0.07
    opportunistic: 0.03
  promote_if:
    metric: win_rate_2p      # NOT overall_win_rate — self-play dilution
    op: ">="
    value: 0.45
    window_updates: 15
```

Ramp sub-stages (optional finer control within Phase E):

| Sub-stage | latest | historical | scripted total |
|-----------|--------|------------|----------------|
| E1 | 0.25 | 0.15 | 0.60 |
| E2 | 0.40 | 0.25 | 0.35 |
| E3 | 0.55 | 0.30 | 0.15 |

Each sub-stage: `min_updates: 80`, promote on `win_rate_2p ≥ 0.48` with entropy guardrail.

### Phase F — self-play refinement (updates 1600–2000, terminal stage)

**Purpose:** Primary self-play with **scripted anchor** to prevent collapse and preserve exploitability for later map-pool / tournament work.

```yaml
- id: self_play_refine
  min_updates: 400          # terminal — no promote_if
  cooldown_updates: 0
  format_weights: {2: 0.5, 4: 0.5}
  opponent_families:
    latest: 0.55
    historical: 0.25
    nearest_sniper: 0.08
    random: 0.05
    turtle: 0.04
    opportunistic: 0.03
    noop: 0.0
  # no promote_if — hold until run end
```

**Why scripted anchor persists:** Pure self-play (`latest_only`) collapses launch activity and produces ~50% win rate with no external improvement signal. Keeping 12–15% scripted + 5% random maintains diversity and tournament-relevant tactics.

## Recommended Hydra bundle (Colab ≥1000u)

Merge **preflight shortlist hyperparams** with fixed geometry:

```bash
uv run ow train colab launch \
  --from-shortlist outputs/colab_runner/shortlist.json --rank 0 \
  --gpu T4 --timeout 86400 \
  --monitor-after-launch \
  --interval-seconds 300 --stale-seconds 900 \
  --eval-baselines noop,random,sniper \
  --eval-seeds 0,1,2,3,4 \
  --eval-formats 2p_vs_baseline \
  training.total_updates=2000 \
  training=2p4p_16_rotate \
  curriculum=colab_long_staged \
  model=transformer_factorized_small \
  model.max_moves_k=3 \
  task=shield_cheap \
  training.rollout_steps=512 \
  training.reseed_every_updates=100 \
  training.ppo_grad_accumulation=true \
  artifacts=disabled \
  artifacts.checkpoint_every=50 \
  output.campaign=colab_long_staged \
  telemetry.wandb.enabled=true \
  telemetry.wandb.group=colab_long_staged
```

No mid-run geometry change required when using `2p4p_16_rotate` — stage promotion alone switches formats.

**W&B:** do **not** tag `preflight` on long runs.

## Update budget scaling

| Total updates | Phase A | B | C | D | E | F |
|---------------|---------|---|---|---|---|---|
| 1000 | 75 | 100 | 250 | 150 | 200 | 225 |
| 2000 | 100 | 150 | 300 | 200+150 | 250 | 400 |
| 3000 | 150 | 200 | 450 | 300 | 400 | 600 |

## Anti-patterns (do not use on long runs)

| Recipe | Why |
|--------|-----|
| `curriculum=production_mix` early | Latest self-play before scripted competence — Colab learning FAIL |
| `self_play_staged` as-is | `soft_start` = 80% latest at u1 — PPO spike documented |
| Promote on `overall_win_rate ≥ 0.90` with self-play mix | Metric saturates / misleads |
| `curriculum=latest_only` for ≥1000u | No scripted anchor → collapse risk, no tournament transfer |
| Static `scripted_heavy` for full 2000u | Works for pilot but **plateaus** — no self-play improvement path |

## Verification checklist

Before declaring a long run successful:

1. **Monitor eval** at u500, u1000, u1500: noop ≥0.70, random ≥0.58 (2p tournament).
2. JSONL: `curriculum_stage_id` progresses A→F; no stall >2× `min_updates`.
3. JSONL: rolling-10 mean of `approx_kl` ≤ 0.15 and `entropy` ≥ 1e-4 at each promotion (compute from per-update scalars — do not expect `approx_kl_window_mean` without preflight tag).
4. JSONL: `mean_active_launches_per_turn` not trending to ~0 across Phase C–F.
5. Phase F checkpoint: tournament vs sniper trending up (not required to “beat sniper” — required to **improve** vs u900 checkpoint).

## Implementation next steps

1. Add `conf/curriculum/colab_long_staged.yaml` from stage ladder above (fix `self_play_staged` soft_start ordering).
2. Gate 4-style test: `ow benchmark gate run curriculum_staged` with new profile at 500u smoke.
3. Colab pilot: 300u through Phase C only before full 2000u commit.
4. Optional: add multi-metric promotion or always-on window-mean logging for long runs (today: single `promote_if` metric + operator guardrails on raw `approx_kl`/`entropy`).

## Related docs

- `docs/colab_runner.md` — launch/monitor commands
- `docs/solutions/workflow-issues/colab-long-run-monitor-sync-recovery.md` — pilot evidence
- `conf/curriculum/self_play_staged.yaml` — prior art (needs reordering)
- `docs/benchmarks/preflight-calibration.json` — stability thresholds
