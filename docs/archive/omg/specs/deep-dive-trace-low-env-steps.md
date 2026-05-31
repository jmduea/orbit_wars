# Deep-Dive Trace: Low env_steps_per_sec

## Problem

`env_steps_per_sec` was observed around 44-60 during a long `gnn_pointer` run, far below prior baseline values around 489 env steps/sec. The goal is to identify root causes and likely optimizations that improve throughput across training configurations.

## Most Likely Explanation

The low rate is primarily real rollout/action-generation cost, not a PPO update bottleneck and not only a reporting artifact. The stopped long run reported roughly 20.71 rollout seconds, 2.37 PPO seconds, and 23.08 update seconds at update 2300, yielding 44.38 env steps/sec and 49.44 rollout env steps/sec. That means rollout dominates the update budget.

The main likely multiplier is the `gnn_pointer` architecture running inside mixed 2p/4p self-play rollout collection:

- `gnn_pointer` uses a 224-wide GNN encoder with KNN graph construction and two message-passing layers.
- It uses an autoregressive pointer decoder with `max_moves_k=3`.
- In 4p rollout, each environment step encodes all four player perspectives and samples or builds actions for non-learner players.
- Mixed format runs separate jitted collectors for 2p and 4p groups, then concatenates transition batches for PPO.
- The long run used `format=mix_2p_4p_8env`, `rollout_steps=64`, `rollout_microbatch_envs=8`, `update_chunk_rows_min=256`, and `update_chunk_rows_max=2048`; that config was inferred from fast Stage 1 results for other models, not measured `gnn_pointer` results.

## Evidence For

### Code Path

- `src/jax_train.py` resolves mixed-format rollout groups and creates one jitted collector per static format. Active groups are collected sequentially in Python per update, then transition batches are concatenated.
- `src/jax_train.py` records `env_steps_per_sec = env_steps / update_seconds` and `rollout_env_steps_per_sec = env_steps / rollout_seconds`; the long run's rollout-only rate was still low, so PPO and logging do not explain the whole issue.
- `src/jax_ppo.py` collects each rollout inside a `jax.lax.scan`, so per-step policy/action/environment work is repeated for every rollout step.
- `src/jax_ppo.py` 4p rollout builds player-specific games, encodes all player turns, applies trajectory shielding, and samples/builds actions for each player before `batched_step_multi_player`.
- `src/jax_policy.py` `gnn_pointer` combines `GNNBackboneEncoder` and `AutoregressivePointerDecoder`, making it materially heavier than `mlp` and plain `attention`.

### Config And Runtime

- `conf/model/gnn_pointer.yaml` uses `hidden_size=224`, `max_moves_k=3`, `gnn_k_neighbors=5`, and `gnn_message_passing_layers=2`.
- `conf/format/mix_2p_4p_8env.yaml` runs two active groups, each with 8 envs; active env parallelism is owned by `format.rollout_groups`, not by `training.num_envs`.
- `conf/sweeps/wandb/baseline_stage1_comfort.yaml` uses a 3-update grid optimized for `samples_per_sec`; it includes first-run JIT/compilation effects and had no completed `gnn_pointer` rows at trace time.
- Prior promoted baseline results show attention/mix_2p_4p_8env around 489 env steps/sec, while heavier sentinel combinations already dropped to 298 env steps/sec. `gnn_pointer` is heavier than those sentinel models.

### Measurement

- `env_steps_per_sec` is per-update, not lifetime, and does not include JSONL append or telemetry logging because those occur after `update_seconds` is computed.
- `samples_per_sec` is learner decision samples per update second, not env steps. It can be much higher because each env step can produce many valid decision rows/sequence decisions.
- Stage 1's 3-update runs include first-run JIT effects, so Stage 1 is not a pure steady-state benchmark. The long run's update 2300 summary is a better steady-state signal.

## Evidence Against Or Limits

- No profiler trace has yet split rollout time into policy forward pass, action shielding, opponent action generation, environment step, reset, and feature encoding.
- No measured `gnn_pointer` Stage 1 rows were available at trace time, so config recommendations are still partly inferred.
- GPU utilization and VRAM snapshots suggested heavy memory pressure, but not enough to distinguish memory bandwidth, compute saturation, or XLA fusion/shape effects.
- `samples_per_sec` and `env_steps_per_sec` answer different questions; neither alone describes all useful throughput.

## Critical Unknowns

1. Within rollout time, which component dominates: GNN encoder, autoregressive decoder, trajectory shield, 4p opponent action generation, environment stepping, reset/feature encoding, or mixed-group sequencing?
2. Does `gnn_pointer` throughput scale with more envs/rollout steps, or does VRAM/compile/update pressure make it plateau or regress?
3. Are 4p rows disproportionately expensive relative to 2p rows, and would separate 2p/4p throughput metrics change config selection?
4. Is `rollout_microbatch_envs=8` optimal for `gnn_pointer`, or would smaller microbatches reduce memory pressure enough to improve wall-clock throughput?
5. Should the optimization target maximize `env_steps_per_sec`, `samples_per_sec`, workstation comfort, or policy quality per wall-clock hour?

## Discriminating Probes

1. Add per-format timing: 2p rollout seconds, 4p rollout seconds, per-format env steps/sec, and per-format samples/sec.
2. Run a short controlled `gnn_pointer` throughput matrix over `format=mix_2p_4p_8env|mix_2p_4p_16env`, `rollout_steps=32|64|128`, and `rollout_microbatch_envs=4|8` with W&B artifacts disabled.
3. Add optional rollout sub-timers or a JAX profiler run to split policy/action generation, shield, env step, reset, and encode costs.
4. Compare `opponents=latest_only curriculum=latest_only` against self-play curriculum to quantify opponent-pool and historical-action overhead.
5. Compare pure 2p and pure 4p rollout configs, or temporarily activate one mixed group at a time, to isolate 4p cost.

## Prioritized Optimization Candidates

1. Instrument per-format rollout timing and sample/env rates first; this is the fastest way to determine whether mixed 2p/4p cost is dominated by 4p.
2. Create a `gnn_pointer`-specific throughput sweep instead of reusing MLP/attention-selected settings.
3. Optimize 4p rollout action generation by avoiding unnecessary per-player policy applications for scripted/noop/random opponents where possible.
4. Reduce `gnn_pointer` model cost variants for throughput testing: smaller hidden size, lower `max_moves_k`, fewer message-passing layers, or smaller neighbor count.
5. Explore rollout group scheduling/parallelism only after profiling; current groups are collected sequentially per update, but parallel collection may increase VRAM pressure.
6. Add clearer throughput reporting: per-format rollout rates, update breakdown percentages, and docs clarifying env steps versus learner samples.
