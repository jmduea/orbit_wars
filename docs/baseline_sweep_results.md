# Workstation-Friendly Baseline Sweep Results

This note records the first promoted active-workstation baseline from the May 2026 W&B sweep pass. The result is intended as a practical default comparison point for future Orbit Wars sweeps, not as a long-horizon training claim.

## Selected Baseline

Hydra overrides:

- `model=attention`
- `format=mix_2p_4p_8env`
- `training.rollout_steps=32`
- `training.minibatch_size=256`
- `training.rollout_microbatch_envs=4`
- `training.update_chunk_rows_min=1024`
- `training.update_chunk_rows_max=2048`
- `training.lr=0.0003`
- `training.ent_coef=0.005`
- `artifacts.checkpoint_every=1000`
- `artifacts.artifact_pipeline.enabled=false`
- `artifacts.replay.enabled=false`

Decision: promote this as the default workstation-friendly comparison baseline for near-term sweeps.

Reason: it was the fastest Stage 1 finalist under the comfort gate, completed a 3-seed Stage 2 validation with stable timing, and passed bounded sentinel checks without runtime failures. The sentinel checks show that larger candidate counts and the entity-transformer model are heavier, so future sweeps on those axes should treat throughput changes as expected interaction effects.

## Stage 1 Comfort Screen

W&B sweep: `upsuqs23`

All four Stage 1 runs finished. The user reported each run was smooth enough for foreground use, including video playback on another monitor.

| Run | Rollout steps | Minibatch | Samples/sec | Env steps/sec | Update sec | Comfort |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dr80ezqp` | 16 | 128 | 2495.78 | 498.77 | 0.513 | Pass |
| `nqbvbf9x` | 32 | 128 | 4837.81 | 477.44 | 1.072 | Pass |
| `flcwxalw` | 16 | 256 | 2252.42 | 468.03 | 0.547 | Pass |
| `w66kvvi2` | 32 | 256 | 5002.18 | 486.53 | 1.052 | Pass |

Stage 1 was a throughput and comfort screen only. These 3-update jobs produced zero completed episodes, so they were not used for performance conclusions.

Promoted Stage 1 finalist: `w66kvvi2`, because it had the best Stage 1 `samples_per_sec` while remaining comfortable.

The earlier abandoned sweep `vs6pqof5` is ignored because its first template was too large for the intended active-workstation screen.

## Stage 2 Seed Validation

W&B sweep: `il27mv5r`

Fixed config: selected Stage 1 finalist with `training.total_updates=25` and seeds `101`, `202`, and `303`.

| Run | Seed | Overall win rate | Avg 4p placement | Episodes | Samples/sec | Env steps/sec | Approx KL | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `x3vf51ng` | 101 | 0.00 | 3.000 | 29 | 5995.92 | 489.39 | 0.0159 | 2.333 |
| `x1pnc588` | 202 | 0.00 | 3.500 | 31 | 5326.43 | 496.02 | 0.0068 | 2.222 |
| `1bs41m0w` | 303 | 0.25 | 2.833 | 32 | 5054.82 | 486.02 | 0.0527 | 2.177 |

Medians:

- `samples_per_sec`: 5326.43
- `env_steps_per_sec`: 489.39
- `completed_episodes`: 31

Policy-health note: no seed collapsed during this bounded validation. Seed `303` had the highest `approx_kl`, but the run completed normally and entropy stayed in the same broad range as the other seeds.

Performance caveat: this 25-update validation is intentionally short. It confirms that the baseline is viable and comparable under the active-workstation budget; it does not prove long-horizon policy quality.

## Sentinel Checks

W&B sweep: `7vahrhf8`

All four sentinel runs finished. These are warning checks, not a full interaction matrix.

| Run | Model | Candidate count | Episodes | Samples/sec | Env steps/sec | Rollout sec | PPO sec | Update sec | Approx KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dao45oba` | `attention` | 8 | 7 | 4561.32 | 486.34 | 0.937 | 0.116 | 1.053 | 0.0097 |
| `npead3ia` | `attention` | 12 | 14 | 4816.07 | 338.57 | 1.369 | 0.143 | 1.512 | 0.0144 |
| `84e9ncw5` | `entity_transformer_500k` | 8 | 8 | 4220.10 | 430.07 | 1.007 | 0.184 | 1.190 | 0.0387 |
| `4piy4g8d` | `entity_transformer_500k` | 12 | 15 | 3456.97 | 298.38 | 1.470 | 0.246 | 1.716 | 0.0095 |

Interaction warning: `candidate_count=12` and `entity_transformer_500k` both increase per-update cost. The combination is the heaviest sentinel point, with median-like `env_steps_per_sec` dropping to 298.38 and update time increasing to 1.716 seconds. It still completed under the bounded sentinel budget, but future sweeps on model capacity or task complexity should use focused comparisons before attributing performance changes solely to hyperparameters.

## Recommended Follow-Ups

- Use the selected baseline as the default comparison point for near-term sweeps.
- Run longer validation before claiming final policy strength or publishing a performance baseline.
- Run focused opponent-profile sentinels only when pairing compatible `opponents` and `curriculum` profiles, such as `opponents=latest_only curriculum=latest_only`.
- Revisit workstation comfort if increasing `format`, rollout length, model size, or task complexity.
