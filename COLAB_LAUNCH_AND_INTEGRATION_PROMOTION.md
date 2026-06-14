# Colab Launch and Integration Promotion Tracker

Created: 2026-06-14
Status: active

This tracker supersedes earlier scattered plans for the current Orbit Wars decision push. The operating goal is to launch an efficient Colab-backed training run from `orbit_wars-integration`, produce a submit-valid artifact plus convincing held-out evidence by 2026-06-20, and only then finish making `orbit_wars-integration` the clean successor to `main`.

## Priority Contract

1. Colab training wins until the real long run is successfully underway.
2. `orbit_wars-integration` is the candidate training base and the source of truth for launch prep.
3. Work from `main` is not presumed beneficial. Cherry-pick or migrate only if the change meaningfully improves training correctness, learning quality, artifact validity, or throughput.
4. Negative throughput impact requires a visible learning-quality or correctness benefit. Negative throughput with no visible benefit must not be merged.
5. Before launch, allow only high-confidence, low-risk training-pipeline improvements that are reasonably expected to measurably improve throughput or learning quality.
6. After the long run is underway, shift attention to the integration promotion path.
7. Do not push or change remotes unless explicitly requested.
8. If credentials or manual Colab/W&B/Kaggle steps cannot be handled from the terminal, stop and ask for help.

## Launch Strategy

Use the repo's current best-known training recipe unless inspection shows it is broken, stale, or wasting compute. Colab efficiency is a hard requirement: unoptimized or visibly wasted GPU credits are not acceptable.

The launch is two-stage:

1. Medium pilot run.
2. Full long run only if the pilot passes the gate below.

If the pilot is operationally healthy but learning is weak or ambiguous, take one focused iteration on recipe, throughput, or learning fixes and rerun the pilot. If the second pilot is still weak or ambiguous, stop and reassess before spending long-run credits.

## Pilot Gate

The pilot must prove all of the following before the full run starts:

- Colab GPU is actually used and utilization is reasonable for the selected geometry.
- Training is stable enough to run without immediate intervention.
- W&B logging works for live monitoring.
- Local campaign outputs can be synced back for auditability.
- Checkpoints and artifacts are written in the expected run layout.
- Early learning metrics trend in the right direction for the configured opponent/curriculum.
- A pilot checkpoint can reach the artifact dry-run path, including package/Docker validation where available.

## Full-Run Gate

The full run may start only after the pilot gate passes. Use the current repo plan/defaults for duration and stopping behavior, but stop early if pilot-derived evidence or live metrics show credits are being wasted.

The 2026-06-20 continuation decision requires both:

- Submit-valid artifact evidence: packaging/Docker validation and a Kaggle-ready artifact path.
- Held-out evaluation evidence: tournament/gate evidence strong enough to justify continued development.

The exact continuation bar must be proposed from current calibration docs before launch, not assumed from stale thresholds.

## State and Evidence

Both monitoring and local audit state are required:

- W&B is the live monitoring source.
- Synced local `outputs/campaigns/...` state is the source for packaging, tournament proof, and auditability.

## Integration Promotion Track

After the long Colab run is safely underway:

1. Inventory current `orbit_wars-integration` state and dirty files.
2. Derive any remaining migration candidates from diffs, manifests, docs, and current failures.
3. Evaluate each candidate against learning quality, correctness, and throughput impact.
4. Apply only candidates with a clear benefit case.
5. Clean up the integration branch so it can serve as the new main if the trained submission justifies continued work.
6. Prepare and, if locally appropriate, execute the local promotion path. Do not push or remote-merge without explicit approval.

Prefer simplicity over new branch/worktree sprawl. Preserve existing local changes and work around them unless there is a real safety need.

## Progress

- [x] Interview completed and priority contract captured.
- [x] Inspect main harness and integration worktree state.
- [x] Identify the intended Colab launch path.
- [x] Inspect current best-known training recipe and Colab efficiency controls.
- [x] Inspect calibration docs and propose continuation bar.
- [x] Define exact pilot command/config and sync procedure.
- [x] Run local smoke/config validation.
- [x] Launch medium Colab pilot.
- [x] Evaluate pilot against gate.
- [x] Apply one focused iteration if pilot learning is weak or ambiguous.
- [ ] Launch full long Colab run after pilot passes.
- [ ] Shift to integration promotion once long run is safely underway.

## Current Workspace Boundary

Initial boundary, subject to inspection:

- `orbit_wars-integration`: candidate training base and primary edit target.
- `orbit_wars`: main harness for comparison/delegated commands only, unless a harness change is required.
- Existing local changes must be preserved.

## Inspection Notes - 2026-06-14

Workspace:

- `orbit_wars` is on `main` and has unrelated local edits. Treat it as harness/comparison only.
- `orbit_wars-integration` is on `refactor/artifacts-metric-promotion-commit`, ahead of its remote by one commit before this tracker, with an existing untracked benchmark JSON.
- Current edits for this push live in `orbit_wars-integration`.

Colab path:

- `ow train colab` exists and routes through `src/cli/train_hosts.py`, `src/cli/colab_runner.py`, `src/orchestration/colab_runner.py`, `src/orchestration/remote_package.py`, `src/orchestration/remote_worker.py`, and `scripts/colab_worker_entry.py`.
- `docs/colab_runner.md` records a previous T4 proof run with successful launch, sync, checkpoints, logs, and 10 updates.
- `uv run ow train colab preflight` passed locally: Colab CLI found, auth OK, T4 accepted, package directory writable.
- `uv run ow train colab launch --dry-run ...` rendered a package successfully. The tarball included the map pool and emitted no package warnings.

Training recipe findings:

- The long-run-shaped integration recipe should pass `training.rollout_steps=256` explicitly. `training=2p4p_32_split` sets env split and microbatching but does not set rollout steps by itself.
- `task=map_pool` composes with `train_bundle=production_mix`, `training=2p4p_32_split`, 32 envs, 256 rollout steps, candidate count 3, and the packaged map pool at `data/jax_map_pool/default_v1.npz`.
- `task=rollout_selected_validate` composes with the same geometry and uses `rollout_factorized_sampling=selected_validate`; keep it as a preflight/sweep option, not the default long-run shape unless evidence says it is needed.
- The current production-mix train bundle resolves to latest/self-play weighting with snapshot pool size 2 and interval 10.

Continuation bar proposal:

- Pilot learning signal: use the calibration JSON learning-signal window where applicable: 10-update window, win-rate delta at least 0.05, approximate KL no more than 0.15, entropy at least 0.0001. Because production-mix self-play can make raw win rate less interpretable, also inspect loss stability, rollout timing, opponent composition, and checkpoint health before treating win-rate alone as decisive.
- Submit-valid proof: run `ow eval package --checkpoint <pilot_checkpoint> --output-dir <dir> --validate-docker --packaging-seed 0 --packaging-player-count 4` and require JSON `"ok": true` or equivalent validation marker.
- Held-out proof before the June 20 continue decision: use the unified tournament thresholds in `docs/benchmarks/preflight-calibration.json`: combined noop and random floors of 0.76 with `games_per_pair=2`, plus the prerequisite seed set recorded there.

Known caveat:

- On this local machine, `ow eval package --help` and `ow benchmark tournament-proof --help` printed valid help text but exited with native memory errors afterward. Do not treat help exit code alone as submit-valid failure; validate the real command on a checkpoint and track any repeat crash separately.

## Pilot 1 - 2026-06-14

Command shape:

```bash
uv run ow train colab launch --gpu T4 --timeout 7200 \
  training.total_updates=50 \
  output.campaign=colab_pilot \
  task=map_pool \
  training=2p4p_32_split \
  training.rollout_steps=256 \
  train_bundle=production_mix \
  telemetry.wandb.enabled=true \
  telemetry.wandb.group=colab_pilot \
  telemetry.metric_groups.losses=true \
  artifacts=disabled \
  artifacts.checkpoint_every=50
```

Result:

- Colab session: `ow-colab_pilot-321b917`
- Worker status: `colab_complete`
- GPU proof: JAX default backend `gpu`, device `cuda:0`
- Synced run: `outputs/colab_runner/synced/colab_pilot/runs/20260614T165511Z-s42-3c1b5463`
- Checkpoints: `jax_ckpt_000050.pkl`, `jax_ckpt_last.pkl`
- W&B local files synced under the run cache; final W&B summary recorded update 50.
- Package/Docker validation: PASS, JSON `"ok": true`
- Submission artifact: `outputs/colab_runner/synced/colab_pilot/package_validation/submission.tar.gz`

Throughput:

- First update: `update_seconds=132.21`, `rollout_seconds=98.73`, `samples_per_sec=123.93`
- Warm mean after initial compile: `update_seconds=5.53`, `rollout_seconds=4.57`, `samples_per_sec=3009.60`
- Final update: `update_seconds=12.62`, `rollout_seconds=11.69`, `samples_per_sec=1298.14`

Learning signal:

- `overall_win_rate` first 10 = `0.452`, last 10 = `0.304`
- `win_rate_2p` first 10 = `0.516`, last 10 = `0.439`
- `episode_reward_mean` first 10 = `0.004`, last 10 = `-0.392`
- Final `episode_reward_mean=-1.0`, final `overall_win_rate=0.0`

Gate decision:

- Operational gate: PASS
- W&B/sync gate: PASS
- Artifact dry-run gate: PASS
- Early learning gate: FAIL/ambiguous

Next action per priority contract: take one focused iteration before spending full long-run credits.

## Pilot 2 - 2026-06-14

Focused iteration:

- Keep the long-run-shaped `task=map_pool`, `training=2p4p_32_split`, `training.rollout_steps=256`, and `train_bundle=production_mix`.
- Increase pilot length from 50 to 100 updates.
- Add `training.reseed_every_updates=25`, matching the fixed preflight sweep axis.
- Keep remote artifact workers disabled and checkpoint at the final update.

Command shape:

```bash
uv run ow train colab launch --gpu T4 --timeout 10800 \
  training.total_updates=100 \
  output.campaign=colab_pilot_iter2 \
  task=map_pool \
  training=2p4p_32_split \
  training.rollout_steps=256 \
  training.reseed_every_updates=25 \
  train_bundle=production_mix \
  telemetry.wandb.enabled=true \
  telemetry.wandb.group=colab_pilot \
  telemetry.metric_groups.losses=true \
  artifacts=disabled \
  artifacts.checkpoint_every=100
```

Result:

- Colab session: `ow-colab_pilot_iter2-321b917`
- Worker status: `colab_complete`
- GPU proof: JAX default backend `gpu`, device `cuda:0`
- Synced run: `outputs/colab_runner/synced/colab_pilot_iter2/runs/20260614T170811Z-s42-8278dd50`
- Checkpoints: `jax_ckpt_000100.pkl`, `jax_ckpt_last.pkl`
- No active Colab sessions after stop.

Throughput:

- First update: `update_seconds=114.47`, `rollout_seconds=84.86`, `samples_per_sec=143.13`
- Warm mean after initial compile: `update_seconds=5.44`, `rollout_seconds=4.50`, `samples_per_sec=3037.09`
- Final update: `update_seconds=6.81`, `rollout_seconds=5.87`, `samples_per_sec=2405.66`

Learning signal:

- `overall_win_rate` first 10 = `0.452`, last 10 = `0.383`, delta `-0.068`
- `win_rate_2p` first 10 = `0.516`, last 10 = `0.551`, delta `+0.035`
- `episode_reward_mean` first 10 = `0.004`, last 10 = `-0.233`, delta `-0.237`
- `average_placement_4p` worsened from `1.648` first 10 to `2.590` last 10.

Gate decision:

- Operational gate: PASS
- W&B/sync gate: PASS
- Artifact path: already proven by Pilot 1 package/Docker validation
- Early learning gate: FAIL/ambiguous after the single focused iteration

Decision: do not launch the full long run yet. Stop and reassess before spending long-run credits.

## Correction - Preflight Sweep Is Candidate Selection

The branch already carries a W&B `preflight` sweep recipe intended to identify promising configurations before spending long-run Colab compute. The Colab pilots above are valid operational/package proofs, but they are not sufficient recipe selection.

Current sweep recipe, as initially registered:

- Source: `conf/wandb_sweep/preflight.yaml`
- Generated YAML: `outputs/_meta/sweeps/preflight.yaml`
- W&B sweep: `jmduea-jdueadev/orbit_wars/85u3e192`
- Objective: maximize `preflight_sweep_score`
- Run cap: 24
- Fixed axes: `task=rollout_selected_validate`, `training=2p4p_32_split`, `training.rollout_steps=256`, `training.total_updates=100`, `training.reseed_every_updates=25`, `train_bundle=production_mix`
- Search space: model size, shield mode/horizon, feature-history steps, edge rank mode, PPO lr/clip/entropy/value/grad norm, seed

Fixes made before relying on this path:

- Corrected `conf/wandb_sweep/fixed/preflight.yaml` so `training.reseed_every_updates` is a W&B parameter spec (`value: 25`) rather than a bare scalar.
- Corrected `ow sweep list` to query W&B with `api.project(project, entity=entity)`.
- Corrected `ow sweep create --backend wandb` to pass `--project` and `--entity` through to `wandb sweep`.
- Corrected shortlist ranking so `preflight_sweep_score` / `ssot_preflight_sweep_score` wins over raw `episode_reward_mean` when present.
- Later verification found this registered W&B sweep is not suitable for preflight candidate selection because `train_bundle=production_mix` resolves to latest self-play, not noop/random. Treat sweep `85u3e192` as invalidated for long-run launch decisions.

Next decision path:

- Use the preflight sweep as the candidate-selection surface before any full long Colab run.
- Do not launch a long run from a candidate whose final `preflight_sweep_score` is ineligible.
- Prefer running more sweep agents over hand-tuning unless the sweep recipe itself is broken.

## Preflight Sweep Validation - 2026-06-14

Validation command:

```bash
uv run wandb agent --count 1 --project orbit_wars --entity jmduea-jdueadev \
  jmduea-jdueadev/orbit_wars/85u3e192
```

Result:

- W&B run: `noxtxka5`
- Local run: `outputs/campaigns/preflight/runs/20260614T173456Z-s43-ae05c98e`
- Training completed 100 updates and wrote `jax_ckpt_last.pkl`.
- The sweep recipe executed on the intended preflight shape: `task=rollout_selected_validate`, `training=2p4p_32_split`, `training.rollout_steps=256`, `training.total_updates=100`, `train_bundle=production_mix`.
- Early score was briefly positive (`preflight_sweep_score=0.095` around update 14), but the final score was ineligible.

Final shortlist snapshot:

- Command: `uv run ow train colab shortlist --project orbit_wars --entity jmduea-jdueadev --sweep-id 85u3e192 --limit 5 --out outputs/colab_runner/preflight_85u3e192_shortlist.json`
- Output: `outputs/colab_runner/preflight_85u3e192_shortlist.json`
- Final `preflight_sweep_score=-1.0`
- Final `approx_kl_window_mean=0.26832658126950265`
- Final `entropy_window_mean=1.2732656955718995`
- Final `samples_per_sec=2521.3150196653837`
- W&B checkpoint artifact: none attached

Gate decision:

- Sweep recipe smoke: PASS
- Candidate quality: FAIL
- Long-run launch from this candidate: NO

Additional fix:

- `rows_from_wandb_runs` now skips non-scalar W&B summary values instead of crashing during shortlist extraction.
- Regression coverage added in `tests/test_kaggle_runner.py`.

## Launch Hygiene / Deduplication Audit - 2026-06-14

User request: while the sweep validation finished, double-check that integration contains the launch deduplication and masking techniques from main/cleanup branches that improve learning quality at limited throughput cost.

Audit result:

- Integration contains the launch hygiene bundle lineage:
  - `e4463c4` launch hygiene bundle for K-step factorized decoder
  - `39217d5` align hygiene `launch_valid` with real non-noop bucket/fraction
  - `97025e3` launch hygiene review tests
  - `c224f8a` incremental hygiene carry
  - `18a548c` compact PPO carry and replay shortcuts
  - `2408556` selected-validate sampling path
  - `f9f5442` preflight sweep wired to `rollout_selected_validate`
  - `e565912` O(K) incremental decode in rollout scan
- `src/jax/launch_hygiene.py` has turn-static `HygieneLookups`, compact `ForbiddenCarry`, duplicate source/target masking, and friendly reverse-edge masking.
- `src/jax/action_sampling.py` applies cumulative hygiene before source-mask construction, updates carry only after selected/tiered validation accepts a launch, and recomputes replay log-probs for selected-validate/tiered paths.
- `src/jax/factored_sequence_scan.py` reconstructs hygiene during PPO replay from the stored shield-only mask stack, avoiding double hygiene application.
- `conf/task/base.yaml` keeps `rollout_factorized_sampling: lattice` as the default; `conf/task/rollout_selected_validate.yaml` is opt-in for preflight/sweep use.
- Cleanup branches checked did not contain additional unlanded launch-hygiene semantics ahead of integration. Several branches are behind integration or missing pieces of this stack.

Focused verification:

```bash
uv run --group dev pytest \
  tests/test_launch_hygiene.py \
  tests/test_rollout_selected_action_validation.py \
  tests/test_factored_sequence_scan.py
```

Result: 25 passed, 1 skipped.

Conclusion: integration has the critical launch deduplication/masking learning-quality techniques. No launch-hygiene cherry-pick is currently indicated before continuing the preflight shortlist path.

## Preflight Opponent Verification - 2026-06-14

Verification request: confirm that preflight sweeps test against noop and/or random opponents.

Result:

- Registered W&B sweep `85u3e192` does not test against noop or random.
- Its fixed axis used `train_bundle=production_mix`, which composes to `opponents=self_play_curriculum` and `curriculum=production_mix`.
- `production_mix` currently weights `latest: 1.0`, `historical: 0.0`, `noop: 0.0`, `random: 0.0`.
- A local agent run from `85u3e192` was interrupted and its child `ow train` process was terminated to avoid spending hardware on the wrong opponent mix.

Correction:

- `conf/wandb_sweep/fixed/preflight.yaml` now uses `train_bundle=opponent_recovery_floor`, which resolves to noop dispatch with self-play off and curriculum off.
- Tags now identify this as `noop` instead of `production_mix`.
- Corrected W&B sweep registered: `jmduea-jdueadev/orbit_wars/0mn8n6g0`

1. Register a corrected W&B preflight sweep from the local recipe.
2. Run one local W&B preflight agent at a time against the corrected sweep.
3. Generate a Colab shortlist from that corrected sweep.
4. Package/Docker validate the shortlist winner checkpoint.
5. Launch the long Colab run from the validated shortlist winner's overrides only if the preflight objective and artifact validation pass.
