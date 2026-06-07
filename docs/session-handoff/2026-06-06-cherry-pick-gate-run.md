# Session handoff: cherry-pick gate run (2026-06-06)

Pick up after the **anchor learn-proof + throughput gate** finishes (operator running manually in background).

## Git state (main)

| Commit | Summary |
|--------|---------|
| `21aea59` | Admission throughput CLI, config picker, terminal hook, manifest |
| `357d46c` | Learning-first baseline JSON captured |
| **Uncommitted** | `--repo-root` on gate CLI + `output.root` worktree fix (`src/jax/preflight.py`, tests) |

**Branch:** `main`, ahead of origin by 2 commits (+ local uncommitted fix).

## Worktrees

| Path | SHA / branch | Role |
|------|----------------|------|
| `~/projects/orbit_wars` | `main` | Gate harness, baseline JSON, docs |
| `~/projects/orbit_wars-throughput-anchor` | `52dfdb0` `throughput-baseline` | **Candidate training code** for admission |
| `~/projects/orbit_wars-pre-hygiene` | `79162a2` detached | Baseline capture anchor (done) |

## Locked admission recipe

See `docs/session-handoff/2026-06-05-cherry-pick-manifest.md` and manifest `admission_profile` (`operator_locked`).

**Baseline:** `docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json`  
~5,432 env-steps/s mean, ~1.51 s/update (3×20-update capture on pre-hygiene @ `79162a2`).

## Gate command (run from main)

```bash
cd ~/projects/orbit_wars

uv run ow benchmark gate run beat_noop \
  --repo-root ~/projects/orbit_wars-throughput-anchor \
  --output-root ~/projects/orbit_wars-throughput-anchor/outputs \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3 \
    telemetry.wandb.enabled=true telemetry.wandb.group=preflight artifacts.replay.enabled=false \
  --out ~/projects/orbit_wars-throughput-anchor/outputs/benchmarks/cherry-pick/anchor_learn_proof.json \
  --also-throughput \
  --throughput-baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json \
  --throughput-within-pct 10
```

**Why not from the worktree?** Anchor lacks `ow benchmark gate run` (old CLI). `--repo-root` runs **anchor training code** with **main gate harness**. Requires uncommitted `output.root=outputs` fix — commit first if dry-run still shows absolute paths.

## After the gate completes

1. **Commit** uncommitted `output.root` / `--repo-root` fix if not already done.
2. **Inspect** `--out` JSON: learning `verdict`, `stage.log_path`, throughput section if `--also-throughput`.
3. **Update** `docs/benchmarks/cherry-pick-manifest.json` `baseline_gates.learn_proof` and throughput verdicts from artifacts.
4. **`make test-kaggle-parity`** on anchor (or main after cherry-picks) before Phase 2 env-parity cherry-picks onto integration @ `79162a2`.
5. **Phase 2:** env-parity commits onto `throughput-baseline-integration` per manifest — only if learn-proof + throughput + parity green.

## Threshold caveat

`beat_noop` floors were calibrated on 2p-only / 16 env / 128 steps. Locked recipe is mixed 2p/4p, 32 env, 256 steps — treat FAIL as **recalibration signal** if metrics look like geometry mismatch, not necessarily a bad cherry-pick.

## Compound doc

`docs/solutions/workflow-issues/worktree-preflight-gate-repo-root.md`

## Do not

- Compare throughput to old `launch-hygiene-e2e-baseline.json` (self-play / full model).
- Run `ow benchmark gate` inside the anchor worktree without cherry-picking gate CLI.
- Pipe gate output to `tail`/`head` — use `--out` and `ow runs watch` / log tail on jsonl.
