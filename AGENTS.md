# Orbit Wars Agent Guide

Hydra-first Python 3.12 JAX PPO project. See `docs/ONBOARDING.md` for architecture depth, `docs/AGENT_CAPABILITIES.md` for task prompts, `docs/CURSOR.md` for plugin setup, and `docs/agent-native-phase3-status.md` for shipped benchmark/sweep primitives.

## Commands

```bash
uv sync --group dev
make help                                         # Makefile targets + agent-context
make agent-context                              # JSON: preflight, roadmap, recent runs
make test                                         # default verification (= test-fast)
make test-fast                                    # alias for daily CPU loop
make preflight-calibrate                          # refresh thresholds from preflight_calibrate_* campaigns
make preflight-learn-proof                        # Gates 2–3 JAX trend checks
uv run ow benchmark learn-proof --eval-checkpoint ... --baselines noop  # Gate 5 tournament win proof
uv run ow train training.total_updates=1000
uv run ow train print_resolved_config=true
uv run ow train kaggle
uv run ow eval tournament --checkpoint ... --campaign my_campaign
uv run ow eval package --checkpoint ... --validate-docker
uv run ow runs list
uv run ow runs show --run outputs/campaigns/<c>/runs/<run_id>
uv run ow runs watch --run outputs/campaigns/<c>/runs/<run_id>
uv run ow eval status --run outputs/campaigns/<c>/runs/<run_id>
uv run ow eval results list --run outputs/campaigns/<c>/runs/<run_id>
uv run ow eval worker --run outputs/campaigns/<c>/runs/<run_id> [--verbose]
uv run ow eval submit --checkpoint outputs/.../jax_ckpt_last.pkl
uv run ow train ... artifacts=hybrid_promotion   # strict promotion: docker + tournament async
```

## Layout

| Path | Role |
|------|------|
| `src/config/` | Dataclass schema + Hydra composition |
| `src/jax/` | Training loop, env, features, PPO, rollout |
| `src/features/` | Registry + extractor (planet-edge encoding) |
| `src/game/` | Reference Python game logic |
| `conf/` | Hydra responsibility groups |
| `docs/solutions/` | Documented solutions to past problems (bugs, best practices, workflow patterns); organized by category with YAML frontmatter (`module`, `tags`, `problem_type`); relevant when implementing or debugging in documented areas |
| `tests/` | pytest; `@pytest.mark.slow` / `jax` for expensive tiers |

## Test tiers

- **Default:** `make test` / `make test-fast` — serial CPU (`-m "not slow and not jax and not sweep"`).
- **Daily (parallel wall clock):** `make test-daily` — `test-fast` ∥ `test-jax`; or `make test-daily-parallel` with xdist.
- **Kaggle env parity:** `make test-kaggle-parity` — `test_jax_env_parity.py` in the jax tier (not slow).
- **Pre-merge:** `make test-premerge` = `test-daily` + `test-slow` (`slow and not sweep`). Before release: `make test-sweep`.
- **Parallel CPU tiers:** `make test-fast-parallel` / `make test-jax-parallel` only (sets `ORBIT_WARS_PYTEST_XDIST=1`; never on slow/sweep/GPU).
- **Never:** bare `pytest -n` without the Makefile targets; never xdist on slow/sweep; `make test-full` only when user asks (~15 min WSL2).

## Key invariants

- Feature path: obs → `encode_turn` (`TurnBatch`); golden tests in `tests/test_feature_encoding_golden.py`.
- Rollout metrics: merge with `_merge_metric_dicts` inside `jax.lax.scan`; finalize rates after the scan.
- Priorities: human `docs/ROADMAP.md` (no automated validation).
- **Verification thresholds:** derive pass/fail bars from measured calibration or baseline runs (`docs/benchmarks/preflight-calibration.json`, `make preflight-calibrate`); never invent round numbers or relax a threshold until a run passes — that is not verification.

<!-- preflight-thresholds -->
- **Calibrated learning signal (Gates 2–4):** `window_updates=10`, `min_win_rate_delta=0.05`, `max_approx_kl=0.15`, `min_entropy=0.0001` — source `docs/benchmarks/preflight-calibration.json` (commit `614cf36e732c`).
- **Tournament win proof (Gate 5):** `noop_min_win_rate=0.7`, `random_min_win_rate=0.58`.
<!-- /preflight-thresholds -->

- **Metric gates:** before gating on a training metric, confirm its denominator and what “chance” means for that opponent and reward mode (e.g. `overall_win_rate` vs `noop_only` is not ~50%; `episode_reward_mean` under `binary_win` is not win rate; self-play ~50% is not a learning signal).
- **Seed scheduler:** default `training.reseed_every_updates=50` (calibrated 2026-06-03, `docs/benchmarks/seed-scheduler-calibration.json`); use `-1` for auto `max(25, total_updates // 10)`. Run `ow benchmark calibrate-seed-scheduler` before changing. Reseed resets rollout env state, not just the PRNG key.

## Learned User Preferences

- Prefer unified v2-only feature encoding; remove legacy v1 paths rather than parallel encoders.
- Favor full rework over incremental shims when simplifying encoding, rollout, training modules, or any refactor — no backward-compat re-export modules, `_underscore` aliases, or parallel APIs for module moves/renames in a solo project; update call sites instead.
- Daily dev loop: `make test-fast` or a domain Makefile target — not bare full `pytest` and not slow/JAX-compile smokes unless explicitly requested.
- Use `make test-fast-parallel` / `make test-jax-parallel` for CPU xdist only; never xdist on slow/sweep or with GPU pytest.
- Commit verified work locally without asking; **do not push** to remote unless the user explicitly requests it.
- Check the terminals folder before starting tests or expensive GPU work (`ow train`, calibration, Gate 5 tournaments); skip if another session is running pytest/training or the user says verification is done — one GPU, parallel jobs contend.
- New train/eval/benchmark capabilities belong in the `ow` CLI (`src/cli/<module>.py` + dispatch in `src/cli/__init__.py`), not standalone `scripts/*.py`; defer heavy imports (JAX) until command execution, not module load. For agent loops prefer **primitive** subcommands (`ow runs`, `ow eval status`, `ow eval jobs cancel`, `ow benchmark gate run`, `ow benchmark tournament-proof`, `ow sweep`) over workflow wrappers (`learn-proof`, hybrid promotion train); task prompts in `docs/AGENT_CAPABILITIES.md`.
- Parallel multi-agent work: at most one full pytest/Makefile suite repo-wide; executor agents run targeted tests only; coordinator runs `make test-fast` after integration.
- Prefer `task=shield_cheap` or `task=shield_off` for training experiments; avoid `shield_tiered` unless explicitly requested.
- When the user attaches PR diff context or cites a PR number, fetch with `gh pr view` / `gh pr diff` — do not rely on inline diff blobs alone.
- After requirements/brainstorm doc review, commit the doc locally and defer implementation planning to a separate agent (`/ce-plan` or LFG) unless asked to implement in the same pass.
- `.audit/` and `.cursor/hooks/state/` are gitignored; keep audit trails and hook state local, not in commits.

## Learned Workspace Facts

- Canonical feature path: Kaggle/JAX obs → `encode_turn` (planet-edge `TurnBatch`); golden tests live in `tests/test_feature_encoding_golden.py`.
- JAX concerns are split: rollout in `src/jax/rollout/collect.py`, PPO in `src/jax/ppo_update.py`, learner shielded sampling in `src/jax/action_sampling.py`, shield in `src/jax/shield/*` (Python refs in `src/game/shield.py`, `shield_config.py`), training loop in `src/jax/train/`, opponent builders in `src/opponents/jax_actions/`. `OPPONENT_FAMILY_*` in `src/opponents/constants.py`; import `src.opponents.pool` only for pool logic.
- Per-format timing metrics (`*_2p`/`*_4p`) emit only when telemetry `metric_groups.debug` is enabled; `average_placement_4p` stays on the default path.
- Hydra dataclass defaults in `src/config/schema.py` can differ from `conf/` YAML; verify with `uv run ow train print_resolved_config=true`.
- `docs/ROADMAP.md` is human-only — no `scripts/roadmap.py` funnel or `tests/test_roadmap.py` enforcement.
- **Submit-valid funnel:** Agent decision tree and copy-paste prompts in `docs/AGENT_CAPABILITIES.md` (hybrid poll → `ow eval results show` for `validation_ok`, or `ow eval package --validate-docker` with JSON `"ok": true`). Local replay HTML and packaging-only are never submit-valid proof.
- **Hybrid promotion:** `artifacts=hybrid_promotion` sets `promotion.strategy=hybrid`, `tournament.enabled=true`, `checkpoint_eval_async=true`, `replay.enabled=false`. Scalar metric improvements queue a composite `checkpoint_eval` worker job (Docker validation → tournament → promote). Training never writes promoted manifests on metrics alone under hybrid. Artifacts: `evaluations/checkpoint_eval_u<update>_<id>/{docker_validation,tournament}/`. Profile requires Docker on the worker host. Poll `ow eval status --run <path> --watch` until idle; status JSON includes `checkpoint_evals[]` with `validation_ok` (use `ow eval results show` for full manifest). Dry-run `ow eval jobs cancel` before cancelling queue jobs. Bare `ow train` uses `artifacts=default` (metric promotion + replay per `conf/artifacts/base.yaml`)—not the submit-valid funnel.
- **Train run layout:** `ow train` with `output.campaign=<name>` writes under `outputs/campaigns/<campaign>/runs/<run_id>/` — update metrics in `logs/*_jax.jsonl` (not `metrics.jsonl` at run root), checkpoints in `checkpoints/jax_ckpt_*.pkl`, artifact jobs in `queue/optional_jobs/`, evaluation outputs in `evaluations/`.
- **Curriculum pre-flight:** `curriculum.enabled=true` requires `opponents.snapshot.pool_size > 0` and `opponents.snapshot.interval_updates > 0`; use `curriculum=off` for isolated train smokes that don't exercise historical opponents.
- **Parametric edge catalog:** default `intercept_anchors` `[1.0, 3.0, 6.0]`; edge dim `6×N+7` (25 for default). Implemented in `src/features/catalog/edge.py` and `conf/task/base.yaml`; `docs/feature-encoding-v2.md` may still describe the older two-anchor layout.
- **Preflight gates:** Gates 2–4 read JAX learning-signal trend from `logs/*_jax.jsonl`; Gate 5 runs **Docker packaging validation first**, then held-out unified tournament proof via `kaggle_environments`. **Submit-valid order:** package/Docker validate → tournament ladder → upload (`ow eval package --validate-docker`, `ow benchmark tournament-proof`, hybrid `checkpoint_eval`). **Unified ladder** (2p+4p combined score, noop/random prerequisites, incumbent Stage 2 with per-seed 100%): `ow benchmark tournament-proof` and hybrid `checkpoint_eval` when `artifacts.unified_tournament.enabled=true` (`artifacts=hybrid_promotion`). Floors in `docs/benchmarks/preflight-calibration.json` `unified_tournament` section; `enforcement: false` until GPU calibration campaigns complete (`ow benchmark calibrate-unified-tournament`). Train overrides: `conf/benchmark/gates/*.yaml` via `src/jax/preflight_gate_loader.py`. Primitives: `ow benchmark gate run <name>`, `ow benchmark tournament-proof`; composer: `ow benchmark learn-proof`.
- **Launch hygiene throughput (tier-1 vs tier-2):** `make test-launch-hygiene-throughput` runs `ow benchmark factorized-sampler` (tier-1 microbench; script `scripts/benchmark_factorized_sampler.py` is deprecated) — merge throughput health requires tier-2 `make test-launch-hygiene-e2e-throughput`, which subprocesses `ow benchmark training` on the production path with `--preset primary` vs `docs/benchmarks/launch-hygiene-e2e-baseline.json` (`--assert-within-pct`, same GPU machine as baseline capture). Baseline SHA: first parent of PR #163 merge (`79162a2088160b8ed05c3e3a050e064c7f6c9556`, pre-hygiene). Capture: worktree at that SHA, N≥3 runs, `env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0`. Example:
  ```bash
  uv run ow benchmark training --preset primary --label capture --repeats 3 \
    --updates 20 --warmup 2 --out docs/benchmarks/launch-hygiene-e2e-baseline.json
  uv run ow benchmark training --preset primary --label gate --updates 20 --warmup 2 \
    --out /tmp/gate.json --baseline docs/benchmarks/launch-hygiene-e2e-baseline.json --assert-within-pct 10
  ```
