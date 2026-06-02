# Orbit Wars Agent Guide

Hydra-first Python 3.12 JAX PPO project. See `docs/ONBOARDING.md` for architecture depth, `docs/AGENT_CAPABILITIES.md` for task prompts, and `docs/CURSOR.md` for plugin setup.

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
uv run ow eval status --run outputs/campaigns/<c>/runs/<run_id>
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
- **Metric gates:** before gating on a training metric, confirm its denominator and what “chance” means for that opponent and reward mode (e.g. `overall_win_rate` vs `noop_only` is not ~50%; `episode_reward_mean` under `binary_win` is not win rate; self-play ~50% is not a learning signal).
- **Seed scheduler:** default `training.reseed_every_updates=-1` auto-scales to `max(25, total_updates // 10)`; run `ow benchmark calibrate-seed-scheduler` before changing. Reseed resets rollout env state, not just the PRNG key.

## Learned User Preferences

- Prefer unified v2-only feature encoding; remove legacy v1 paths rather than parallel encoders.
- Favor full rework over incremental shims when simplifying encoding, rollout, training modules, or any refactor — no backward-compat re-export modules, `_underscore` aliases, or parallel APIs for module moves/renames in a solo project; update call sites instead.
- Daily dev loop: `make test-fast` or a domain Makefile target — not bare full `pytest` and not slow/JAX-compile smokes unless explicitly requested; CPU xdist only via `make test-fast-parallel` / `make test-jax-parallel` (never on slow/sweep/GPU).
- Check the terminals folder before pytest or GPU work; one GPU job at a time; parallel multi-agent: at most one full Makefile suite repo-wide (executors run targeted tests only).
- Commit and push only when the user explicitly requests it; do not push otherwise.
- New train/eval/benchmark capabilities belong in the `ow` CLI (`src/cli/<module>.py` + dispatch in `src/cli/__init__.py`), not standalone `scripts/*.py`; defer heavy imports (JAX) until command execution, not module load.
- Prefer `task=shield_cheap` or `task=shield_off` for training experiments; avoid `shield_tiered` unless explicitly requested.
- Prefer even 2p/4p env splits for training and benchmarks (`training=workstation` → `2p4p_32_split`) to match leaderboard dynamics.
- Default factorized decoding uses `max_moves_k: 2` in `conf/model/transformer_factorized.yaml` (not 1); raise K only with evidence.
- Hydra/config tests: prefer composition and required-key validation over asserting full resolved configs match hardcoded snapshots.
- After major training-loop or checkpoint refactors, verify with short train smoke then a ~100-update benchmark; for PR context use `gh pr view` / `gh pr diff`, not inline diff blobs alone.
- `.audit/` and `.cursor/hooks/state/` are gitignored; keep audit trails and hook state local, not in commits.

## Learned Workspace Facts

- Canonical feature path: Kaggle/JAX obs → `encode_turn` (planet-edge `TurnBatch`); golden tests in `tests/test_feature_encoding_golden.py`; parametric edges in `src/features/catalog/edge.py` (`intercept_anchors` `[1.0, 3.0, 6.0]`, dim `6×N+7`).
- JAX is split across rollout, PPO, shielded sampling, shield, and training loop modules; factorized turns encode `TurnBatch` once (cached `encoder_out`) with decoder-only prefix forwards for shield/replay.
- PPO uses a shared planet-edge encoder with separate policy/value heads (not dual encoders); try `model.value_head=format_routed` before a second encoder experiment.
- `model.decoder_carry` persists decoder GRU hidden state across turns — not action tokens in observations; incoming carry is stored on transitions for PPO replay.
- Trajectory shield: JAX in `src/jax/shield/*`; Python reference in `src/game/shield.py`. `OPPONENT_FAMILY_*` in `src/opponents/constants.py`; import `src.opponents.pool` only for pool logic.
- PPO tile size: `training.update_chunk_rows` (single knob). General PPO defaults in `conf/training/base.yaml` (post-`ppo_stability_kl`: lr `6e-5`, epochs `1`, clip `0.15`, vf_coef `1.0`, max_grad_norm `1.0`, ent_coef `0.006`).
- **Obs normalization:** `model.normalize_observations` (default true) runs Welford mean/var on `TurnBatch` planet/edge/global with `obs_norm_clip` (default 10) in rollout and PPO (`src/jax/normalization.py`, `train/loop.py`).
- **`overall_win_rate` under pure `binary_win`:** counts wins from terminal reward sign (`reward>0` on `done`), not `terminal_is_first` alone (`src/jax/rollout/metrics.py`).
- Hydra schema defaults in `src/config/schema.py` may differ from `conf/` YAML — verify with `print_resolved_config=true`.
- **Hybrid promotion:** `artifacts=hybrid_promotion` → hybrid strategy + async checkpoint eval (Docker → tournament → promote).
- **Train runs:** `outputs/campaigns/<campaign>/runs/<run_id>/` — metrics in `logs/*_jax.jsonl`, checkpoints `checkpoints/jax_ckpt_*.pkl`; `curriculum.enabled=true` requires snapshot pool + interval (`curriculum=off` for isolated smokes).
- **Preflight gates:** Gates 2–4 trend from `logs/*_jax.jsonl`; Gate 5 tournament via `kaggle_environments`. Gates/calibrate append per-model PPO from `docs/benchmarks/preflight-profiles.json` (not drifting `base.yaml`); refresh profiles and `preflight-calibration.json` together when promoting sweep winners. Watch `approx_kl_v2` and |approx_kl| — signed v1 ceiling alone misses negative-KL pathology.
- **Launch hygiene throughput:** tier-1 `make test-launch-hygiene-throughput`; tier-2 merge gate `make test-launch-hygiene-e2e-throughput` vs `docs/benchmarks/launch-hygiene-e2e-baseline.json` on the same GPU machine.
- **`ow benchmark` dispatch:** register subcommands in `src/cli/__init__.py`, not only test parsers.
