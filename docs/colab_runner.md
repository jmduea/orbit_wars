# Colab Runner

Preferred operator entry: **`uv run ow train colab`**. Use `python -m src.cli.colab_runner` for direct script invocation.

## Shape

Colab is the hosted GPU backend for long training runs after local W&B preflight sweeps:

- The local launcher renders a tarball package (`orbit_wars.tgz`) plus `worker-env.json`.
- `google-colab-cli` provisions a GPU session, uploads the payload, and execs a bootstrap script.
- The remote worker runs `uv sync --group dev`, verifies JAX GPU, then `uv run ow train …`.
- Sync checkpoints and `logs/*_jax.jsonl` back locally for eval, gates, and submit.

Preflight W&B sweeps stay **local**; Colab does not run `wandb agent` in v1.

## Prerequisites

```bash
uv tool install google-colab-cli
colab auth
uv run ow train colab preflight
```

Packages, ledgers, and synced outputs default to `outputs/colab_runner/` (`kernel/`, `launches.jsonl`, `sessions.json`, `synced/`).

## Files

- CLI: `uv run ow train colab` → `src/cli/train_hosts.py` → `src/cli/colab_runner.py`
- Orchestration: `src/orchestration/colab_runner.py`, `src/orchestration/colab_cli.py`
- Shared packaging: `src/orchestration/remote_package.py`
- Shared bootstrap: `src/orchestration/remote_worker.py`
- Worker entry: `scripts/colab_worker_entry.py`

## Operator commands

```bash
# Preflight / prepare
uv run ow train colab preflight
uv run ow train colab prepare --gpu T4 training.total_updates=10 output.campaign=colab_smoke

# Shortlist after local W&B preflight sweep
uv run ow train colab shortlist --sweep-id <id> --out outputs/colab_runner/shortlist.json

# Long run (explicit overrides)
uv run ow train colab launch --gpu T4 --timeout 86400 \
  training.total_updates=2000 \
  opponents=throughput_recovery \
  output.campaign=colab_long \
  telemetry.wandb.enabled=true

# Long run (from shortlist row 0, with extra overrides)
uv run ow train colab launch --from-shortlist outputs/colab_runner/shortlist.json --rank 0 \
  --gpu T4 training.total_updates=2000 task=map_pool

# Poll + pull artifacts
uv run ow train colab status --session ow-colab_long-<sha>
uv run ow train colab sync --session ow-colab_long-<sha>
uv run ow train colab stop --session ow-colab_long-<sha>
```

Hydra overrides after Colab flags use the same token rules as `ow train kaggle`.

## Worker behavior

Inside Colab, the worker:

1. Loads packaged environment values from `worker-env.json`.
2. Installs `uv` if needed, then runs `uv sync --group dev`.
3. Verifies JAX sees a GPU unless `ORBIT_WARS_COLAB_ALLOW_CPU=1`.
4. Runs `uv run ow train` with packaged `HYDRA_OVERRIDES`.
5. Writes `worker-summary.json` with diagnostics and exit code.

Useful worker environment keys:

- `ORBIT_WARS_COLAB_WORKER_MODE` — `standalone` (v1 only)
- `ORBIT_WARS_COLAB_TRUST_BASE_JAX` — default `0` (full `uv sync` JAX pins)
- `WANDB_API_KEY` — optional; inject via `worker-env.json` for remote W&B logging

## Preflight → long-run recipe

```bash
# Local preflight (existing)
uv run ow sweep create --backend wandb \
  --sweep-yaml conf/wandb_sweep/fixed/preflight.yaml
wandb agent <entity>/orbit_wars/<sweep_id>

# Pick winner
uv run ow train colab shortlist --sweep-id <sweep_id> \
  --out outputs/colab_runner/shortlist.json

# Long run on Colab
uv run ow train colab launch \
  --from-shortlist outputs/colab_runner/shortlist.json --rank 0 \
  --gpu T4 --timeout 86400 \
  training.total_updates=2000 \
  output.campaign=colab_long \
  task=map_pool

# Pull results for local pipeline
uv run ow train colab sync --session <slug>
uv run ow runs show --run outputs/colab_runner/synced/colab_long/runs/<run_id>
```

## Proof runs

### U0 spike (2026-06-07)

- Colab CLI 0.5.9, OAuth2 auth, T4 provisioned
- Tarball upload + `uv sync --group dev` + 3-update `ow train` smoke: **PASS**
- Bootstrap wall ~255 s; cold update 1 `rollout_s≈68s`, steady-state update 3 `rollout_s≈1.5s`

### U6 operator proof (2026-06-07)

**Status: PASS**

| Item | Result |
|------|--------|
| Date/time | 2026-06-07 20:43–20:48 local (2026-06-08T01:43–01:48Z) |
| Worktree | `orbit_wars-integration` on `feat/colab-train-host` |
| Session | `ow-colab_smoke-12c2f68` |
| GPU | T4 |
| Launch wall | ~282 s (attempt 2 after one Colab API 503 retry) |
| Worker `exit_code` | **0** (`worker-summary.json`) |
| Sync path | `outputs/colab_runner/synced/colab_smoke/` |
| Run id | `20260608T014414Z-s42-f34fcd96` |
| Updates | 10 (`training.total_updates=10`, noop, `task=shield_cheap`) |
| `rollout_seconds` | update 1 **70.07** (cold compile), update 3 **1.50**, update 10 **1.49** |
| `ppo_seconds` | update 1 **28.13**, update 10 **0.40** |
| Checkpoints synced | `jax_ckpt_000010.pkl`, `jax_ckpt_last.pkl` |
| Log synced | `runs/.../logs/*_jax.jsonl` |

Operator commands (Hydra overrides as separate CLI args):

```bash
uv run ow train colab preflight
uv run ow train colab launch --gpu T4 --timeout 7200 \
  training.total_updates=10 curriculum=off output.campaign=colab_smoke \
  task=shield_cheap opponents=base opponents.mode.opponent=noop \
  telemetry.wandb.enabled=false
uv run ow train colab sync --session ow-colab_smoke-12c2f68
uv run ow train colab stop --session ow-colab_smoke-12c2f68
```

**Fixes landed during proof:** `colab upload/download/exec/stop/status` use `--session` flags (CLI 0.5.9); bootstrap is Python (not shell) for `colab exec -f`; sync archives campaign dir to tarball before download (directory download unsupported).

## Notes

- One Colab session at a time; sync before `stop` when checkpoints matter.
- Eval, Docker packaging validation, tournament ladders, and Kaggle submit stay **local**.
- Do not pipe long `ow train colab launch` output through `tail`/`head`; JSON prints on stdout, progress on stderr from the worker stream.
