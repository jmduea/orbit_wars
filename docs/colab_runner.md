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

### U6 operator proof

**Status: BLOCKED (deferred to operator with live Colab session)**

Automated U6 launch was not run in this implementation session to avoid occupying a long-lived Colab GPU while other operator jobs may be active. Re-run manually:

```bash
uv run ow train colab launch --gpu T4 --timeout 7200 \
  training.total_updates=10 curriculum=off output.campaign=colab_smoke \
  task=shield_cheap opponents=base opponents.mode.opponent=noop

uv run ow train colab sync --session <slug>
test -f outputs/colab_runner/synced/colab_smoke/runs/*/logs/*_jax.jsonl
```

Record wall time, GPU type, and `worker-summary.json` exit code in this section after a successful operator run.

## Notes

- One Colab session at a time; sync before `stop` when checkpoints matter.
- Eval, Docker packaging validation, tournament ladders, and Kaggle submit stay **local**.
- Do not pipe long `ow train colab launch` output through `tail`/`head`; JSON prints on stdout, progress on stderr from the worker stream.
