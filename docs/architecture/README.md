# Architecture docs

Stage-level design notes for Orbit Wars pipelines. Each doc includes data/control
flow and owner modules; update the index when adding a new stage doc.

| Doc | Topic | Primary owners |
| --- | --- | --- |
| [output-layout.md](output-layout.md) | Canonical `outputs/` paths, campaign identity, sweep YAML | `src/artifacts/run_paths.py`, `src/config/runtime.py`, `scripts/make_wandb_sweep.py` |
| [jax-policy-encoder.md](jax-policy-encoder.md) | Planet-edge encoder dispatch and checkpoint metadata | `src/jax/policy.py`, `src/jax/encoders/` |
| [tournament-eval.md](tournament-eval.md) | Local Kaggle-env tournament ranking and promotion gates | `src/artifacts/tournament/`, `src/cli/eval.py` |
| [jax-trace-tiers.md](jax-trace-tiers.md) | JAX tier A/B/C trace-safe boundaries and CI hygiene | `scripts/jax_trace_hygiene.sh`, `tests/test_jax_trace_hygiene.py` |
