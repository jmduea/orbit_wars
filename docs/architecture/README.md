# Architecture docs

Stage-level design notes for Orbit Wars pipelines. Each doc includes data/control
flow and owner modules; update the index when adding a new stage doc.

| Doc | Topic | Primary owners |
| --- | --- | --- |
| [output-layout.md](output-layout.md) | Canonical `outputs/` paths, campaign identity, sweep YAML | `src/artifacts/run_paths.py`, `src/config/runtime.py`, `scripts/make_wandb_sweep.py` |
| [jax-policy-encoder.md](jax-policy-encoder.md) | Planet-edge encoder dispatch and checkpoint metadata | `src/jax/policy.py`, `src/jax/encoders/` |
