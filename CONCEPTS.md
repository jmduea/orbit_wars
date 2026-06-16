# Concepts

> Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Colab train host

The third `ow train` execution backend alongside local and Kaggle. Packages the repo as a tarball, provisions a Colab GPU VM, runs training remotely, and syncs checkpoints and logs back to the operator machine. Colab v1 is training-only — packaging validation, tournament ladders, and Kaggle submit stay local after sync.

## Fixed-path pilot

A deliberately narrow Hydra recipe used to prove Colab long-run viability before promoting to heavier geometry such as map pool or mixed 2p/4p production curriculum. Operational success (GPU, sync, checkpoints) and learning success (win rate or reward trend on the intended opponent mix) are separate gates; a pilot that passes the first does not authorize a full long run without the second.

## Monitor-after-launch

The post-launch operator loop that polls Colab session status, periodically syncs remote campaign outputs, detects stale progress, and runs local checkpoint eval on newly synced weights. Restarts against the same session slug recover from terminal loss without relaunching a duplicate worker.

## Preflight sweep

A local W&B hyperparameter search over short update counts whose objective measures noop or random recovery, not self-play win rate. Winners are shortlisted for remote long runs; sweeps that fix production-mix or latest self-play axes produce ineligible scores and must not drive Colab launch decisions.
