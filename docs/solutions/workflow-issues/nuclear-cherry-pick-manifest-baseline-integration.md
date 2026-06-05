---
title: Nuclear cherry-pick manifest — baseline-first integration with worktree gate capture
date: 2026-06-05
category: workflow-issues
module: benchmarks
problem_type: workflow_issue
component: development_workflow
severity: high
applies_when:
  - "main production training throughput is far below the pre-hygiene anchor while learning signal must be preserved"
  - "Planning selective cherry-picks onto a throughput-baseline instead of blind revert or redesign"
  - "Capturing tier-2 gate JSON from a worktree at anchor SHA 79162a2"
tags:
  - cherry-pick-manifest
  - throughput-baseline
  - worktree
  - launch-hygiene-e2e
  - kaggle-mechanics-parity
  - nomenclature-rfc
  - baseline-first
related_components:
  - docs/benchmarks/cherry-pick-manifest.json
  - docs/benchmarks/launch-hygiene-e2e-baseline.json
  - docs/nomenclature-rfc.md
  - tests/test_training_benchmark_gate.py
  - docs/plans/2026-06-05-002-feat-nuclear-cherry-pick-manifest-plan.md
---

# Nuclear cherry-pick manifest — baseline-first integration with worktree gate capture

## Context

After PR #163 (within-turn launch dedup masks / launch hygiene), **production training throughput** on `main` fell well below the documented pre-hygiene anchor while **Kaggle mechanics parity** (`make test-kaggle-parity`) stayed green — evidence that correctness and throughput are independent tracks (see `docs/nomenclature-rfc.md`). Bare revert to the anchor restores throughput but loses learning; blind forward on slow `main` or untracked cherry-picks lack auditability.

Session consolidation (docs PR #219 → `main`, stale branch cleanup, `throughput-baseline-integration` fork) established the **primary integration strategy**: pin `throughput-baseline` at `79162a2088160b8ed05c3e3a050e064c7f6c9556`, layer env-parity substrate on `throughput-baseline-integration`, then admit only learning commits that pass **both** the production training throughput gate and the learning proof ladder. Semantic rollout redesign stays **halted**; SSOT train-spine handoff is a **deferred parallel** track — not a substitute for manifest gates.

Phase 1 U1 completed in commit `9cdc8fe`: `baseline_gates.throughput_e2e.verdict: admit` at ~9628 `env_steps_per_sec` vs `launch-hygiene-e2e-baseline.json`. Learn-proof ceiling and parity captures remain `pending` before env-parity cherry-picks.

## Guidance

### Strategy and phase order

| Phase | Branch | Goal | Gate authority |
| --- | --- | --- | --- |
| 1 | `throughput-baseline` @ `79162a2` | Prove anchor throughput + record learn-proof ceiling | `baseline_gates` in manifest |
| 2 | `throughput-baseline-integration` | Cherry-pick env-parity fixes until parity green | `make test-kaggle-parity` |
| 3 | integration head | Topological learning cherry-picks | parity → tier-2 throughput → learn-proof |

Advance `throughput-baseline-integration` only on manifest **`admit`**; human merge to `main` when `integration_status: ready_for_main`. Do not use stagger smokes, `ssot_preflight`, or validation-preset bisect as tier-2 pass/fail — those answer different questions (see [jax-validation-throughput-benchmark-and-bisect.md](jax-validation-throughput-benchmark-and-bisect.md)).

### Worktree + harness copy (anchor capture)

`docs/benchmarks/launch-hygiene-e2e-baseline.json` `merge_topology_notes` documents PR #163 as a **merge** (not squash): first parent `79162a2` is pre-hygiene `main`. Capture at that SHA uses a dedicated worktree; **copy U1/U2 benchmark harness from current `main` for measurement semantics only** — training code at the worktree stays at the anchor.

```bash
# From main checkout after planning docs land
git branch throughput-baseline 79162a2088160b8ed05c3e3a050e064c7f6c9556
git worktree add ../orbit_wars-throughput-anchor throughput-baseline
git branch throughput-baseline-integration throughput-baseline   # fork integration line

# In worktree: sync measurement harness from main (paths per session — benchmark CLI / gate helpers only)
# Then capture anchor throughput (same GPU host as baseline JSON)
cd ../orbit_wars-throughput-anchor
env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
  uv run ow benchmark training --preset primary --label anchor-u1 \
  --updates 20 --warmup 2 \
  --baseline docs/benchmarks/launch-hygiene-e2e-baseline.json \
  --assert-within-pct 10 \
  --out outputs/benchmarks/cherry-pick/anchor_throughput.json
```

Record results in `docs/benchmarks/cherry-pick-manifest.json` (`baseline_gates.throughput_e2e`). Schema guard: `tests/test_training_benchmark_gate.py::test_committed_cherry_pick_manifest_artifact`.

### Gitignored gate artifacts and JSON inspection

Gate `--out` paths (e.g. `outputs/benchmarks/cherry-pick/anchor_throughput.json`) live under **`outputs/`**, which is **gitignored**. A capture from the worktree will look **missing** when grepping the main repo checkout — that is expected. The committed manifest holds verdict + summary metrics; raw JSON stays local under the worktree that ran the gate.

`ow benchmark training` prints **minified JSON on stdout** at exit. For inspection, use **`jq` on the `--out` file**, not stdout parsing or pipes through `tail`/`head` (progress and final payload behavior documented in [ow-long-cli-stderr-progress-no-tail-pipe.md](../developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md)):

```bash
jq '{
  gate_passed,
  env_steps_per_sec: .aggregate.env_steps_per_sec.mean,
  measured_commit_sha: .commit_sha
}' outputs/benchmarks/cherry-pick/anchor_throughput.json
```

U1 anchor reference (manifest `9cdc8fe`): `env_steps_per_sec` ≈ **9628**, `gate_passed: true`, `verdict: admit`. `measured_commit_sha` may differ from `baseline_sha` when the harness on `main` is newer than the anchor tree — manifest records both.

### Two parallel tracks (nomenclature)

Use `docs/nomenclature-rfc.md` prose when writing runbooks or manifest decisions:

| Track | User-facing term | Guard | Not interchangeable with |
| --- | --- | --- | --- |
| Correctness | **Kaggle mechanics parity** | `make test-kaggle-parity` | Throughput gates |
| Performance | **production training throughput gate** (tier-2) | `make test-launch-hygiene-e2e-throughput` vs `launch-hygiene-e2e-baseline.json` | Validation-preset bisect, factorized sampler microbench (tier-1) |

Launch hygiene slowed tier-2 without breaking parity tests — do not treat “parity green” as throughput recovery or vice versa.

### Consolidation hygiene

After planning docs merge (PR #219): prune stale local branches whose upstream is gone; recover doc-only commits via `git reflog` + cherry-pick if a hard reset dropped them ([git-stash-recovery-after-parallel-branch-cleanup.md](git-stash-recovery-after-parallel-branch-cleanup.md)). Keep at most one full pytest suite repo-wide while integration worktrees run targeted gate captures.

## Why This Matters

Without a committed manifest + worktree discipline, throughput recovery becomes either a destructive revert or opaque cherry-picks with no record of which preset admitted each commit. Gitignored `outputs/` artifacts plus minified stdout create false “missing proof” signals unless operators know to `jq` the `--out` path in the worktree that ran the gate.

Baseline-first integration is the **primary** path; redesign and SSOT migration proceed in parallel only when they do not waive manifest dual gates.

## When to Apply

- Before cherry-picking learning commits onto a throughput anchor.
- When tier-2 numbers from `main` disagree with validation-preset bisect — re-run **tier-2 primary** at the integration head, record preset in manifest `candidates[]`.
- When onboarding agents to throughput vs parity investigations — read nomenclature RFC first.
- After stale-branch cleanup when multiple worktrees (`orbit_wars-throughput-anchor`, integration checkout) are active.

## Examples

**Manifest baseline_gates after U1 (committed):**

```json
"throughput_e2e": {
  "preset": "tier2_primary",
  "source": "docs/benchmarks/launch-hygiene-e2e-baseline.json",
  "artifact": "outputs/benchmarks/cherry-pick/anchor_throughput.json",
  "verdict": "admit",
  "gate_passed": true,
  "env_steps_per_sec": 9627.62
}
```

**Wrong — treating validation bisect as manifest admission:**

```bash
# HEAD ~299 env_steps/sec on --preset validation does NOT override tier-2 admit at anchor
uv run python scripts/issues_jax_30update_benchmark.py --preset validation ...
# → use for Phase 2 comet-era localization only
```

**Wrong — expecting artifact in main repo:**

```bash
ls docs/benchmarks/cherry-pick/anchor_throughput.json   # not committed
ls outputs/benchmarks/cherry-pick/anchor_throughput.json  # worktree-local, gitignored
```

## Related

- Validation-preset bisect (orthogonal measurement frame): [jax-validation-throughput-benchmark-and-bisect.md](jax-validation-throughput-benchmark-and-bisect.md)
- Tier-2 gate semantics and ablation tiebreaker: [launch-hygiene-learner-ablation-gate.md](../tooling-decisions/launch-hygiene-learner-ablation-gate.md)
- Requirements: `docs/brainstorms/2026-06-05-nuclear-cherry-pick-manifest-requirements.md`
- Implementation plan: `docs/plans/2026-06-05-002-feat-nuclear-cherry-pick-manifest-plan.md`
- Nomenclature RFC: `docs/nomenclature-rfc.md`
- Long-lived integration pattern (feature landings): [multi-branch-agent-merge-orchestration.md](multi-branch-agent-merge-orchestration.md)
