# Agent-Native Architecture Review: Orbit Wars

**Date:** 2026-06-02  
**Last refreshed:** 2026-06-03 (post–PR [#184](https://github.com/jmduea/orbit_wars/pull/184), merge `191fef3`)  
**Scope:** Hydra + JAX PPO RL project; primary surface is `ow` CLI (`src/cli/`), not a web UI.  
**References:** `ce-agent-native-audit` skill, `ce-agent-native-architecture` principles, `docs/AGENT_CAPABILITIES.md`, `docs/audits/agent-native-status.md`, canonical operator learning `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`, Phase 1 precursor `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`.

**Refresh note:** Original audit was read-only (no GPU train). Post-#184, seed-scheduler calibration is **complete** (`training.reseed_every_updates: 50`, `docs/benchmarks/seed-scheduler-calibration.json`), Phase 2/3 operator primitives and capability-map test shipped; scores below are updated where evidence changed — overall **78%** mean is still representative.

---

## Executive summary

Orbit Wars is **strongly agent-native at the operator layer**: humans and coding agents share the same `ow` CLI, the same `outputs/campaigns/` tree, and documented primitive-vs-workflow tiers. Phase 3 shipped composable benchmark gates (`conf/benchmark/gates/*.yaml`), `ow benchmark gate run`, `ow sweep`, and Cursor session-start context injection. PR #184 closed deferred audit gaps: **seed interval 50**, `ow runs archive` / `ow runs checkpoint delete`, `ow sweep cancel`, `ow benchmark factorized-sampler`, richer `make agent-context` (gate ids, W&B sweep summary, GPU contention hint), and `tests/test_agent_capability_map.py`.

**Overall agent-native score: 84%** (unweighted mean of eight principle percentages below; was **78%** on 2026-06-02 before PR #184 context/CRUD/parity closes).

Remaining gaps are predictable for a CLI-only RL repo: no live UI integration (substituted by log/JSON polling), intentionally partial strict CRUD on immutable train artifacts, demoted `scripts/*` entrypoints still runnable with stderr hints, and core RL mechanics remaining code-defined (appropriate).

---

## Overall Score Summary

| Core Principle | Score | Percentage | Status |
|----------------|-------|------------|--------|
| Action Parity | 43/46 | 93% | ✅ |
| Tools as Primitives | 32/40 | 80% | ✅ |
| Context Injection | 8/9 | 89% | ✅ |
| Shared Workspace | 12/12 | 100% | ✅ |
| CRUD Completeness | 0/6 full CRUD; 6/6 operator-adequate | 0% / 100% | ⚠️ / ✅ |
| UI Integration | 7/8 (CLI observability) | 88% | ✅ (N/A UI) |
| Capability Discovery | 6/7 | 86% | ✅ |
| Prompt-Native Features | 18/26 operator-tunable | 69% | ⚠️ |

**Overall Agent-Native Score: 84%**

### Status legend

- ✅ Excellent (80%+)
- ⚠️ Partial (50–79%)
- ❌ Needs Work (&lt;50%)

---

## Principle 1: Action Parity

**Principle:** Whatever the user can do, the agent can do.

**Adaptation:** User actions ≈ `ow` / `make` operator commands (no React UI). Parity = same commands documented for agents in `docs/AGENT_CAPABILITIES.md` and `AGENTS.md`.

### User actions found (canonical operator surface)

| Action | Location | Agent equivalent | Status |
|--------|----------|------------------|--------|
| Train (local Hydra) | `ow train` / bare `ow` | Same | ✅ |
| Print resolved config | `ow train print_resolved_config=true` | Same | ✅ |
| Kaggle preflight / prepare / launch / status / sync / shortlist / latest ckpt | `ow train kaggle …` | Same | ✅ |
| Tournament eval | `ow eval tournament` | Same | ✅ |
| Artifact worker | `ow eval worker` | Same (prefer over `scripts/run_artifact_worker.py`) | ✅ |
| Eval queue status | `ow eval status` | Same | ✅ |
| Eval results list/show | `ow eval results` | Same | ✅ |
| Cancel queued jobs | `ow eval jobs cancel` | Same | ✅ |
| Package / Docker validate | `ow eval package --validate-docker` | Same | ✅ |
| Kaggle submit | `ow eval submit` | Same | ✅ |
| List/show/tail/watch runs | `ow runs` | Same | ✅ |
| Promotion show/history/demote | `ow promote` | Same | ✅ |
| Benchmark training throughput | `ow benchmark training` | Same | ✅ |
| Preflight sanity | `ow benchmark sanity` / `make preflight-sanity` | Same | ✅ |
| Preflight gates (noop/random/curriculum) | `ow benchmark gate run <id>` | Same | ✅ |
| Gate 5 tournament proof | `ow benchmark tournament-proof` | Same | ✅ |
| Preflight calibrate | `ow benchmark calibrate` / `make preflight-calibrate` | Same | ✅ |
| Learn-proof ladder (composer) | `ow benchmark learn-proof` / `make preflight-learn-proof` | Same | ✅ |
| Seed-scheduler calibration sweep | `ow benchmark calibrate-seed-scheduler` | Same (GPU; defer when busy) | ✅ |
| Planet-flow shortlist / noop smoke | `ow benchmark` subcommands | Same | ✅ |
| W&B/Kaggle sweep create/status/list | `ow sweep` | Same | ✅ |
| Generate sweep YAML | `ow make` | Same | ✅ |
| Fast/domain tests | `make test-fast`, `make test-domain-*` | Same | ✅ |
| Session context JSON | `make agent-context` | Same (+ Cursor hook) | ✅ |
| Tier-1 launch hygiene microbench | `make test-launch-hygiene-throughput` → `ow benchmark factorized-sampler` | Same (script stderr-hints `ow`) | ✅ |
| Archive run tree | `ow runs archive` | Same (`--dry-run` / `--confirm`) | ✅ |
| Delete checkpoint file | `ow runs checkpoint delete` | Same | ✅ |
| Cancel W&B sweep runs | `ow sweep cancel --backend wandb` | Same | ✅ |
| Direct Docker validate script | `scripts/validate_kaggle_docker_submission.py` | Demoted; stderr → `ow eval package` | ⚠️ |
| Direct artifact worker script | `scripts/run_artifact_worker.py` | Demoted; stderr → `ow eval worker` | ⚠️ |
| Manual queue JSON edit | Filesystem | No CLI; anti-pattern | 🚫 N/A |
| W&B/Kaggle web-only flows | External UIs | Not automatable via `ow` | 🚫 N/A |

### Score: **43/46 (93%)**

Excluded from denominator: 2 N/A (external web UIs). Remaining gaps: demoted `scripts/*` paths still runnable (stderr `prefer:` hints shipped in #184); no hard-delete run primitive beyond archive.

### Recommendations

1. ~~Capability map in `docs/AGENT_CAPABILITIES.md`~~ — **shipped** (#184 + `tests/test_agent_capability_map.py`).
2. ~~`ow benchmark factorized-sampler` / `ow runs archive`~~ — **shipped** (#184).
3. Optional: stronger deprecation (exit non-zero or remove script `main`) for demoted validate/worker scripts if agents keep invoking them.
4. Optional `make agent-context RESOLVED=smoke` in default Cursor hook (flag exists; not always on).

---

## Principle 2: Tools as Primitives

**Principle:** Tools provide capability, not behavior (atomic ops; agents compose workflows).

**Adaptation:** “Tools” = leaf `ow` subcommands (and `make` targets that delegate to a single `ow` command).

### Tool analysis (leaf commands)

| Command | Type | Reasoning |
|---------|------|-----------|
| `ow runs list/show/logs/watch/archive/checkpoint delete` | PRIMITIVE | Read/poll/archive/delete artifacts |
| `ow eval status/results/worker/jobs cancel/package/submit/tournament` | PRIMITIVE | Single artifact or eval operation |
| `ow promote show/history/demote` | PRIMITIVE | Read or single rollback mutation |
| `ow benchmark gate run/list`, `tournament-proof`, `training`, `sanity` | PRIMITIVE | One gate or one measurement |
| `ow sweep create/status/list/cancel` | PRIMITIVE | Sweep lifecycle (+ W&B cancel) |
| `ow train` (Hydra) | PRIMITIVE | Capability with config-defined behavior |
| `ow benchmark learn-proof` | WORKFLOW | Composes gate ladder + documents primitives in report |
| `ow benchmark calibrate` | WORKFLOW | May run sweep + derive thresholds + refresh AGENTS.md |
| `ow benchmark calibrate-seed-scheduler` | WORKFLOW | Multi-run GPU sweep orchestration |
| `ow benchmark shortlist-planet-flow-sweep` | WORKFLOW | W&B fetch + ranking report |
| `ow train … artifacts=hybrid_promotion` | WORKFLOW (config) | Queues composite checkpoint_eval pipeline |

### Score: **32/40 (80%)**

Eight of forty classified leaf operator entrypoints are workflow-shaped; the rest are primitives. Phase 3 intentionally keeps `learn-proof` as a **thin composer** (see `run_learn_proof_cli` docstring in `src/cli/benchmark.py`).

### Recommendations

1. Keep agent docs emphasizing **primitive tier** (`gate run`, `tournament-proof`) over `learn-proof` for targeted loops.
2. Split `calibrate` into `calibrate analyze` vs `calibrate sweep` if agents need analyze-only without implicit training launches.
3. Avoid new monolithic `ow benchmark *-proof` composers without listing composed primitives in JSON output (pattern already in learn-proof report).

---

## Principle 3: Context Injection

**Principle:** System/session context includes dynamic app state.

### Context types analysis

| Context type | Injected? | Location | Notes |
|--------------|-----------|----------|-------|
| Preflight thresholds | ✅ | `make agent-context` → `preflight` | From `docs/benchmarks/preflight-calibration.json` |
| ROADMAP Now/Next | ✅ | `agent_context.py` | Excerpt only |
| Recent runs index | ✅ | `outputs/indexes/runs.jsonl` | Last N rows |
| Latest run eval queue | ✅ | `latest_run_eval` | Via `summarize_run_status` |
| Git branch | ✅ | `agent_context.py` | |
| Doc pointers | ✅ | `docs` keys in JSON | |
| Cursor rules / AGENTS.md | ✅ | Workspace rules | Static + threshold block |
| Preflight gate ids | ✅ | `preflight.gates` in JSON | From `conf/benchmark/gates/*.yaml` |
| Resolved Hydra config | ⚠️ | `make agent-context --resolved smoke` | Opt-in snapshot; default session omits |
| Active GPU / terminal contention | ✅ | `gpu_contention` | Heuristic from terminals + recent GPU runs |
| User test-tier preference | ⚠️ | AGENTS.md learned prefs | Not in `agent-context` JSON |
| W&B sweep state | ✅ | `wandb_sweeps` | Summary of recent sweep activity |
| Full session chat history | ⚠️ | Cursor product | Out of repo scope |

### Score: **8/9 (89%)**

Strong session-start hook (`.cursor/hooks/session-start-agent-context.sh`); gate ids, sweep summary, and GPU contention ship in `scripts/agent_context.py` (#184). Remaining gap: resolved-config snapshot is opt-in, not default.

### Recommendations

1. Enable `make agent-context RESOLVED=smoke` in the default Cursor hook if agents routinely need Hydra defaults without a train subprocess.
2. Document in `docs/CURSOR.md` that agents should check terminals folder before GPU commands (contention hint is advisory — reinforce in hook failure text).
3. ~~Gate ids in agent-context~~ — **shipped** (`preflight.gates.gate_ids`).

---

## Principle 4: Shared Workspace

**Principle:** Agent and user work in the same data space.

### Data store analysis

| Data store | User access | Agent access | Shared? |
|------------|-------------|--------------|---------|
| `outputs/campaigns/*/runs/*` | CLI + FS | Same | ✅ |
| `logs/*_jax.jsonl` | `ow runs logs` | Same | ✅ |
| `checkpoints/jax_ckpt_*.pkl` | Paths in manifests | Same | ✅ |
| `queue/optional_jobs/` | `ow eval status` | Same | ✅ |
| `evaluations/**/manifest.json` | `ow eval results` | Same | ✅ |
| `promoted/current_best/` | `ow promote show` | Same | ✅ |
| `outputs/indexes/runs.jsonl` | Index + `ow runs list` | Same | ✅ |
| `docs/benchmarks/*.json` | FS + calibrate | Same | ✅ |
| `.audit/`, `.cursor/hooks/state/` | Local gitignored | Same (not committed) | ✅ |
| Separate agent sandbox DB | — | None | ✅ (no anti-pattern) |

### Score: **12/12 (100%)**

No isolated agent database; artifacts are filesystem-canonical.

### Recommendations

1. Continue policy: new features write under `outputs/campaigns/` with indexed manifests, not ad-hoc `/tmp`-only state.
2. Keep `.audit/` gitignored; document that agents must not expect remote parity for audit dirs.

---

## Principle 5: CRUD Completeness

**Principle:** Every entity has full Create, Read, Update, Delete.

**Adaptation:** RL artifacts are append-heavy; full CRUD is often **intentionally incomplete**. Score both strict CRUD and operator-adequate coverage.

### Entity CRUD analysis

| Entity | Create | Read | Update | Delete | Strict CRUD | Operator-adequate |
|--------|--------|------|--------|--------|-------------|-------------------|
| Train run | `ow train` | `ow runs *` | — (immutable) | `runs archive` | ❌ | ✅ C+R+D(archive) |
| Checkpoint | training | paths, tournament, package | — | `runs checkpoint delete` | ❌ | ✅ R+D |
| Eval queue job | training/hybrid | `ow eval status` | cancel | `jobs cancel` | ❌ | ✅ C+R+U/D |
| Eval result (checkpoint_eval) | worker | `results list/show` | — | — | ❌ | ✅ R |
| Campaign promotion | metrics/hybrid | `ow promote show/history` | demote | demote | ❌ | ✅ R+U |
| W&B/Kaggle sweep | `ow sweep create` | status/list | — | `sweep cancel` (wandb) | ❌ | ✅ C+R+D |

### Scores

- **Full CRUD:** **0/6 (0%)** — no entity exposes four CLI operations (by design).
- **Operator-adequate (create + read + targeted mutation where needed):** **6/6 (100%)** — archive, checkpoint delete, and W&B sweep cancel shipped in #184.

### Recommendations

1. Document in `AGENT_CAPABILITIES.md` that **hard delete** for run trees remains archive-first (`ow runs archive`), not blind `rm` — already in capability map.
2. ~~`ow sweep cancel`~~ — **shipped** (W&B backend; Kaggle may still need UI).
3. Treat `ow promote demote` as the canonical “delete promotion pointer” (already shipped).

---

## Principle 6: UI Integration

**Principle:** Agent actions immediately reflected in UI.

**Adaptation:** **No live product UI.** Substitutes: terminal banners, JSON stdout, JSONL logs, poll/watch CLIs, external W&B.

### Agent action → observability analysis

| Agent action | Mechanism | Immediate? | Notes |
|--------------|-----------|------------|-------|
| Train start/complete | `orbit_train_*` lines | ✅ | `src/jax/train/loop.py` |
| Artifact worker | `artifact_worker_started` + queue logs | ✅ | |
| Queue progress | `ow eval status --watch` | ✅ | Poll |
| Run metrics | `ow runs watch`, `ow runs logs` | ✅ | Poll/tail |
| Benchmark gate | JSON stdout + `--out` | ✅ | |
| Hybrid promotion | status + `results show` manifest | ✅ | Submit-valid contract |
| W&B metrics | External dashboard | ⚠️ | Not in-repo |
| Local replay HTML | Files on disk | ⚠️ | Inspect-only per docs |

### Score: **7/8 (88%)** — **N/A for web UI**

One gap: W&B/Kaggle dashboards are outside repo control; agents rely on CLI + JSONL.

### Recommendations

1. Keep `--watch` / `--idle-exit-seconds` on eval status (already documented).
2. Do not treat local replay HTML as submit-valid without canonical poll (already in decision tree).

---

## Principle 7: Capability Discovery

**Principle:** Users/agents can discover what the system can do.

### Discovery mechanism analysis

| Mechanism | Exists? | Location | Quality |
|-----------|---------|----------|---------|
| Onboarding | ✅ | `docs/ONBOARDING.md`, `docs/AGENT_CAPABILITIES.md` | High |
| Help documentation | ✅ | `uv run ow --help`, per-subcommand help, `make help` | High |
| Capability hints in UI | 🚫 N/A | — | CLI-only repo |
| Agent rules self-describe | ✅ | `AGENTS.md`, `.cursor/rules/orbit-wars.mdc` | High |
| Suggested prompts / copy-paste | ✅ | `AGENT_CAPABILITIES.md` § Copy-paste agent prompts | High |
| Empty-state guidance | ✅ | `ow eval` / `ow benchmark` print subcommand menus | Good |
| Slash commands | 🚫 N/A | — | Use `ow` not `/tools` |

### Score: **6/7 (86%)** (denominator excludes N/A UI hints)

### Recommendations

1. Add `uv run ow runs --help` cross-link to hybrid promotion decision tree (if not already in `print_eval_help`).
2. ~~Capability map in `AGENT_CAPABILITIES.md`~~ — **shipped** with `tests/test_agent_capability_map.py`.
3. ~~Gate ids in agent-context~~ — **shipped**; keep `ow benchmark gate --list` in operator docs.

---

## Principle 8: Prompt-Native Features

**Principle:** Operator-visible behavior defined in prompts/config, not hardcoded orchestration.

### Feature definition analysis (operator layer)

| Feature | Defined in | Type | Notes |
|---------|------------|------|-------|
| Preflight gate recipes | `conf/benchmark/gates/*.yaml` | PROMPT/YAML | Phase 3 authoritative |
| Gate thresholds | `docs/benchmarks/preflight-calibration.json` | DATA + docs | Calibrate refreshes AGENTS block |
| Training hyperparams / opponents | Hydra `conf/` | YAML | `print_resolved_config` |
| Shield/task/reward modes | Hydra | YAML | |
| Hybrid promotion funnel | Hydra `artifacts=hybrid_promotion` | YAML | |
| Agent task prompts | `docs/AGENT_CAPABILITIES.md` | PROMPT | |
| Submit-valid decision tree | Markdown + mermaid | PROMPT | |
| Core PPO / rollout / env | `src/jax/*` | CODE | Appropriate |
| Feature encoding schema | `src/features/` | CODE | v2 unified |
| Preflight verdict logic | `src/jax/preflight.py` + loader | CODE | Recipes in YAML |
| `learn-proof` ladder composition | Python composer | CODE | Thin; lists primitives in JSON |
| Metric promotion rules | Python + config | CODE | |

### Score: **18/26 (69%)** operator-tunable vs code-defined for the same surface area

Core RL remains code-native (expected). Operator gates and Hydra paths are strongly prompt/YAML-native post–Phase 3.

### Recommendations

1. Extend YAML for any remaining gate override tuples still only in Python (per Phase 3 goal — verify `preflight_gate_loader.py` coverage).
2. Add new benchmark gates via YAML only (document in `AGENT_CAPABILITIES.md`).
3. Keep copy-paste agent prompts updated when CLI changes.

---

## Top 10 Recommendations by Impact

| Priority | Action | Principle | Effort |
|----------|--------|-----------|--------|
| 1 | Agent docs: prefer `gate run` / `tournament-proof` over `learn-proof` in loops | Tools as Primitives | Low |
| 2 | Split `ow benchmark calibrate` analyze vs sweep clarity for agents (analyze-only flags exist) | Tools as Primitives | Low |
| 3 | Default `make agent-context RESOLVED=smoke` in Cursor hook (optional today) | Context Injection | Low |
| 4 | Document GPU/terminal contention in hook failure text (hint exists in JSON) | Context Injection | Low |
| 5 | Stronger demoted-script guardrails (non-zero exit) if agents keep calling `scripts/*` | Action Parity | Low |
| 6 | Extend YAML for any remaining gate overrides only in Python | Prompt-Native | Medium |
| 7 | Launch hygiene Phase B / tier-2 throughput (ROADMAP Later) | Performance | High |
| 8 | Planet Flow U7 relaunch after reachability mask | Operator GPU | High |
| 9 | ~~Seed-scheduler calibration + interval 50~~ | — | **Done** (#184) |
| 10 | ~~Capability map + factorized-sampler + archive/checkpoint/sweep cancel~~ | — | **Done** (#184) |

---

## What's Working Excellently

1. **Shared workspace (100%)** — Single `outputs/campaigns/` tree; no agent sandbox.
2. **Action parity (~91%)** — Same `ow` CLI for humans and agents; Makefile preflight aliases delegate to `ow`.
3. **Phase 3 primitives** — `ow benchmark gate run`, `tournament-proof`, YAML gates, thin `learn-proof` composer.
4. **Capability discovery (~86%)** — `AGENT_CAPABILITIES.md`, `AGENTS.md`, layered `--help`, copy-paste prompts, session-start hook.
5. **CLI observability (~88%)** — `runs watch`, `eval status --watch`, train banners, JSON manifests for submit-valid proof.

---

## CLI-only / N/A notes

| Principle | Note |
|-----------|------|
| UI Integration | **N/A** for web UI; scored on CLI/log polling substitutes. |
| Capability Discovery | UI hints and slash commands **N/A**; docs + `ow --help` are canonical. |
| CRUD | Full four-op CRUD **N/A** for immutable train artifacts; use operator-adequate scoring. |
| Action Parity | External W&B/Kaggle browser actions **N/A**; `ow eval submit` / sweeps cover agent paths. |

---

## Related artifacts

- **Canonical learning (calibration + phase-2 CLI):** `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`
- Phase 1 precursor: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`
- Operator status: `docs/audits/agent-native-status.md`
- Completed plans: `docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md`, `docs/plans/2026-06-02-015-feat-agent-native-audit-gaps-plan.md`, `docs/plans/2026-06-02-016-feat-agent-native-deferred-crud-plan.md`, `docs/plans/2026-06-02-017-feat-seed-u2-u3-capability-map-plan.md`
- Plan backlog (items 1–4 shipped): `docs/plans/2026-06-02-agent-native-phase3-refactors.md`

---

## Audit metadata

- **Original pass (2026-06-02):** `uv run ow --help` (read-only); file/code review of `src/cli/`, `scripts/agent_context.py`, `.cursor/hooks/`, Makefile preflight targets. No GPU train.
- **Refresh pass (2026-06-03):** Verified PR #184 merge `191fef3` — `reseed_every_updates: 50`, `tests/test_agent_capability_map.py`, `ow runs archive`, `ow runs checkpoint delete`, `ow sweep cancel`, `ow benchmark factorized-sampler`, `agent_context` gate/sweep/GPU fields; read `docs/audits/agent-native-status.md`.
