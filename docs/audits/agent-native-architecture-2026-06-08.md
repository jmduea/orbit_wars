# Agent-Native Architecture Review: Orbit Wars

**Date:** 2026-06-08  
**Scope:** `orbit_wars-integration` worktree; primary surface is `ow` CLI (`src/cli/`), not a web UI. Includes **new** `ow train colab` host (PR [#226](https://github.com/jmduea/orbit_wars/pull/226), merge `7f7a38c`).  
**References:** `ce-agent-native-audit` skill, `ce-agent-native-architecture` principles, `docs/AGENT_CAPABILITIES.md`, `tests/test_agent_capability_map.py`, `scripts/agent_context.py`, `docs/colab_runner.md`.  
**Prior audits:** [`agent-native-architecture-2026-06-02.md`](agent-native-architecture-2026-06-02.md) (84%), [`agent-native-architecture-2026-06-04.md`](agent-native-architecture-2026-06-04.md) (~87%).

---

## Executive summary

Orbit Wars remains **strongly agent-native at the operator layer**: humans and coding agents share the same `ow` CLI, the same `outputs/` tree (campaigns + colab_runner), tested capability-map regression, and session-start context injection. PR #226 landed **`ow train colab`** as a third train host with Kaggle-parity primitives (preflight, prepare, launch, status, sync, shortlist, stop), JSON-on-stdout contracts, and shared remote packaging (`src/orchestration/remote_package.py`, `remote_worker.py`).

**Overall agent-native score: 88%** (unweighted mean of eight principle percentages below; up from **84%** on 2026-06-02 and **~87%** on 2026-06-04).

Remaining gaps: `AGENTS.md` and `make agent-context` have not caught up to Colab; several admission/throughput benchmark primitives exist in `ow benchmark --help` but not the capability map; demoted `scripts/*` remain runnable; core RL and Colab session defaults stay code-defined (appropriate).

---

## Overall Score Summary

| Core Principle | Score | Percentage | Status |
|----------------|-------|------------|--------|
| Action Parity | 45/48 | 94% | ✅ |
| Tools as Primitives | 34/42 | 81% | ✅ |
| Context Injection | 8/9 | 89% | ✅ |
| Shared Workspace | 12/13 | 92% | ✅ |
| CRUD Completeness | 6/6 operator-adequate *(0/6 strict full CRUD)* | 100% / 0% | ✅ / by design |
| UI Integration (CLI observability) | 8/9 | 89% | ✅ |
| Capability Discovery | 7/8 | 88% | ✅ |
| Prompt-Native Features | 18/26 | 69% | ⚠️ |

**Overall Agent-Native Score: 88%**

### Status legend

- ✅ Excellent (80%+)
- ⚠️ Partial (50–79%)
- ❌ Needs Work (&lt;50%)

---

## Delta vs prior audits

| Change | 2026-06-02 → 2026-06-08 | Notes |
|--------|-------------------------|-------|
| **Colab train host** | **New** — closes remote long-run gap vs Kaggle-only | PR #226; `feat/colab-train-host` → `refactor/artifacts-metric-promotion-commit` |
| Action Parity | 93% → **94%** | Colab lifecycle + benchmark admission/phase primitives via `ow` |
| Tools as Primitives | 80% → **81%** | Six Colab leaf subcommands; `launch` remains workflow-shaped |
| Context Injection | 89% → **89%** | Unchanged; Colab pointer still missing from `agent-context` JSON |
| Shared Workspace | 100% → **92%** | `outputs/colab_runner/` is shared but secondary to `campaigns/` |
| Capability Discovery | 86% → **88%** | Colab in capability map + `ow train --help` + capability-map test |
| UI Integration | 88% → **89%** | Colab JSON stdout + stderr progress convention |
| Prompt-Native | 69% → **69%** | Colab GPU/timeout defaults in code; Hydra overrides work |
| SSOT spine in capabilities doc | — | Carried from 2026-06-04 refresh |
| `shape-calibrate` | Still planned | Not in CLI; test asserts unregistered |

---

## Principle 1: Action Parity

**Principle:** Whatever the user can do, the agent can do.

**Adaptation:** User actions ≈ `ow` / `make` operator commands. Parity = same commands in `docs/AGENT_CAPABILITIES.md`, `AGENTS.md`, and tested `--help` trees.

### User actions found

| Action | Location | Agent equivalent | Status |
|--------|----------|------------------|--------|
| Train (local Hydra) | `ow train` / bare `ow` | Same | ✅ |
| Print resolved config | `ow train print_resolved_config=true` | Same | ✅ |
| Kaggle train lifecycle | `ow train kaggle` | Same | ✅ |
| **Colab train lifecycle** | `ow train colab` | Same (preflight/prepare/launch/status/sync/shortlist/stop) | ✅ |
| Tournament eval | `ow eval tournament` | Same | ✅ |
| Artifact worker | `ow eval worker` | Same | ✅ |
| Eval queue status / results / cancel | `ow eval status`, `results`, `jobs cancel` | Same | ✅ |
| Package / Docker validate / submit | `ow eval package`, `submit` | Same | ✅ |
| Runs list/show/logs/watch/archive/checkpoint delete | `ow runs` | Same | ✅ |
| Promotion show/history/demote | `ow promote` | Same | ✅ |
| Benchmark training / sanity / gates | `ow benchmark` | Same | ✅ |
| Admission gate + throughput extract | `ow benchmark gate run admission`, `admission-throughput` | Same (`--help`; not capability-map row) | ✅ |
| Rollout phase profile / breakdown | `ow benchmark rollout-phase-*` | Same (`--help`) | ✅ |
| Env parity A/B | `ow benchmark env-parity-ab` | Same (`--help`) | ✅ |
| Preflight calibrate / seed / unified / qualifier | `ow benchmark calibrate*` | Same | ✅ |
| Tournament proof / learn-proof | `ow benchmark tournament-proof`, `learn-proof` | Same | ✅ |
| Planet Flow shortlist / noop smoke | `ow benchmark shortlist-planet-flow-sweep`, `planet-flow-noop-smoke` | Same | ✅ |
| Factorized sampler (tier-1 hygiene) | `ow benchmark factorized-sampler` | Same | ✅ |
| W&B/Kaggle sweep lifecycle | `ow sweep` | Same | ✅ |
| Generate sweep YAML | `ow make` | Same | ✅ |
| Bracket inspect | `ow eval bracket status` | Same | ✅ |
| Session context JSON | `make agent-context` | Same (+ Cursor hook) | ✅ |
| Fast/domain tests | `make test-fast`, `make test-domain-*` | Same | ✅ |
| Gate admission Makefile | `make gate-admission` | Same | ✅ |
| Colab one-time CLI install/auth | `uv tool install google-colab-cli`, `colab auth` | Documented in `docs/colab_runner.md`; not `ow` | ⚠️ |
| Kaggle `latest-checkpoint` | `ow train kaggle latest-checkpoint` | No Colab equivalent | ⚠️ |
| Direct Docker validate script | `scripts/validate_kaggle_docker_submission.py` | Demoted; stderr → `ow eval package` | ⚠️ |
| Direct artifact worker script | `scripts/run_artifact_worker.py` | Demoted; stderr → `ow eval worker` | ⚠️ |
| Env shaping calibration | `ow benchmark shape-calibrate` | Planned only | ❌ |
| Manual queue JSON edit | Filesystem | Anti-pattern | 🚫 N/A |
| W&B/Kaggle browser-only flows | External UIs | Not automatable via `ow` | 🚫 N/A |

### Score: **45/48 (94%)**

Excluded from denominator: 2 N/A (external web UIs). Colab **improves** parity vs 2026-06-02/04: agents gain a third remote train host with the same primitive decomposition as Kaggle (minus `latest-checkpoint`). Remaining gaps: external Colab auth bootstrap, demoted scripts, planned `shape-calibrate`.

### Recommendations

1. Add Colab copy-paste workflow to `AGENT_CAPABILITIES.md` (shortlist → launch → sync → `ow runs show` on synced path).
2. Add capability-map rows for `admission-throughput`, `rollout-phase-profile`, `gate run admission` (or a single “admission stack” row).
3. Optional `ow train colab latest-checkpoint` parity with Kaggle sweep winner resolution.
4. Stronger demoted-script guardrails if agents keep invoking `scripts/*`.

---

## Principle 2: Tools as Primitives

**Principle:** Tools provide capability, not behavior.

**Adaptation:** Leaf `ow` subcommands and `make` targets that delegate to a single `ow` command.

### Tool analysis (representative leaf paths)

| Command | Type | Reasoning |
|---------|------|-----------|
| `ow runs list/show/logs/watch/archive/checkpoint delete` | PRIMITIVE | Read/poll/archive/delete artifacts |
| `ow eval status/results/worker/jobs cancel/package/submit/tournament` | PRIMITIVE | Single artifact or eval operation |
| `ow promote show/history/demote` | PRIMITIVE | Read or single rollback mutation |
| `ow benchmark gate run/list`, `tournament-proof`, `training`, `sanity`, `factorized-sampler` | PRIMITIVE | One gate or one measurement |
| `ow benchmark admission-throughput`, `rollout-phase-profile`, `env-parity-ab` | PRIMITIVE | Post-hoc extract or offline profile |
| `ow train colab preflight/prepare/status/sync/stop/shortlist` | PRIMITIVE | Single Colab lifecycle step |
| `ow sweep create/status/list/cancel` | PRIMITIVE | Sweep lifecycle |
| `ow train` (local Hydra) | PRIMITIVE | Capability with config-defined behavior |
| `ow train colab launch` (default) | WORKFLOW | Provisions session + upload + exec bootstrap |
| `ow train kaggle launch` (default) | WORKFLOW | Package + push + poll kernel |
| `ow benchmark learn-proof` | WORKFLOW | Composes gate ladder; documents primitives |
| `ow benchmark calibrate` | WORKFLOW | Sweep + analyze + refresh thresholds |
| `ow benchmark calibrate-seed-scheduler` | WORKFLOW | Multi-run GPU sweep orchestration |
| `ow benchmark shortlist-planet-flow-sweep` | WORKFLOW | W&B fetch + ranking report |
| `ow train … artifacts=hybrid_promotion` | WORKFLOW (config) | Queues composite checkpoint_eval |

### Score: **34/42 (81%)**

Eight workflow-shaped entrypoints among ~42 classified leaf paths (up from ~40 in June audits). Colab adds **six primitives** and **one workflow** (`launch`); net primitive ratio similar to pre-Colab. Phase 3 guidance still applies: prefer `gate run`, `tournament-proof`, and Colab `preflight`/`sync` over composers.

### Recommendations

1. Agent docs: decompose Colab long runs into `prepare` → `launch` → `status` → `sync` when debugging; use monolithic `launch` only when appropriate.
2. Keep `learn-proof` as thin composer with `--print-primitives` (already shipped).
3. Split `calibrate` analyze-only path is documented — reinforce in agent loops.

---

## Principle 3: Context Injection

**Principle:** System/session context includes dynamic app state.

### Context types analysis

| Context type | Injected? | Location | Notes |
|--------------|-----------|----------|-------|
| Preflight thresholds | ✅ | `make agent-context` → `preflight` | From `docs/benchmarks/preflight-calibration.json` |
| Preflight gate ids | ✅ | `preflight.gates` | From `conf/benchmark/gates/*.yaml` |
| ROADMAP Now/Next | ✅ | `agent_context.py` | Excerpt only |
| Recent runs index | ✅ | `outputs/indexes/runs.jsonl` | Last N rows |
| Latest run eval queue | ✅ | `latest_run_eval` | Via `summarize_run_status` |
| Git branch | ✅ | `agent_context.py` | |
| Doc pointers | ✅ | `docs` keys in JSON | `AGENT_CAPABILITIES.md`, `AGENTS.md` |
| W&B sweep state | ✅ | `wandb_sweeps` | Recent sweep summary |
| Active GPU / process contention | ✅ | `gpu_contention` | Heuristic `pgrep` patterns |
| Cursor rules / AGENTS.md | ✅ | Workspace rules | Static + threshold block |
| Resolved Hydra config | ⚠️ | `make agent-context --resolved smoke` | Opt-in; default session omits |
| **Colab runner / active sessions** | ⚠️ | `outputs/colab_runner/sessions.json` | Exists on disk; not in agent-context JSON |
| Capability map / SSOT flowchart URL | ⚠️ | Docs only | Not embedded in session JSON |
| Full session chat history | ⚠️ | Cursor product | Out of repo scope |

### Score: **8/9 (89%)**

Strong session-start hook (`.cursor/hooks/session-start-agent-context.sh`). **Colab gap:** active Colab sessions and `docs/colab_runner.md` are not surfaced in default `agent-context` despite being operator-critical during long remote runs.

### Recommendations

1. Add `colab_runner` doc pointer and optional `sessions.json` summary to `build_context()` when `outputs/colab_runner/sessions.json` exists.
2. Embed SSOT flowchart URL + primitive/workflow tier one-liner in agent-context `docs` block.
3. Default `RESOLVED=smoke` in Cursor hook if Hydra drift is common.

---

## Principle 4: Shared Workspace

**Principle:** Agent and user work in the same data space.

### Data store analysis

| Data store | User access | Agent access | Shared? |
|------------|-------------|--------------|---------|
| `outputs/campaigns/*/runs/*` | CLI + FS | Same | ✅ |
| `outputs/colab_runner/synced/<campaign>/runs/*` | `ow train colab sync` + FS | Same (`ow runs show` after sync) | ✅ |
| `outputs/colab_runner/{kernel,launches.jsonl,sessions.json}` | Colab CLI + FS | Same | ✅ |
| `logs/*_jax.jsonl` | `ow runs logs` | Same | ✅ |
| `checkpoints/jax_ckpt_*.pkl` | Paths in manifests | Same | ✅ |
| `queue/optional_jobs/` | `ow eval status` | Same | ✅ |
| `evaluations/**/manifest.json` | `ow eval results` | Same | ✅ |
| `promoted/current_best/` | `ow promote show` | Same | ✅ |
| `outputs/indexes/runs.jsonl` | Index + `ow runs list` | Same | ✅ |
| `docs/benchmarks/*.json` | FS + calibrate | Same | ✅ |
| `.audit/`, `.cursor/hooks/state/` | Local gitignored | Same (not committed) | ✅ |
| Separate agent sandbox DB | — | None | ✅ |

### Score: **12/13 (92%)**

No isolated agent database. Colab synced trees are a **secondary** canonical path until operators copy or reference them like local campaigns; index may not list colab-only runs until sync.

### Recommendations

1. Document synced-run path convention in `AGENT_CAPABILITIES.md` (`outputs/colab_runner/synced/<campaign>/runs/<id>`).
2. Optional: append sync events to `runs.jsonl` index on successful `colab sync`.

---

## Principle 5: CRUD Completeness

**Principle:** Every entity has full CRUD.

**Adaptation:** RL artifacts are append-heavy; score operator-adequate coverage separately.

### Entity CRUD analysis

| Entity | Create | Read | Update | Delete | Operator-adequate |
|--------|--------|------|--------|--------|-------------------|
| Train run | `ow train` | `ow runs *` | — | `runs archive` | ✅ |
| Checkpoint | training | paths, tournament, package | — | `runs checkpoint delete` | ✅ |
| Eval queue job | training/hybrid | `ow eval status` | cancel | `jobs cancel` | ✅ |
| Eval result (checkpoint_eval) | worker | `results list/show` | — | — | ✅ R |
| Campaign promotion | metrics/hybrid | `ow promote show/history` | demote | demote | ✅ |
| W&B/Kaggle sweep | `ow sweep create` | status/list | — | `sweep cancel` (wandb) | ✅ |
| **Colab remote session** | `ow train colab launch` | `colab status` | — | `colab stop` | ✅ C+R+D |

### Scores

- **Full CRUD:** **0/7 (0%)** — by design for immutable train artifacts.
- **Operator-adequate:** **7/7 (100%)** — Colab adds session stop; still no hard-delete run trees.

### Recommendations

1. Document Colab session teardown in capability map CRUD boundaries section.
2. Treat `ow promote demote` as canonical promotion rollback (unchanged).

---

## Principle 6: UI Integration (CLI observability)

**Principle:** Agent actions immediately reflected in UI.

**Adaptation:** No web product UI. Substitutes: terminal banners, JSON stdout, JSONL logs, poll/watch CLIs.

### Agent action → observability analysis

| Agent action | Mechanism | Immediate? | Notes |
|--------------|-----------|------------|-------|
| Local train start/complete | `orbit_train_*` lines | ✅ | `src/jax/train/loop.py` |
| Colab preflight/launch/sync | JSON stdout; stderr tails | ✅ | `src/orchestration/colab_runner.py` |
| Artifact worker | `artifact_worker_started` + queue logs | ✅ | |
| Queue progress | `ow eval status --watch` | ✅ | Poll |
| Run metrics | `ow runs watch`, `ow runs logs` | ✅ | Poll/tail |
| Benchmark gate | JSON stdout + `--out` | ✅ | |
| Hybrid promotion | status + `results show` manifest | ✅ | Legacy funnel |
| Colab remote progress | `colab exec` stderr stream | ⚠️ | No `status --watch`; manual poll |
| W&B metrics | External dashboard | ⚠️ | Not in-repo |

### Score: **8/9 (89%)**

Colab follows the established **JSON stdout / stderr progress** convention (R6 in colab plan). One gap: no `--watch` on `ow train colab status` comparable to `ow eval status --watch`.

### Recommendations

1. Add `ow train colab status --watch --idle-exit-seconds` when remote session polling becomes common.
2. Keep anti-pattern docs: no `tail` pipes on long `colab launch` (already in `docs/colab_runner.md`).

---

## Principle 7: Capability Discovery

**Principle:** Users/agents can discover what the system can do.

### Discovery mechanism analysis

| Mechanism | Exists? | Location | Quality |
|-----------|---------|----------|---------|
| Onboarding | ✅ | `docs/ONBOARDING.md`, `docs/AGENT_CAPABILITIES.md` | High |
| Help documentation | ✅ | `uv run ow --help`, per-subcommand help, `make help` | High |
| Colab operator doc | ✅ | `docs/colab_runner.md`, `ow train --help` | High (not in `AGENTS.md`) |
| Agent rules self-describe | ✅ | `AGENTS.md`, `.cursor/rules/orbit-wars.mdc` | High (Colab absent in AGENTS.md) |
| Suggested prompts / copy-paste | ✅ | `AGENT_CAPABILITIES.md` § Copy-paste agent prompts | High (no Colab prompt yet) |
| Empty-state guidance | ✅ | `ow eval` / `ow benchmark` print subcommand menus | Good |
| Tested capability map | ✅ | `tests/test_agent_capability_map.py` | High — includes `colab` host token |
| Capability hints in product UI | 🚫 N/A | — | CLI-only repo |
| Slash commands | 🚫 N/A | — | Use `ow` not `/tools` |

### Score: **7/8 (88%)**

Colab is registered in capability map, `ow train --help`, and `_EXTRA_NESTED_TOKENS` in the capability-map test. Gaps: `AGENTS.md` silent on Colab; no copy-paste Colab agent prompt; admission/phase benchmark commands only in `ow benchmark --help`.

### Recommendations

1. Add Colab section to `AGENTS.md` (install, preflight → shortlist → launch → sync).
2. Add copy-paste prompt: “Run Colab long train after W&B preflight shortlist.”
3. Extend capability map with admission-stack and rollout-phase rows.

---

## Principle 8: Prompt-Native Features

**Principle:** Operator-visible behavior defined in prompts/config, not hardcoded orchestration.

### Feature definition analysis (operator layer)

| Feature | Defined in | Type | Notes |
|---------|------------|------|-------|
| Preflight gate recipes | `conf/benchmark/gates/*.yaml` | PROMPT/YAML | Authoritative |
| Gate thresholds | `docs/benchmarks/preflight-calibration.json` | DATA + docs | Calibrate refreshes AGENTS block |
| Training hyperparams / opponents | Hydra `conf/` | YAML | `print_resolved_config` |
| Colab Hydra overrides | CLI → `worker-env.json` | YAML/CLI | Same token rules as Kaggle |
| W&B preflight / SSOT sweep | `conf/wandb_sweep/*` | YAML | |
| Hybrid promotion funnel | Hydra `artifacts=hybrid_promotion` | YAML | Legacy |
| Agent task prompts | `docs/AGENT_CAPABILITIES.md` | PROMPT | |
| Colab default GPU / timeout | `colab_runner.py`, `train_hosts.py` | CODE | T4, 86400s on default launch |
| Core PPO / rollout / env | `src/jax/*` | CODE | Appropriate |
| Feature encoding schema | `src/features/` | CODE | v2 unified |
| `learn-proof` ladder composition | Python composer | CODE | Thin; lists primitives |
| Dual SSOT vs hybrid/Gate5 docs | Markdown + code paths | MIXED | Until SSOT U8 teardown |

### Score: **18/26 (69%)**

Colab long-run **behavior** is tunable via Hydra overrides; **infrastructure defaults** (GPU, timeout, bootstrap) remain code-native — same pattern as Kaggle accelerator defaults. Core RL and dual-spine legacy keep score in ⚠️ band.

### Recommendations

1. Expose Colab defaults in `conf/` or documented Hydra profile (`training=colab_long`) when profiles stabilize.
2. Continue YAML-only gate extension policy.
3. SSOT U8 teardown to collapse dual-spine prompt-native debt.

---

## Top 10 Recommendations by Impact

| Priority | Action | Principle | Effort |
|----------|--------|-----------|--------|
| 1 | Add Colab workflow to `AGENTS.md` + copy-paste prompt in `AGENT_CAPABILITIES.md` | Discovery / Parity | Low |
| 2 | Extend `make agent-context` with `colab_runner` pointer + optional active sessions | Context | Low |
| 3 | Capability-map rows: admission gate stack, `admission-throughput`, `rollout-phase-profile` | Discovery / Parity | Low |
| 4 | `ow train colab status --watch` for remote session polling | UI (CLI) | Medium |
| 5 | Agent docs: prefer Colab primitives + local `ow eval`/gates after `sync` | Tools / Parity | Low |
| 6 | SSOT U5–U8 + R29 legacy teardown | Prompt-native | Large |
| 7 | Default `RESOLVED=smoke` in session hook | Context | Low |
| 8 | Optional `ow train colab latest-checkpoint` (Kaggle parity) | Parity | Medium |
| 9 | Stronger demoted-script exit codes | Parity | Low |
| 10 | Ship or de-map `ow benchmark shape-calibrate` | Parity | Medium |

---

## What's Working Excellently

1. **Colab train host (PR #226)** — Third `ow train` host with JSON stdout, stderr progress, shared remote packaging, and tests (`test_colab_cli.py`, `test_colab_runner.py`, `test_cli_train_hosts.py`).
2. **Tested capability map** — `tests/test_agent_capability_map.py` registers `colab` alongside `kaggle` and `local`.
3. **Shared workspace** — Single filesystem for campaigns, colab_runner artifacts, and eval outputs; no agent sandbox.
4. **Primitive eval/runs/benchmark surface** — `ow eval status --watch`, `gate run`, `admission-throughput`, `runs watch`.
5. **Session bootstrap** — `make agent-context` + Cursor hook with thresholds, gate ids, GPU contention hint.

---

## Colab-specific agent-native assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Action parity vs local/Kaggle | ✅ Strong | Same `ow train` router; seven subcommands |
| Tools as primitives | ✅ Good | `launch` is workflow; other subcommands atomic |
| JSON agent loops | ✅ | All subcommands print JSON on stdout |
| Shared workspace after sync | ✅ | `outputs/colab_runner/synced/<campaign>/` |
| Discovery | ⚠️ Partial | Help + capability map; missing AGENTS.md + agent-context |
| U6 operator smoke | ⚠️ Blocked | Full GPU launch deferred per PR #226 test plan |

**Colab improved action parity vs 2026-06-02 audit:** **Yes** — prior audits had only local + Kaggle remote paths; Colab closes the long-run remote GPU workflow without Kaggle embedded-payload constraints.

---

## CLI-only / N/A notes

| Principle | Note |
|-----------|------|
| UI Integration | **N/A** for web UI; scored on CLI/log/JSON polling. |
| Capability Discovery | Product UI hints and slash commands **N/A**. |
| CRUD | Full four-op CRUD **N/A** for immutable train artifacts; operator-adequate scoring used. |
| Action Parity | External `colab auth`, W&B/Kaggle browser actions **N/A** or documented one-time setup. |

---

## Related artifacts

- Colab plan: `docs/plans/2026-06-07-005-feat-colab-train-host-plan.md`
- Colab operator doc: `docs/colab_runner.md`
- Prior audits: `docs/audits/agent-native-architecture-2026-06-02.md`, `docs/audits/agent-native-architecture-2026-06-04.md`
- Phase status: `docs/agent-native-phase3-status.md`
- Capability test: `tests/test_agent_capability_map.py`

---

## Audit metadata

- **Pass (2026-06-08):** Read-only code/doc review on `orbit_wars-integration` at PR #226 merge `7f7a38c`; verified `uv run ow train --help`, `docs/AGENT_CAPABILITIES.md`, `scripts/agent_context.py`, `src/cli/train_hosts.py`, `src/orchestration/colab_runner.py`, `tests/test_agent_capability_map.py`. No GPU train or Colab launch executed.
- **PR #226 state:** **MERGED** into `refactor/artifacts-metric-promotion-commit` at `2026-06-08T01:50:29Z` — https://github.com/jmduea/orbit_wars/pull/226
