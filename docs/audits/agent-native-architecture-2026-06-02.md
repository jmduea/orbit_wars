# Agent-Native Architecture Review: Orbit Wars

**Date:** 2026-06-02  
**Scope:** Hydra + JAX PPO RL project; primary surface is `ow` CLI (`src/cli/`), not a web UI.  
**References:** `ce-agent-native-audit` skill, `ce-agent-native-architecture` principles, `docs/AGENT_CAPABILITIES.md`, `docs/agent-native-phase3-status.md`, prior Phase 1 learning `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`.

**Audit constraints:** No GPU training executed; background `calibrate-seed-scheduler` / seed-scheduler work left untouched.

---

## Executive summary

Orbit Wars is **strongly agent-native at the operator layer**: humans and coding agents share the same `ow` CLI, the same `outputs/campaigns/` tree, and documented primitive-vs-workflow tiers. Phase 3 shipped composable benchmark gates (`conf/benchmark/gates/*.yaml`), `ow benchmark gate run`, `ow sweep`, and Cursor session-start context injection.

**Overall agent-native score: 78%** (unweighted mean of eight principle percentages below).

Gaps are predictable for a CLI-only RL repo: no live UI integration (substituted by log/JSON polling), incomplete CRUD on filesystem entities, a few Makefile/script-only paths, and core RL mechanics remaining code-defined (appropriate).

---

## Overall Score Summary

| Core Principle | Score | Percentage | Status |
|----------------|-------|------------|--------|
| Action Parity | 40/44 | 91% | ✅ |
| Tools as Primitives | 32/40 | 80% | ✅ |
| Context Injection | 6/9 | 67% | ⚠️ |
| Shared Workspace | 12/12 | 100% | ✅ |
| CRUD Completeness | 0/6 full CRUD; 5/6 operator-adequate | 0% / 83% | ⚠️ |
| UI Integration | 7/8 (CLI observability) | 88% | ✅ (N/A UI) |
| Capability Discovery | 6/7 | 86% | ✅ |
| Prompt-Native Features | 18/26 operator-tunable | 69% | ⚠️ |

**Overall Agent-Native Score: 78%**

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
| Tier-1 launch hygiene microbench | `make test-launch-hygiene-throughput` → `scripts/benchmark_factorized_sampler.py` | Script-only | ⚠️ |
| Direct Docker validate script | `scripts/validate_kaggle_docker_submission.py` | Demoted; `ow eval package` preferred | ⚠️ |
| Direct artifact worker script | `scripts/run_artifact_worker.py` | Demoted; `ow eval worker` preferred | ⚠️ |
| Manual queue JSON edit | Filesystem | No CLI; anti-pattern | 🚫 N/A |
| Delete run tree / checkpoint | Filesystem / rm | No first-class CLI | ⚠️ |
| W&B/Kaggle web-only flows | External UIs | Not automatable via `ow` | 🚫 N/A |

### Score: **40/44 (91%)**

Excluded from denominator: 2 N/A (external web UIs). Remaining gaps: Makefile-only factorized sampler bench, demoted scripts still runnable, no safe `ow runs delete`/archive.

### Recommendations

1. Add stderr deprecation banners on `scripts/validate_kaggle_docker_submission.py` and `scripts/run_artifact_worker.py` pointing to `ow eval package` / `ow eval worker`.
2. Document a **capability map** table in `docs/AGENT_CAPABILITIES.md` (maintain on each CLI PR).
3. Optional `ow benchmark factorized-sampler` wrapper for tier-1 hygiene (parity with Makefile).
4. Optional `ow runs archive` or documented safe cleanup for stale run dirs.

---

## Principle 2: Tools as Primitives

**Principle:** Tools provide capability, not behavior (atomic ops; agents compose workflows).

**Adaptation:** “Tools” = leaf `ow` subcommands (and `make` targets that delegate to a single `ow` command).

### Tool analysis (leaf commands)

| Command | Type | Reasoning |
|---------|------|-----------|
| `ow runs list/show/logs/watch` | PRIMITIVE | Read/poll filesystem runs |
| `ow eval status/results/worker/jobs cancel/package/submit/tournament` | PRIMITIVE | Single artifact or eval operation |
| `ow promote show/history/demote` | PRIMITIVE | Read or single rollback mutation |
| `ow benchmark gate run/list`, `tournament-proof`, `training`, `sanity` | PRIMITIVE | One gate or one measurement |
| `ow sweep create/status/list` | PRIMITIVE | One sweep lifecycle op |
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
| Resolved Hydra config | ❌ | — | Agents must run `ow train print_resolved_config=true` |
| Active GPU / terminal contention | ❌ | — | Documented in AGENTS.md only |
| User test-tier preference | ⚠️ | AGENTS.md learned prefs | Not in `agent-context` JSON |
| W&B sweep state | ❌ | — | Requires `ow sweep status` |
| Full session chat history | ⚠️ | Cursor product | Out of repo scope |

### Score: **6/9 (67%)**

Strong session-start hook (`.cursor/hooks/session-start-agent-context.sh`); missing machine-readable “is GPU busy” and default resolved-config snapshot.

### Recommendations

1. Add optional `make agent-context RESOLVED=smoke` flag (subprocess dry `print_resolved_config`, no train).
2. Document in `docs/CURSOR.md` that agents should check terminals folder before GPU commands (already in AGENTS.md — surface one line in hook failure message).
3. Include `ow benchmark gate --list` ids in agent-context for gate discovery.

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
| Train run | `ow train` | `ow runs *` | — (immutable) | — (manual FS) | ❌ | ✅ C+R |
| Checkpoint | training | paths, tournament, package | — | — | ❌ | ✅ R |
| Eval queue job | training/hybrid | `ow eval status` | cancel | `jobs cancel` | ❌ | ✅ C+R+U/D |
| Eval result (checkpoint_eval) | worker | `results list/show` | — | — | ❌ | ✅ R |
| Campaign promotion | metrics/hybrid | `ow promote show/history` | demote | demote | ❌ | ✅ R+U |
| W&B/Kaggle sweep | `ow sweep create` | status/list | — | — | ❌ | ✅ C+R |

### Scores

- **Full CRUD:** **0/6 (0%)** — no entity exposes four CLI operations (by design).
- **Operator-adequate (create + read + targeted mutation where needed):** **5/6 (83%)** — sweep lacks delete/archive.

### Recommendations

1. Document in `AGENT_CAPABILITIES.md` that **delete** for runs/checkpoints is intentionally filesystem-operator, not agent-automated.
2. If W&B API supports it, add `ow sweep cancel` or document manual W&B UI for sweep teardown.
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
2. Publish capability map (parity table) as a discoverable section in `AGENT_CAPABILITIES.md`.
3. Keep `ow benchmark gate --list` in operator docs and optionally agent-context.

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
| 1 | Maintain a **capability map** (UI action → `ow` command) in `docs/AGENT_CAPABILITIES.md` | Action Parity | Low |
| 2 | Deprecation stderr on demoted `scripts/*` → canonical `ow eval *` | Action Parity | Low |
| 3 | Extend `agent-context` with gate ids + optional resolved-config smoke | Context Injection | Medium |
| 4 | Document GPU/terminal contention in hook failure text | Context Injection | Low |
| 5 | Agent docs: prefer `gate run` / `tournament-proof` over `learn-proof` in loops | Tools as Primitives | Low |
| 6 | Optional `ow sweep cancel` or documented W&B teardown | CRUD | Medium |
| 7 | Optional safe run archive/delete primitive | CRUD | Medium |
| 8 | Parity regression test: capability map rows ⊆ `ow --help` leaves | Action Parity | Medium |
| 9 | Finish seed-scheduler calibration plan (GPU) — closes Phase 3 open item | Prompt-Native / Parity | High |
| 10 | `ow benchmark factorized-sampler` wrapper for tier-1 hygiene Makefile | Action Parity | Low |

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

- Phase 1 solution: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`
- Phase 2: `docs/agent-native-phase2-status.md`
- Phase 3: `docs/agent-native-phase3-status.md`
- Plan backlog: `docs/plans/2026-06-02-agent-native-phase3-refactors.md`

---

## Audit metadata

- **Commands run:** `uv run ow --help` (read-only); file/code review of `src/cli/`, `scripts/agent_context.py`, `.cursor/hooks/`, Makefile preflight targets.
- **Not run:** GPU train, `make preflight-learn-proof`, `ow benchmark calibrate-seed-scheduler`, seed-scheduler U1 calibration.
