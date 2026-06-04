# Agent-Native Architecture Review: Orbit Wars

**Date:** 2026-06-04  
**Last refreshed:** 2026-06-04 (post–PR [#212](https://github.com/jmduea/orbit_wars/pull/212) SSOT foundation slice; `AGENT_CAPABILITIES.md` SSOT spine)  
**Scope:** Hydra + JAX PPO RL project; primary surface is `ow` CLI (`src/cli/`), not a web UI.  
**References:** `ce-agent-native-audit` skill, `docs/AGENT_CAPABILITIES.md`, `docs/audits/agent-native-architecture-2026-06-02.md`, SSOT learning `docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md`.

---

## Executive summary

Orbit Wars remains **strongly agent-native at the operator layer**: humans and coding agents share the same `ow` CLI, `outputs/campaigns/` tree, and tested capability map. PR #212 landed SSOT foundation (U1–U4): seed partition, W&B preflight sweep recipe, packaging CLI flags, `artifacts=ssot_pipeline` stub. This refresh adds the **canonical SSOT production spine** and flowchart link **above** the legacy hybrid submit-valid tree in `AGENT_CAPABILITIES.md` — closing the top discovery gap from the 2026-06-02 audit.

**Overall agent-native score: ~87%** (unweighted mean of eight principles below; up from **84%** on 2026-06-02).

Remaining gaps before SSOT U5–U8: prompt-native score held back by dual spines in code until teardown; packaging defaults still 42/`both` without explicit SSOT flags; `make agent-context` does not embed capability-map body or SSOT flowchart pointer.

---

## Overall Score Summary

| Core Principle | Score | Percentage | Status |
|----------------|-------|------------|--------|
| Action Parity | 41/44 | 93% | ✅ |
| Tools as Primitives | 35/44 leaf `ow` paths | 80% | ✅ |
| Context Injection | 8/9 | 89% | ✅ |
| Shared Workspace | 11/12 | 92% | ✅ |
| CRUD Completeness | 8/8 operator-adequate *(0/8 strict full CRUD)* | 100% / 0% | ✅ / by design |
| UI Integration | 7/8 CLI channels | ~88% | ✅ |
| Capability Discovery | 6/7 | 86% | ✅ |
| Prompt-Native Features | 18/26 operator features | 69% | ⚠️ |

**Overall Agent-Native Score: ~87%**

### Status legend

- ✅ Excellent (80%+)
- ⚠️ Partial (50–79%)
- ❌ Needs Work (&lt;50%)

---

## Changes since 2026-06-02 audit

| Item | Status |
|------|--------|
| SSOT spine in `AGENT_CAPABILITIES.md` + flowchart link | **Done** (this refresh) |
| SSOT foundation code (U1–U4) | **Merged** PR #212 |
| Capability map: `ow make`, `ow eval bracket status`, SSOT rows | **Done** |
| Legacy hybrid mermaid demoted below SSOT section | **Done** |
| U5–U8 implementation + R29 doc/code teardown | **Open** (#211) |
| `shape-calibrate` CLI | Still planned |
| Capability map excerpt in `make agent-context` JSON | Open |

---

## Top 10 Recommendations by Impact

| Priority | Action | Principle | Effort |
|----------|--------|-----------|--------|
| 1 | LFG SSOT U5–U8; default train → `artifacts=ssot_pipeline`; R29 teardown | Prompt-native | Large |
| 2 | Embed primitive/workflow tier + flowchart URL in `make agent-context` | Context | Small |
| 3 | SSOT packaging defaults (seed 0 / 4p) without mandatory CLI flags | Parity | Medium |
| 4 | Agent loops: `gate run` + `tournament-proof` — not `learn-proof` | Primitives | Policy |
| 5 | Ship or de-map `ow benchmark shape-calibrate` | Parity | Medium |
| 6 | Move `GATE_ORDER` to gate YAML metadata | Prompt-native | Medium |
| 7 | Default `RESOLVED=smoke` in session hook if Hydra drift is common | Context | Small |
| 8 | Optional: `ow runs checkpoint list`, Kaggle `ow sweep cancel` | CRUD | Medium |
| 9 | Gate reports under `outputs/` when cross-session sharing matters | Workspace | Small |
| 10 | Update SSOT learning doc as each U* lands | Discovery | Small |

---

## What's Working Excellently

1. **Tested capability map** — `tests/test_agent_capability_map.py` prevents doc/CLI drift.
2. **Session bootstrap** — `make agent-context` + Cursor hook (thresholds, gates, recent runs).
3. **Primitive eval/runs surface** — `ow eval status --watch`, `results show`, `jobs cancel`, `runs archive`.
4. **Shared `outputs/` contract** — no parallel agent sandbox.
5. **YAML-first preflight gates** — `conf/benchmark/gates/*.yaml` + calibration JSON.
6. **SSOT operator map** — interactive flowchart + production spine table in agent capabilities doc.

---

## Principle summaries (abbreviated)

Full detail in sub-audits run 2026-06-04; prior depth in `agent-native-architecture-2026-06-02.md`.

### Action Parity (93%)

38 capability-map `ow` commands + `make agent-context` + `ow eval bracket` + Makefile targets. Gaps: planned `shape-calibrate`; no `ow runs delete` (archive only).

### Tools as Primitives (80%)

Prefer: `ow runs *`, `ow eval status/results/jobs`, `ow benchmark gate run`, `tournament-proof`, `ow sweep create/cancel`, `ow make` + SSOT sweep YAML. Avoid in agent loops: `learn-proof`, `hybrid_promotion` train (legacy).

### Context Injection (89%)

`scripts/agent_context.py` + session hook. Gaps: capability map body, `docs/README.md` pointer, default resolved config, terminals-aware GPU.

### Shared Workspace (92%)

Single filesystem + W&B via CLI; bulk run dirs de-indexed — use `ow` primitives.

### CRUD (operator-adequate 100%)

Append-only runs; archive/delete checkpoint/cancel jobs/demote promotion. Strict full CRUD: 0/8 by design.

### UI Integration (~88%)

CLI-first: JSONL, `ow runs watch`, `ow eval status --watch`. W&B external; flowchart static HTML.

### Capability Discovery (86%)

Layered `ow --help`, `AGENT_CAPABILITIES.md`, `AGENTS.md`, capability-map test, session hook. SSOT flowchart now linked from capabilities doc.

### Prompt-Native (69%)

Hydra gates strong; dual SSOT (doc) vs hybrid/Gate 5/bracket (code) until #211 U8.

---

## Next audit trigger

Re-run after SSOT U8 teardown or material CLI surface change (`ow benchmark shape-calibrate`, qualifier calibration primitive).
