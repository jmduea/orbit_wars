---
date: 2026-06-03
topic: shape-calibrate
seed_ideation: docs/ideation/2026-06-03-searchable-measurable-env-shaping-ideation.md
---

# Requirements: `ow benchmark shape-calibrate`

## Summary

Add an agent-native **`ow benchmark shape-calibrate`** command that runs a bounded factorial over **reward profile × training opponent profile × `reseed_every_updates`**, scores each cell with a **dual contract** (preflight gate trends filter survivors; held-out eval breaks ties), and writes **`docs/benchmarks/shaping-calibration.json`** with the chosen MDP bundle, pinned run paths, and calibration commit metadata—mirroring `calibrate-seed-scheduler` but for joint env shaping.

---

## Problem Frame

Orbit Wars MDP shaping knobs are spread across Hydra groups (`reward/`, `opponents/`, `training.reseed_*`, plus task/shield/curriculum). Operators and agents run ad-hoc trains and W&B sweeps without a single **measure → decide → pin** loop. Preflight gates and tournament proof exist, but there is no primitive that says: “run this small shaping grid, pick a winner on calibrated criteria, record the decision in-repo.”

That gap blocks **searchable, measurable env shaping**: outer-loop search needs enumerable candidates, a documented selection rule, and artifacts that `make agent-context` and planners can trust—without inventing thresholds or promoting on self-play noise.

---

## Key Decisions

- **Dual winner criterion.** Each grid cell must pass calibrated **Gates 2–4-style trends** on training logs (noop/random gate recipes, not self-play ~50%). Among survivors, rank by **held-out eval win rate** (seed-scheduler pattern). **Top 3** proceed to a **unified Stage-1 micro-bracket**; **only #1** may optionally run full **`ow benchmark tournament-proof`** (Docker + ladder) as a confirmation step—not required for every cell.

- **v1 search space.** Factorial **reward × opponents × reseed**, hard-capped at **≤12 cells** (CLI must error or subsample if the Cartesian product exceeds the cap). Reward arms come from existing `conf/reward/` profiles and/or `conf/wandb_sweep/space/reward_shaping.yaml` bounds. Opponent arms: `noop_only`, `random_only`, and at most one mixed/self-play arm if needed for training signal (not for winner selection). Reseed arms: small grid (e.g. `0,25,50`) aligned with `seed-scheduler-calibration.json` semantics.

- **Smoke training per cell.** Default **`training=smoke`** (or equivalent ~50–100 updates) per cell to keep one-GPU calibration affordable. Full 500-update runs are out of v1 scope unless explicitly overridden via `--total-updates`.

- **Analyze-only path.** Support **`--analyze-only`** over existing `shape_cal_*` campaigns (same pattern as `calibrate-seed-scheduler`) so reruns do not require retraining.

- **No invented thresholds.** Gate floors and tournament floors load from **`docs/benchmarks/preflight-calibration.json`** (and unified section when enforced). The command may **propose** updates to a dedicated `shaping-calibration.json` but must not silently relax bars to force a winner.

- **Shaping catalog is optional in v1.** Cells may be specified via Hydra override tuples; named `shaping_profiles` (ideation #1) are a follow-on convenience, not a blocker.

---

## Actors

- **A1. Operator / maintainer** — Runs calibration before changing default reward/opponent/reseed bundles; refreshes benchmark JSON and AGENTS threshold excerpts when decision changes.
- **A2. Coding agent** — Invokes `ow benchmark shape-calibrate --dry-run`, interprets `shaping-calibration.json` `decision` block, never promotes on training self-play win rate alone.
- **A3. CI / preflight** — May consume pinned `decision.chosen_*` fields for regression smoke grids (future); v1 is operator/agent-driven.

---

## Requirements

### CLI surface

- R1. **Subcommand registration.** `uv run ow benchmark shape-calibrate` appears in `ow benchmark` help alongside `calibrate-seed-scheduler` and `calibrate-unified-tournament`.

- R2. **Outputs.** Writes `docs/benchmarks/shaping-calibration.json` and companion `docs/benchmarks/shaping-calibration.md` (human summary). JSON includes: `commit_sha`, `gate` id, `decision` (chosen reward/opponent/reseed Hydra keys), `runs[]` per cell with `run_dir`, `log_path`, `checkpoint_path`, gate pass/fail, trend metrics, eval win rates, and rank.

- R3. **Grid definition flags.** `--reward-profiles`, `--opponents`, `--reseed-intervals` (comma-separated) with `--max-cells 12` enforced. `--dry-run` prints resolved Hydra commands without training.

- R4. **Campaign naming.** Training campaigns use predictable prefix `shape_cal_<reward>_<opponent>_reseed<N>_u<updates>` under `outputs/campaigns/` for discoverability in `--analyze-only`.

### Per-cell pipeline

- R5. **Train.** Each cell: `ow train` with smoke budget, fixed `train_seed`, `curriculum=off` unless explicitly included in a cell definition, and cell-specific Hydra overrides.

- R6. **Inner filter (gate trends).** After train, run `ow benchmark gate run beat_noop` and `beat_random` (or a single composite gate recipe if added) using logs from that cell; record pass/fail and trend scalars. Cells failing both trends are **eliminated** from held-out ranking (not merely deprioritized).

- R7. **Held-out rank (tier 1).** For survivors, run held-out eval on **noop and random** (same eval seed set policy as seed-scheduler calibration: document count in JSON). Rank by mean eval win rate (or combined 2p metric if eval harness already supports it—must match seed-scheduler semantics for v1).

- R8. **Micro-bracket (tier 2).** Top **3** cells by tier-1 rank enter unified **Stage-1** noop/random micro-tournament (calibrated `games_per_pair` from preflight JSON). Winner = highest combined score subject to prerequisite floors.

- R9. **Optional full proof.** CLI flag `--confirm-winner-tournament` runs `ow benchmark tournament-proof` on the tier-2 winner only; failure downgrades decision to “provisional” in JSON with explicit reason.

### Decision & integration

- R10. **Decision block.** `decision` records chosen Hydra overrides, supporting metrics, `candidate_count`, eliminated count, and whether confirmation tournament passed.

- R11. **AGENTS sync (manual or follow-up).** Requirements do not mandate auto-editing AGENTS.md in v1; operator runs existing threshold sync workflow after accepting calibration. Document chosen values in `shaping-calibration.md`.

- R12. **Metric context in reports.** Every cell report states opponent mix, reward profile, reseed interval, and which metrics are **not** learning signals (self-play, overall mix under wrong opponent).

### Non-goals (v1)

- NG1. LLM-generated reward code (Eureka-style).
- NG2. Task/shield/feature catalog factorial (compile-cost explosion).
- NG3. Tournament on every grid cell.
- NG4. Automatic promotion / hybrid `checkpoint_eval` enqueue from calibration alone.

---

## Success Criteria

- SC1. Operator can run `--dry-run` and see ≤12 resolved train+gate+eval commands.
- SC2. Full run produces `shaping-calibration.json` where `decision` is traceable to specific `runs[]` entries and log paths.
- SC3. A cell that passes gate trends but fails noop held-out eval cannot win.
- SC4. `--analyze-only` reproduces the same decision from existing `shape_cal_*` campaigns without retraining.
- SC5. No new threshold numbers appear in code without sourcing from calibration JSON inputs.

---

## Outstanding Questions

- Q1. Exact smoke `total_updates` default (50 vs 100) and whether to reuse `training=smoke` Hydra profile verbatim.
- Q2. Whether tier-1 held-out eval uses 2p only or combined 2p+4p for v1 (recommend 2p-only to match seed-scheduler cost).
- Q3. Minimum survivor count before tier-2 micro-bracket (if only one cell passes gates, skip bracket).
- Q4. Whether to require decomposed shaping telemetry (ideation #2) before first calibration campaign for interpretability.

---

## Dependencies & sequencing

- **Soft dependency:** Shaping catalog (#1 ideation) improves ergonomics but v1 can use explicit Hydra lists.
- **Recommended before large grids:** Decomposed shaping metrics in JSONL (ideation #2) for debugging failed cells.
- **Downstream:** `conf/benchmark/gates/shaping_calibrated.yaml` (future) loads floors from `shaping-calibration.json`; `ce-plan` implements CLI + analyzer.

---

## References

- `src/cli/benchmark.py` — `calibrate-seed-scheduler`, `calibrate-unified-tournament` patterns
- `docs/benchmarks/seed-scheduler-calibration.json` — artifact shape
- `docs/benchmarks/preflight-calibration.json` — gate and tournament floors
- `docs/ideation/2026-06-03-searchable-measurable-env-shaping-ideation.md` — idea #6
- External: [ICML 2024 auto environment shaping](https://arxiv.org/html/2407.16186) — bilevel train shaped / eval reference MDP
