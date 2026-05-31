#!/usr/bin/env bash
# Create src audit GitHub issues; prints issue numbers to stdout.
set -euo pipefail
cd /home/jmduea/projects/orbit_wars

REF='[src audit inventory](4f6f328c-8f83-4dea-a074-0562e2000c0d)'
GLOBAL='## Global directives
- Unless backward-compat is **explicitly requested**, prefer aggressive clean refactors/deletion.
- Success bar when touching training/tournament: `make test-fast`, clear docs, `uv run ow train` smoke, **`submission.tar.gz` works through tournament path**.
- Breaking checkpoint changes OK; no migration note required.
- Sacred: `planet_graph_transformer` + `factorized_topk_pointer` only; other encoders/decoders may be removed.

## Reference
'"$REF"

create() {
  local title="$1"
  local labels="$2"
  local body="$3"
  gh issue create --title "$title" --label "$labels" --body "$body"
}

# --- Phase 1 (numbers assigned after creation; deps use placeholders updated in meta) ---

ISSUE_P1_HELP=$(create "src audit phase 1: verbose ow --help and CLI routing" "type:task,area:infra" "$(cat <<EOF
## Summary

First priority among Phase 1 equals: verbose CLI help describing what each \`ow\` command does and available Hydra overrides, without triggering full Hydra composition on \`ow --help\` / \`-h\` / \`ow train --help\`.

## Acceptance criteria

- [ ] \`ow --help\`, \`ow -h\`, and \`ow train --help\` show command descriptions and override hints without Hydra sniffing errors
- [ ] Hydra override sniffing rules consistent between \`src/cli/__init__.py\` and \`src/cli/train_hosts.py\`
- [ ] Help text documents primary \`ow train\`, \`ow eval\`, and Kaggle entrypoints
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P1-A (first among equals within Phase 1)
- **Blocked by:** Phase 0 planning complete
- **Blocks:** Phase 2+ (soft — no hard deps)

## Files likely touched

- \`src/cli/__init__.py\`
- \`src/cli/train_hosts.py\`
- \`src/cli/eval.py\`, \`src/cli/kaggle_runner.py\` (help strings only)

## Verification

\`\`\`bash
make test-fast
ow --help
ow train --help
\`\`\`

$GLOBAL
EOF
)")
echo "P1-A $ISSUE_P1_HELP"

ISSUE_P1_WANDB=$(create "src audit phase 1: fix run_context_for_agent wandb path drift" "type:task,area:infra" "$(cat <<EOF
## Summary

Preventive fix for \`run_context_for_agent\` wandb cache / log path drift in tournament artifact resolution (double \`cache/\` segment; naming mismatch vs \`resolve_run_paths\`).

## Acceptance criteria

- [ ] \`run_context_for_agent\` wandb paths align with \`artifacts/run_paths.py\` conventions
- [ ] \`tests/test_tournament.py\` covers checkpoint-derived run dir and wandb cache layout
- [ ] Tournament eval / worker paths unchanged for valid layouts
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P1-B
- **Blocked by:** none within Phase 1

## Files likely touched

- \`src/artifacts/tournament/resolve.py\`
- \`src/artifacts/run_paths.py\` (reference only unless shared helper extracted)
- \`tests/test_tournament.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-artifacts
\`\`\`

$GLOBAL
EOF
)")
echo "P1-B $ISSUE_P1_WANDB"

ISSUE_P1_KAGGLE=$(create "src audit phase 1: Kaggle P100 default and unified push path" "type:task,area:infra" "$(cat <<EOF
## Summary

Align Kaggle launch defaults to **P100 everywhere** (\`ow train kaggle\` + script paths). Unify kernel push through one implementation (remove duplicate \`_push_kernel\` vs unused \`KaggleCli.push\`).

## Acceptance criteria

- [ ] Single default accelerator **P100** when \`--accelerator\` omitted (\`ow\` CLI + \`orchestration/kaggle_runner.py\`)
- [ ] No blind push/retry across all eight GPUs when accelerator unset
- [ ] One canonical push implementation; dead duplicate removed
- [ ] \`tests/test_kaggle_runner.py\` updated if behavior changes
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P1-C
- **Blocked by:** none within Phase 1

## Files likely touched

- \`src/orchestration/kaggle_runner.py\`
- \`src/orchestration/kaggle_cli.py\`
- \`src/cli/kaggle_runner.py\`
- \`src/orchestration/accelerators.py\`

## Verification

\`\`\`bash
make test-fast
uv run ow train kaggle preflight
\`\`\`

$GLOBAL
EOF
)")
echo "P1-C $ISSUE_P1_KAGGLE"

ISSUE_P1_METRIC=$(create "src audit phase 1: shared rollout metric contract module (Option C)" "type:task,area:train" "$(cat <<EOF
## Summary

Create a neutral rollout metric contract module (Option C): single canonical tuple imported by both \`jax/rollout/metrics.py\` and \`telemetry/metric_registry.py\` to stop hand-maintained key drift.

## Acceptance criteria

- [ ] New contract module owns logged rollout scalar key tuple (+ docs for internal-only keys)
- [ ] \`jax/rollout/metrics.py\` and \`telemetry/metric_registry.py\` import contract (no duplicated lists)
- [ ] Sync test fails if rollout emits keys outside contract or registry omits contract keys
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P1-D
- **Blocks:** telemetry gating issue (P1-E)

## Files likely touched

- \`src/jax/rollout/metrics.py\`
- \`src/telemetry/metric_registry.py\`
- New: \`src/jax/rollout/metric_contract.py\` (or \`src/telemetry/rollout_metric_contract.py\`)
- \`tests/test_metric_registry.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-config
\`\`\`

$GLOBAL
EOF
)")
echo "P1-D $ISSUE_P1_METRIC"

ISSUE_P1_GATE=$(create "src audit phase 1: telemetry gating — skip compute for disabled groups" "type:task,area:train" "$(cat <<EOF
## Summary

Disabled telemetry metric groups must **not** be computed in rollout/PPO/train paths. Aligns with Phase 4 train.py extraction but lands in Phase 1 per interview decision.

## Acceptance criteria

- [ ] Rollout/PPO skip computing metrics for disabled \`telemetry.metric_groups\`
- [ ] Runtime filtering still applied at log boundary; no orphan compute for disabled families
- [ ] Protected metrics (\`protected_metric_names\`, plateau metric) still computed when groups off
- [ ] Tests prove disabled group keys absent from merge paths and JSONL
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P1-E
- **DEPENDS ON:** rollout metric contract issue (P1-D)
- **Blocks:** Phase 4 telemetry record assembly extraction

## Files likely touched

- \`src/jax/train.py\`
- \`src/jax/rollout/metrics.py\`, \`src/jax/rollout/collect.py\`
- \`src/jax/ppo_update.py\`
- \`src/telemetry/metric_registry.py\`
- \`tests/test_metric_registry.py\`

## Verification

\`\`\`bash
make test-fast
\`\`\`

$GLOBAL
EOF
)")
echo "P1-E $ISSUE_P1_GATE"

ISSUE_P1_CONFIG=$(create "src audit phase 1: config required-key audit and composition tests" "type:task,area:config" "$(cat <<EOF
## Summary

Replace brittle full-config equality tests with composition validation: base YAMLs declare required keys; command-critical values asserted as membership in acceptable sets for \`ow\` commands.

## Acceptance criteria

- [ ] Audit \`conf/**/base.yaml\` for required keys vs \`src/config/schema.py\`
- [ ] Composition tests verify Hydra compose succeeds for primary \`ow train\` / \`ow eval\` profiles
- [ ] Command-critical values tested as set membership (not hardcoded full resolved config blobs)
- [ ] Document test philosophy in test module docstring or \`conf/README.md\` one-liner
- [ ] \`make test-fast\` and \`make test-domain-config\` pass

## Dependencies / blocks

- **Parallel group:** P1-F
- **Blocked by:** none within Phase 1

## Files likely touched

- \`tests/test_config_consolidation.py\` (and related)
- \`conf/**/base.yaml\` (add missing keys only)
- \`src/config/runtime.py\` (validation helpers if needed)

## Verification

\`\`\`bash
make test-fast
make test-domain-config
uv run ow train print_resolved_config=true
\`\`\`

$GLOBAL
EOF
)")
echo "P1-F $ISSUE_P1_CONFIG"

# --- Phase 2 ---

ISSUE_P2_SHIELD=$(create "src audit phase 2: delete v1 trajectory_shield dead candidate APIs" "type:task,area:train" "$(cat <<EOF
## Summary

Aggressively delete ~15 v1 \`trajectory_shield\` candidate APIs that AttributeError today (\`TurnBatch\` / joint-flat paths). Delete **in place** in \`game/trajectory_shield.py\` before Phase 3 relocation — do not relocate dead symbols.

## Acceptance criteria

- [ ] Remove v1 candidate/shield APIs with zero live callers (~15 symbols per audit)
- [ ] No references remain in \`src/\`, \`tests/\`, \`scripts/\`
- [ ] Phase 3 move issue retains only live shield paths
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P2-A
- **Blocked by:** all Phase 1 issues complete
- **Blocks:** Phase 3 shield relocation (P3-A) — complete dead API deletion first

## Files likely touched

- \`src/game/trajectory_shield.py\`
- \`src/jax/ppo_update.py\` (stale imports if any)
- \`tests/\` referencing v1 APIs

## Verification

\`\`\`bash
make test-fast
\`\`\`

$GLOBAL
EOF
)")
echo "P2-A $ISSUE_P2_SHIELD"

ISSUE_P2_OPP=$(create "src audit phase 2: remove unused OpponentRegistry" "type:task,area:train" "$(cat <<EOF
## Summary

Remove unused \`OpponentRegistry\` from \`opponents/pool.py\` (~130 lines). Curriculum uses \`StageView\` instead.

## Acceptance criteria

- [ ] \`OpponentRegistry\` class and exports removed
- [ ] No imports of \`OpponentRegistry\` in \`src/\` or \`tests/\`
- [ ] \`opponents/pool.py\` retains live helpers used by curriculum/runtime
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P2-B
- **Blocked by:** all Phase 1 issues complete
- **Note:** Phase 3 adds \`opponents/constants.py\`; do not duplicate family lists here

## Files likely touched

- \`src/opponents/pool.py\`
- \`src/opponents/__init__.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-curriculum
\`\`\`

$GLOBAL
EOF
)")
echo "P2-B $ISSUE_P2_OPP"

ISSUE_P2_FEAT=$(create "src audit phase 2: remove FeatureExtractor and normalization stub" "type:task,area:train" "$(cat <<EOF
## Summary

Verify \`FeatureExtractor\` / \`features/extractor.py\` is not required for agent submissions (\`submission_runtime\` path). If unused, delete \`features/extractor.py\` + \`features/normalization.py\` stub and update \`AGENTS.md\`.

## Acceptance criteria

- [ ] Confirm submission/tournament path works without \`FeatureExtractor\`
- [ ] Delete unused modules and exports
- [ ] Update \`AGENTS.md\` (normalization wired via \`jax/normalization.py\`; extractor not canonical)
- [ ] \`submission.tar.gz\` tournament smoke passes if submission packaging touched
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P2-C
- **Blocked by:** all Phase 1 issues complete

## Files likely touched

- \`src/features/extractor.py\`, \`src/features/normalization.py\`
- \`src/features/__init__.py\`
- \`src/jax/submission_runtime.py\`
- \`AGENTS.md\`

## Verification

\`\`\`bash
make test-fast
make test-domain-features
uv run ow train print_resolved_config=true
\`\`\`

$GLOBAL
EOF
)")
echo "P2-C $ISSUE_P2_FEAT"

ISSUE_P2_CONFIG=$(create "src audit phase 2: remove dead config fields from schema and YAML" "type:task,area:config" "$(cat <<EOF
## Summary

Remove dead config fields from \`schema.py\` and matching \`conf/\` YAML in one pass (aggressive cleanup).

## Acceptance criteria

- [ ] Remove dead fields including: \`trajectory_shield_train_horizon\`, \`multi_opponent_mode\`, \`fail_training_on_optional_job_error\`, \`latest_lag_warning_updates\`, and other audit-identified unused knobs
- [ ] Schema + YAML updated together; no orphan YAML keys
- [ ] \`compose_hydra_train_config\` and validation updated
- [ ] \`make test-fast\` and \`make test-domain-config\` pass

## Dependencies / blocks

- **Parallel group:** P2-D
- **Blocked by:** all Phase 1 issues complete

## Files likely touched

- \`src/config/schema.py\`
- \`src/config/runtime.py\`
- \`conf/**/*.yaml\`

## Verification

\`\`\`bash
make test-fast
make test-domain-config
uv run ow train print_resolved_config=true
\`\`\`

$GLOBAL
EOF
)")
echo "P2-D $ISSUE_P2_CONFIG"

ISSUE_P2_CLI=$(create "src audit phase 2: remove CLI and orchestration cruft" "type:task,area:infra" "$(cat <<EOF
## Summary

Remove CLI/orchestration dead code: \`KAGGLE_FLAGS\`, \`_hydra_entry\`, deprecated Kaggle shims, duplicate parsers where safe.

## Acceptance criteria

- [ ] Remove \`KAGGLE_FLAGS\` / unused flag tables from \`cli/train_hosts.py\`
- [ ] Remove dead \`_hydra_entry\` from \`src/train.py\` if unused
- [ ] Remove deprecated Kaggle script shims (\`scripts/kaggle_wandb_population.py\` etc.) per audit
- [ ] Collapse duplicate argparse where \`ow\` is canonical entry
- [ ] \`make test-fast\` passes; \`uv run ow train kaggle preflight\` still works

## Dependencies / blocks

- **Parallel group:** P2-E
- **Blocked by:** all Phase 1 issues complete
- **DEPENDS ON (soft):** Phase 1 Kaggle unification issue

## Files likely touched

- \`src/cli/train_hosts.py\`
- \`src/train.py\`
- \`src/orchestration/kaggle_runner.py\`
- \`scripts/kaggle_wandb_population.py\` (delete if shim)

## Verification

\`\`\`bash
make test-fast
uv run ow train kaggle preflight
\`\`\`

$GLOBAL
EOF
)")
echo "P2-E $ISSUE_P2_CLI"

# --- Phase 3 ---

ISSUE_P3_SHIELD=$(create "src audit phase 3: relocate trajectory_shield to jax/shield/*" "type:task,area:train" "$(cat <<EOF
## Summary

Split \`game/trajectory_shield.py\` into \`jax/shield/*\` modules; dedupe physics with \`jax/env.py\` and \`jax/feature_primitives.py\`. Keep only constants/types + Python opponent helpers in \`game/\`.

## Acceptance criteria

- [ ] Live shield logic lives under \`src/jax/shield/\`
- [ ] Physics helpers deduped with \`jax/env\` / \`feature_primitives\` (single source)
- [ ] \`game/\` no longer imports \`jax\`; layering tests or import guard documented
- [ ] Update \`docs/architecture/\` stage doc if module boundaries change
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P3-A (serial first within Phase 3)
- **Blocked by:** all Phase 2 issues complete
- **DEPENDS ON:** Phase 2 v1 dead API deletion (P2-A)
- **Blocks:** Phase 4 parametric edge catalog (shield bucket mapping)

## Files likely touched

- \`src/game/trajectory_shield.py\` (relocate/trim)
- \`src/jax/shield/\` (new)
- \`src/jax/env.py\`, \`src/jax/feature_primitives.py\`
- \`src/jax/ppo_update.py\`, \`src/opponents/jax_actions/builders.py\`

## Verification

\`\`\`bash
make test-fast
uv run ow train training.total_updates=2
\`\`\`

$GLOBAL
EOF
)")
echo "P3-A $ISSUE_P3_SHIELD"

ISSUE_P3_SAMPLE=$(create "src audit phase 3: extract jax/action_sampling.py from opponents/builders" "type:task,area:train" "$(cat <<EOF
## Summary

Move learner shielded sampling (\`_sample_shielded_*\`) to \`jax/action_sampling.py\`. \`opponents/jax_actions/builders.py\` keeps scripted opponents only; imports sampling from jax.

## Acceptance criteria

- [ ] \`jax/action_sampling.py\` owns factorized shielded sequence sampling used by rollout + submission
- [ ] \`opponents/jax_actions/builders.py\` thinned to scripted opponent builders
- [ ] Pure helpers unit-testable without full training loop / env scan
- [ ] \`make test-fast\` passes; submission path smoke OK

## Dependencies / blocks

- **Parallel group:** P3-B
- **Blocked by:** all Phase 2 issues complete
- **DEPENDS ON:** P3-A shield relocation (imports from \`jax/shield\`)

## Files likely touched

- \`src/jax/action_sampling.py\` (new)
- \`src/opponents/jax_actions/builders.py\`
- \`src/jax/rollout/collect.py\`
- \`src/jax/submission_runtime.py\`

## Verification

\`\`\`bash
make test-fast
uv run ow train training.total_updates=2
\`\`\`

$GLOBAL
EOF
)")
echo "P3-B $ISSUE_P3_SAMPLE"

ISSUE_P3_CONST=$(create "src audit phase 3: canonical opponent families in opponents/constants.py" "type:task,area:config" "$(cat <<EOF
## Summary

Single source of truth for opponent family IDs in \`opponents/constants.py\` (or \`pool.py\`). \`config/runtime.py\` validates curriculum stages against canonical list.

## Acceptance criteria

- [ ] \`OPPONENT_FAMILY_NAMES\` (or equivalent) defined once in opponents package
- [ ] \`config/runtime._validate_curriculum_config\` imports allowlist (no duplicated strings)
- [ ] Tests for invalid family rejection
- [ ] \`make test-fast\` and \`make test-domain-curriculum\` pass

## Dependencies / blocks

- **Parallel group:** P3-C (parallel with P3-A)
- **Blocked by:** all Phase 2 issues complete

## Files likely touched

- \`src/opponents/constants.py\` (new)
- \`src/opponents/pool.py\`
- \`src/config/runtime.py\`
- \`conf/curriculum/*.yaml\`

## Verification

\`\`\`bash
make test-fast
make test-domain-curriculum
make test-domain-config
\`\`\`

$GLOBAL
EOF
)")
echo "P3-C $ISSUE_P3_CONST"

ISSUE_P3_PROMO=$(create "src audit phase 3: unified promotion manifest writer" "type:task,area:infra" "$(cat <<EOF
## Summary

Extract shared promotion manifest writer used by both \`artifacts/promotion.py\` and \`tournament/promotion.py\` (policies stay separate; I/O unified).

## Acceptance criteria

- [ ] Single writer/helper for manifest JSON schema + atomic append patterns
- [ ] Metric promotion and tournament promotion both call shared writer
- [ ] No behavioral regression in promotion indexes / \`promoted/current_best\`
- [ ] \`make test-fast\` and \`make test-domain-artifacts\` pass

## Dependencies / blocks

- **Parallel group:** P3-D (parallel with P3-A/C)
- **Blocked by:** all Phase 2 issues complete
- **Blocks:** Phase 4 promotion/tournament queue extraction

## Files likely touched

- \`src/artifacts/promotion.py\`
- \`src/artifacts/tournament/promotion.py\`
- \`src/artifacts/run_paths.py\`, \`src/artifacts/pipeline.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-artifacts
\`\`\`

$GLOBAL
EOF
)")
echo "P3-D $ISSUE_P3_PROMO"

# --- Phase 4 ---

ISSUE_P4_CKPT=$(create "src audit phase 4: extract checkpoint and artifact hooks from jax/train.py" "type:task,area:train" "$(cat <<EOF
## Summary

First Phase 4 extraction (auxiliary, testable without full loop): checkpoint save/load, retention, and artifact hook calls out of \`jax/train.py\`.

## Acceptance criteria

- [ ] New module(s) own checkpoint I/O + artifact hooks invoked from thin train orchestrator
- [ ] Unit tests cover hook module without running full training loop
- [ ] Checkpoint retention + W&B artifact policy unchanged unless intentionally fixed
- [ ] \`make test-fast\` passes; short \`uv run ow train\` smoke passes

## Dependencies / blocks

- **Parallel group:** P4-1 (serial first within Phase 4)
- **Blocked by:** all Phase 3 issues complete

## Files likely touched

- \`src/jax/train.py\`
- New: \`src/jax/train_checkpoint.py\` or \`src/artifacts/train_hooks.py\`
- \`src/artifacts/checkpoint_retention.py\`
- \`tests/test_jax_train_timing.py\` (extend)

## Verification

\`\`\`bash
make test-fast
uv run ow train training.total_updates=2
\`\`\`

$GLOBAL
EOF
)")
echo "P4-1 $ISSUE_P4_CKPT"

ISSUE_P4_TELEM=$(create "src audit phase 4: extract telemetry record assembly from jax/train.py" "type:task,area:train" "$(cat <<EOF
## Summary

Extract telemetry update \`record\` assembly from \`jax/train.py\` into a testable module. Builds on Phase 1 metric contract + telemetry gating.

## Acceptance criteria

- [ ] Record assembly module tested without full training loop
- [ ] Respects metric contract + disabled-group gating (no compute for off groups)
- [ ] \`filter_update_record\` applied at log boundary as today
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P4-2
- **Blocked by:** all Phase 3 issues complete
- **DEPENDS ON:** Phase 1 telemetry gating (P1-E); **DEPENDS ON:** P4-1 (serial after checkpoint extraction recommended)

## Files likely touched

- \`src/jax/train.py\`
- New: \`src/jax/train_telemetry.py\` or \`src/telemetry/update_record.py\`
- \`src/telemetry/metric_registry.py\`

## Verification

\`\`\`bash
make test-fast
uv run ow train training.total_updates=2
\`\`\`

$GLOBAL
EOF
)")
echo "P4-2 $ISSUE_P4_TELEM"

ISSUE_P4_QUEUE=$(create "src audit phase 4: extract promotion and tournament queue from jax/train.py" "type:task,area:train" "$(cat <<EOF
## Summary

Extract promotion / tournament queue scheduling from \`jax/train.py\` into a testable auxiliary module.

## Acceptance criteria

- [ ] Queue module owns tournament job enqueue + promotion hook orchestration
- [ ] Uses unified promotion writer from Phase 3
- [ ] Unit tests without full training loop
- [ ] Tournament path + \`submission.tar.gz\` smoke pass when touched
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P4-3
- **Blocked by:** all Phase 3 issues complete
- **DEPENDS ON:** P3-D unified promotion writer; **DEPENDS ON:** P4-1 checkpoint hooks

## Files likely touched

- \`src/jax/train.py\`
- \`src/artifacts/tournament/worker.py\`, \`src/artifacts/tournament/runner.py\`
- \`src/artifacts/promotion.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-artifacts
uv run ow train training.total_updates=2
\`\`\`

$GLOBAL
EOF
)")
echo "P4-3 $ISSUE_P4_QUEUE"

ISSUE_P4_REG=$(create "src audit phase 4: collapse to CatalogView; delete FeatureGroupRegistry" "type:task,area:train" "$(cat <<EOF
## Summary

Registry collapse: \`FeatureCatalog\` remains source of truth; \`*_FEATURE_SCHEMA\` become thin \`CatalogView\` wrappers. **Delete** \`FeatureGroupRegistry\` duplicate slice API.

## Acceptance criteria

- [ ] \`CatalogView\` provides \`.base_slice\`, \`.dim\`, \`.names\`, \`.with_history(n)\`
- [ ] \`FeatureGroupRegistry\` / \`FeatureItem\` mirror removed
- [ ] Policy/checkpoint imports keep \`*_FEATURE_SCHEMA\` names (minimal churn)
- [ ] Drift test: catalog vs views consistent
- [ ] \`make test-fast\` and \`make test-domain-features\` pass

## Dependencies / blocks

- **Parallel group:** P4-4
- **Blocked by:** all Phase 3 issues complete
- **Blocks:** parametric edge catalog issue (P4-5)

## Files likely touched

- \`src/features/catalog/_core.py\`
- \`src/features/registry.py\`
- New: \`src/features/schema_api.py\` or \`src/features/slices.py\`
- \`src/jax/policy.py\`, \`src/artifacts/checkpoint_compat.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-features
\`\`\`

$GLOBAL
EOF
)")
echo "P4-4 $ISSUE_P4_REG"

ISSUE_P4_EDGE=$(create "src audit phase 4: parametric edge catalog default intercept_anchors [1,3,6]" "type:task,area:train" "$(cat <<EOF
## Summary

Parametric edge catalog from \`intercept_anchors\`; default **\`[1.0, 3.0, 6.0]\`** (E=23). Update shield bucket mapping, sniper/opponent intercept usage, checkpoint metadata, goldens. Rationale: mid-speed s3 teaching signal when speed-6 launches are rare early.

## Acceptance criteria

- [ ] \`conf/task/base.yaml\`: \`intercept_anchors: [1.0, 3.0, 6.0]\`
- [ ] Edge catalog generates features per anchor list (\`intercept_distance_s3\`, etc.); **E = 5 × N + 8 = 23**
- [ ] \`jax/features.py\` + catalog assembly aligned (no discarded third anchor)
- [ ] Shield bucket→anchor mapping updated (not hardcoded s1/s6 only)
- [ ] \`nearest_sniper\` intercept ranking considers s3 or min across anchors
- [ ] Checkpoint metadata stores anchor list; mismatch fails clearly
- [ ] Golden tests + \`test_checkpoint_compat.py\` updated
- [ ] \`submission.tar.gz\` tournament path smoke passes
- [ ] \`make test-fast\` passes

## Dependencies / blocks

- **Parallel group:** P4-5 (serial last within Phase 4)
- **Blocked by:** all Phase 3 issues complete
- **DEPENDS ON:** P4-4 CatalogView; **DEPENDS ON:** P3-A shield relocation

## Files likely touched

- \`src/features/catalog/edge.py\`
- \`src/jax/features.py\`
- \`src/config/schema.py\`, \`conf/task/*.yaml\`
- \`src/game/trajectory_shield.py\` or \`src/jax/shield/*\`
- \`src/opponents/jax_actions/builders.py\`
- \`src/artifacts/checkpoint_compat.py\`
- \`tests/test_feature_encoding_golden.py\`, \`tests/test_checkpoint_compat.py\`

## Verification

\`\`\`bash
make test-fast
make test-domain-features
uv run ow train training.total_updates=2
\`\`\`

$GLOBAL
EOF
)")
echo "P4-5 $ISSUE_P4_EDGE"

# Export for meta issue
cat > /home/jmduea/projects/orbit_wars/.omg/tmp/src_audit_issue_numbers.env <<ENV
P1_HELP=$ISSUE_P1_HELP
P1_WANDB=$ISSUE_P1_WANDB
P1_KAGGLE=$ISSUE_P1_KAGGLE
P1_METRIC=$ISSUE_P1_METRIC
P1_GATE=$ISSUE_P1_GATE
P1_CONFIG=$ISSUE_P1_CONFIG
P2_SHIELD=$ISSUE_P2_SHIELD
P2_OPP=$ISSUE_P2_OPP
P2_FEAT=$ISSUE_P2_FEAT
P2_CONFIG=$ISSUE_P2_CONFIG
P2_CLI=$ISSUE_P2_CLI
P3_SHIELD=$ISSUE_P3_SHIELD
P3_SAMPLE=$ISSUE_P3_SAMPLE
P3_CONST=$ISSUE_P3_CONST
P3_PROMO=$ISSUE_P3_PROMO
P4_CKPT=$ISSUE_P4_CKPT
P4_TELEM=$ISSUE_P4_TELEM
P4_QUEUE=$ISSUE_P4_QUEUE
P4_REG=$ISSUE_P4_REG
P4_EDGE=$ISSUE_P4_EDGE
ENV

echo "ALL_ISSUES_CREATED"
