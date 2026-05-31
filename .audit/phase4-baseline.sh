#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Phase 4 baseline $(date -Iseconds) ==="
echo "--- git ---"
git rev-parse HEAD
git status -sb
echo "--- phase gate (Phase 3 must be CLOSED) ---"
for i in 147 148 149 150; do
  gh issue view "$i" --json number,state,title --jq '"#\(.number) \(.state) \(.title)"'
done
echo "--- phase 4 issues ---"
for i in 151 152 153 154 155; do
  gh issue view "$i" --json number,state,title --jq '"#\(.number) \(.state) \(.title)"'
done
echo "--- train.py extraction touch sizes ---"
wc -l \
  src/jax/train.py \
  src/artifacts/checkpoint_retention.py \
  src/artifacts/pipeline.py \
  src/artifacts/promotion.py \
  src/telemetry/metric_registry.py \
  src/features/registry.py \
  src/features/catalog/edge.py
echo "--- train.py helper line ranges (approx) ---"
rg -n "^def (_checkpoint|_queue|handle_checkpoint|_write_filtered|load_jax|save_jax|_restore_curriculum|_historical_pool|run_jax)" src/jax/train.py || true
echo "--- intercept_anchors config ---"
rg -n "intercept_anchors" conf/task/base.yaml src/config/schema.py
echo "--- targeted tests (domain slices) ---"
make test-domain-artifacts
make test-domain-features
uv run --group dev pytest tests/test_jax_train_timing.py tests/test_metric_registry.py tests/test_telemetry.py -m "not slow and not jax" -q
echo "=== baseline OK ==="
