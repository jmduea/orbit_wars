#!/usr/bin/env bash
# Sequential validation seed sweep — workstation profile, CUDA pinned via benchmark script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BENCH=(uv run python scripts/issues_jax_30update_benchmark.py --tier workstation)
BASE_OVERRIDES=(
  model=transformer_factorized
  format=2p_4p_16env
  training=workstation
  curriculum=off
  telemetry.wandb.enabled=false
  artifacts.artifact_pipeline.enabled=false
)

run_one() {
  local label="$1" opponents="$2" seed="$3" updates="$4" out="$5"
  echo "=== ${label} seed=${seed} updates=${updates} opponents=${opponents} ==="
  "${BENCH[@]}" \
    --label "$label" \
    --overrides "${BASE_OVERRIDES[@]}" "opponents=${opponents}" "seed=${seed}" \
    --updates "$updates" \
    --out "$out"
}

# Primary self-play sweep (500 updates)
for seed in 44 45 46; do
  run_one "validation-seed-${seed}-500u" self_play_only "$seed" 500 \
    "docs/benchmarks/validation-seed-${seed}-500u.json"
done

# Opponent contrast (100 updates — shorter; same format/curriculum)
for opponents in noop_only random_only; do
  for seed in 42 43; do
    run_one "validation-${opponents}-seed-${seed}-100u" "$opponents" "$seed" 100 \
      "docs/benchmarks/validation-${opponents}-seed-${seed}-100u.json"
  done
done

echo "=== seed sweep complete ==="
