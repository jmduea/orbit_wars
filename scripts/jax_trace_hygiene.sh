#!/usr/bin/env bash
# Static gate: forbidden patterns in JAX tier-A hot-path modules.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIER_A=(
  src/jax/env.py
  src/jax/features.py
  src/jax/rollout/collect.py
  src/jax/ppo_update.py
  src/jax/action_sampling.py
  src/jax/factored_sequence_scan.py
  src/jax/planet_flow.py
  src/jax/action_codec.py
)

fail=0

check_patterns() {
  local label=$1
  shift
  local patterns=("$@")
  for pattern in "${patterns[@]}"; do
    if rg -n -- "${pattern}" "${TIER_A[@]}" >/dev/null 2>&1; then
      echo "jax_trace_hygiene: ${label} matched '${pattern}' in tier A:" >&2
      rg -n -- "${pattern}" "${TIER_A[@]}" >&2 || true
      fail=1
    fi
  done
}

check_patterns "kaggle/callback" \
  'pure_callback' \
  'io_callback' \
  '_reference_' \
  'env_parity_mode' \
  'from src\.game\.(planet|comet)_generation'

check_patterns "host-io" \
  '\bprint\(' \
  '\bopen\(' \
  '\bbreakpoint\(' \
  '\blogging\.'

if [[ "${fail}" -ne 0 ]]; then
  echo "jax_trace_hygiene: failed (see docs/architecture/jax-trace-tiers.md)" >&2
  exit 1
fi

echo "jax_trace_hygiene: tier A static gate passed"
