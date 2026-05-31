#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Phase 3 baseline $(date -Iseconds) ==="
echo "--- git ---"
git rev-parse HEAD
git status -sb
echo "--- issue gate ---"
for i in 147 148 149 150; do
  gh issue view "$i" --json number,state,title --jq '"#\(.number) \(.state) \(.title)"'
done
echo "--- touch file sizes ---"
wc -l src/game/trajectory_shield.py \
  src/opponents/jax_actions/builders.py \
  src/artifacts/promotion.py \
  src/artifacts/tournament/promotion.py \
  src/opponents/pool.py
echo "--- trajectory_shield importers ---"
rg -l 'trajectory_shield|from src\.game\.trajectory_shield' src tests --glob '*.py' || true
echo "--- tests ---"
make test-fast
make test-domain-config
make test-domain-curriculum
make test-domain-artifacts
echo "=== baseline OK ==="
