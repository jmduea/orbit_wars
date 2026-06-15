#!/usr/bin/env python3
"""CE-optimize measurement harness: multitask smoke JAX training throughput.

Emits a single JSON object on stdout with gate + primary + diagnostic metrics.
Used only by /ce-optimize (immutable scope).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_OVERRIDES = [
    "model=planet_graph_transformer_small",
    "task.candidate_count=3",
    "task.edge_rank_mode=intercept_min",
    "training.num_envs=2",
    "training.rollout_microbatch_envs=2",
    "training.rollout_steps=128",
    "training.update_chunk_rows=2048",
    "curriculum=noop_only",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "seed=42",
]

TARGETED_TESTS = [
    "tests/test_rollout_noop_opponent.py",
    "tests/test_jax_policy_encoder.py::test_build_jax_policy_dispatches_planet_graph_transformer_small",
    "tests/test_config_consolidation.py::test_multitask_smoke_overrides_compose",
]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    env = os.environ.copy()
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env["ORBIT_WARS_PYTEST_JAX_CACHE"] = "0"

    test_proc = _run(
        ["uv", "run", "pytest", *TARGETED_TESTS, "-q", "--tb=no"],
        env=env,
    )
    tests_passed = 1 if test_proc.returncode == 0 else 0

    bench_out = REPO_ROOT / ".context/compound-engineering/ce-optimize/multitask-smoke-throughput/last_benchmark.json"
    bench_out.parent.mkdir(parents=True, exist_ok=True)

    bench_cmd = [
        "uv",
        "run",
        "ow",
        "benchmark",
        "training",
        "--updates",
        "20",
        "--warmup",
        "2",
        "--label",
        "ce_optimize",
        "--out",
        str(bench_out),
        "--overrides",
        *BENCHMARK_OVERRIDES,
    ]
    bench_proc = _run(bench_cmd, env=env)
    if bench_proc.returncode != 0:
        payload = {
            "tests_passed": tests_passed,
            "benchmark_passed": 0,
            "snapshots_all_finite": 0,
            "default_backend_gpu": 0,
            "env_steps_per_sec": 0.0,
            "seconds_per_update_mean": 0.0,
            "rollout_collect_seconds_per_update_mean": 0.0,
            "ppo_seconds_per_update_mean": 0.0,
            "compile_seconds_to_update_3": 0.0,
            "error": (bench_proc.stderr or bench_proc.stdout or "")[-2000:],
        }
        print(json.dumps(payload))
        return 1

    try:
        bench = json.loads(bench_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"tests_passed": tests_passed, "benchmark_passed": 0, "error": str(exc)}))
        return 1

    default_backend = str(bench.get("default_backend", ""))
    devices = bench.get("devices", [])
    default_backend_gpu = 1 if default_backend == "gpu" and devices else 0

    snapshots_finite = bench.get("snapshots_all_finite")
    if snapshots_finite is None:
        snapshots_all_finite = 1
    else:
        snapshots_all_finite = 1 if bool(snapshots_finite) else 0

    payload = {
        "tests_passed": tests_passed,
        "benchmark_passed": 1,
        "snapshots_all_finite": snapshots_all_finite,
        "default_backend_gpu": default_backend_gpu,
        "env_steps_per_sec": float(bench.get("env_steps_per_sec", 0.0)),
        "seconds_per_update_mean": float(bench.get("seconds_per_update_mean", 0.0)),
        "rollout_collect_seconds_per_update_mean": float(
            bench.get("rollout_collect_seconds_per_update_mean") or 0.0
        ),
        "ppo_seconds_per_update_mean": float(
            bench.get("ppo_seconds_per_update_mean") or 0.0
        ),
        "compile_seconds_to_update_3": float(bench.get("compile_seconds_to_update_3") or 0.0),
        "samples_per_sec": float(bench.get("samples_per_sec", 0.0)),
    }
    print(json.dumps(payload))
    return 0 if tests_passed and payload["benchmark_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
