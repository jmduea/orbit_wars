"""Benchmark the end-to-end JAX Orbit Wars rollout + PPO stack."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_runtime_env import (  # noqa: E402
    add_worker_cuda_library_path,
    isolate_worker_python_env,
    pin_jax_platform_from_kaggle,
)

pin_jax_platform_from_kaggle()
isolate_worker_python_env()
add_worker_cuda_library_path()

# Defensive guard: older packaged helpers used JAX_PLATFORMS=gpu on NVIDIA,
# which makes JAX attempt ROCm on Kaggle.  Calibration must use CUDA explicitly.
import os  # noqa: E402

if os.environ.get("KAGGLE_ACCELERATOR_ID", "").strip().lower().startswith("nvidia"):
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import argparse
import json
from copy import deepcopy

from src.benchmark.production import (  # noqa: E402
    production_benchmark_payload,
    run_production_benchmark,
)
from src.config import compose_hydra_train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line options for end-to-end JAX RL benchmarking."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["model=attention"],
        help="Hydra overrides for the benchmark config, such as model=attention training.total_updates=10.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["mlp", "attention", "transformer"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the production-aligned JAX rollout/update benchmark."""

    args = parse_args()
    cfg = deepcopy(compose_hydra_train_config(list(args.overrides)))
    if args.num_envs is not None:
        cfg.training.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.training.rollout_steps = args.rollout_steps
    if args.architecture is not None:
        cfg.model.architecture = args.architecture

    result = run_production_benchmark(cfg, warmup=args.warmup, updates=args.updates)
    print(json.dumps(production_benchmark_payload(result), sort_keys=True))


if __name__ == "__main__":
    main()
