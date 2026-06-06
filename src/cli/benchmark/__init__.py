"""``ow benchmark`` CLI package."""

from __future__ import annotations

from src.cli.benchmark.calibrate import run_calibrate_cli
from src.cli.benchmark.calibrate_seed import run_calibrate_seed_scheduler_cli
from src.cli.benchmark.calibrate_qualifier_seeds import (
    run_calibrate_qualifier_seeds_cli,
)
from src.cli.benchmark.calibrate_unified import run_calibrate_unified_tournament_cli
from src.cli.benchmark.common import (
    LEARN_PROOF_PRIMITIVES,
    _git_head_sha,
    _init_benchmark_runtime,
    print_benchmark_help,
)
from src.cli.benchmark.env_parity_ab import run_env_parity_ab_cli
from src.cli.benchmark.factorized import run_factorized_sampler_cli
from src.cli.benchmark.admission_throughput import run_admission_throughput_cli
from src.cli.benchmark.gate import run_gate_cli
from src.cli.benchmark.learn_proof import run_learn_proof_cli
from src.cli.benchmark.parser import build_parser
from src.cli.benchmark.planet_flow import (
    run_planet_flow_noop_smoke_cli,
    run_shortlist_planet_flow_sweep_cli,
)
from src.cli.benchmark.sanity import run_sanity_cli
from src.cli.benchmark.tournament_proof import run_tournament_proof_cli
from src.cli.benchmark.training import run_training_benchmark_cli

__all__ = [
    "LEARN_PROOF_PRIMITIVES",
    "_git_head_sha",
    "_init_benchmark_runtime",
    "build_parser",
    "main",
    "print_benchmark_help",
    "run_calibrate_cli",
    "run_calibrate_seed_scheduler_cli",
    "run_calibrate_unified_tournament_cli",
    "run_env_parity_ab_cli",
    "run_factorized_sampler_cli",
    "run_admission_throughput_cli",
    "run_gate_cli",
    "run_learn_proof_cli",
    "run_planet_flow_noop_smoke_cli",
    "run_shortlist_planet_flow_sweep_cli",
    "run_sanity_cli",
    "run_tournament_proof_cli",
    "run_training_benchmark_cli",
]

def main(argv: list[str] | None = None) -> int:
    if not argv:
        print_benchmark_help()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        print_benchmark_help()
        return 0
    match args.command:
        case "training":
            return run_training_benchmark_cli(args)
        case "sanity":
            return run_sanity_cli(args)
        case "learn-proof":
            return run_learn_proof_cli(args)
        case "calibrate":
            return run_calibrate_cli(args)
        case "calibrate-seed-scheduler":
            return run_calibrate_seed_scheduler_cli(args)
        case "calibrate-unified-tournament":
            return run_calibrate_unified_tournament_cli(args)
        case "calibrate-qualifier-seeds":
            return run_calibrate_qualifier_seeds_cli(args)
        case "shortlist-planet-flow-sweep":
            return run_shortlist_planet_flow_sweep_cli(args)
        case "planet-flow-noop-smoke":
            return run_planet_flow_noop_smoke_cli(args)
        case "factorized-sampler":
            return run_factorized_sampler_cli(args)
        case "env-parity-ab":
            return run_env_parity_ab_cli(args)
        case "gate":
            return run_gate_cli(args)
        case "admission-throughput":
            return run_admission_throughput_cli(args)
        case "tournament-proof":
            return run_tournament_proof_cli(args)
        case _:
            parser.error(f"unknown benchmark command: {args.command!r}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
