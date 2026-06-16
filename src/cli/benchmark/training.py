"""``ow benchmark training`` command."""

from __future__ import annotations

import argparse
import json
import sys

from src.cli.benchmark.common import _git_head_sha, _init_benchmark_runtime


def run_training_benchmark_cli(args: argparse.Namespace) -> int:
    import jax
    from src.benchmark.production import rollout_group_summary
    from src.benchmark.training import (
        E2E_THROUGHPUT_GATE,
        aggregate_e2e_run_payloads,
        check_baseline_device_match,
        compare_e2e_throughput_to_baseline,
        compose_benchmark_config,
        default_benchmark_updates,
        derive_e2e_pass_band,
        format_profile_name,
        load_e2e_baseline,
        resolve_benchmark_overrides,
        resolve_e2e_measured_for_gate,
        resolve_e2e_pass_band,
        run_training_benchmark,
        training_benchmark_payload,
    )

    _init_benchmark_runtime()
    updates = (
        int(args.updates)
        if args.updates is not None
        else default_benchmark_updates(preset=args.preset)
    )
    overrides = resolve_benchmark_overrides(
        preset=args.preset,
        overrides=args.overrides,
    )
    cfg = compose_benchmark_config(overrides)
    group_specs = rollout_group_summary(cfg)

    run_payloads: list[dict[str, object]] = []
    repeats = max(int(args.repeats), 1)
    commit_sha = _git_head_sha()
    for repeat_idx in range(repeats):
        label = args.label if repeats == 1 else f"{args.label}_r{repeat_idx + 1}"
        result = run_training_benchmark(
            cfg,
            label=label,
            overrides=tuple(overrides),
            warmup=args.warmup,
            updates=updates,
            snapshot_updates=frozenset(args.snapshot_updates),
            detailed_timing=bool(args.detailed_timing),
            profile_dir=args.profile_dir,
        )
        payload = training_benchmark_payload(result)
        payload.update(
            {
                "commit_sha": commit_sha,
                "tier": args.tier,
                "jax_version": jax.__version__,
                "format": format_profile_name(overrides),
                "rollout_groups": [dict(group) for group in group_specs],
                "rollout_microbatch_envs": (
                    int(cfg.training.rollout_microbatch_envs)
                    if cfg.training.rollout_microbatch_envs is not None
                    else None
                ),
                "gate": E2E_THROUGHPUT_GATE
                if args.preset == "primary"
                else "stability",
            }
        )
        run_payloads.append(payload)

    if repeats == 1:
        output_payload: dict[str, object] = run_payloads[0]
    else:
        aggregate = aggregate_e2e_run_payloads(run_payloads)
        within_pct = (
            float(args.assert_within_pct)
            if args.assert_within_pct is not None
            else 10.0
        )
        pass_band = derive_e2e_pass_band(aggregate, within_pct=within_pct)
        output_payload = {
            "gate": E2E_THROUGHPUT_GATE,
            "label": args.label,
            "commit_sha": commit_sha,
            "jax_version": jax.__version__,
            "overrides": overrides,
            "updates": updates,
            "warmup": args.warmup,
            "repeats": repeats,
            "runs": run_payloads,
            "aggregate": aggregate,
            "pass_band": pass_band,
        }

    if repeats == 1 and args.preset == "planet_flow_p0":
        required_control_metrics = (
            "planet_flow_control_emitted_launch_count",
            "planet_flow_control_emitted_ship_mass_rate",
            "planet_flow_emitted_launch_count_delta_vs_control",
        )
        missing = [
            key for key in required_control_metrics if run_payloads[0].get(key) is None
        ]
        if missing:
            print(
                "Planet Flow benchmark proof is missing compiler-control metrics: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 1

    if args.baseline is not None or args.assert_within_pct is not None:
        if args.baseline is None:
            print("--assert-within-pct requires --baseline", file=sys.stderr)
            return 1
        try:
            baseline = load_e2e_baseline(args.baseline)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        measured = resolve_e2e_measured_for_gate(
            repeats=repeats,
            run_payloads=run_payloads,
            aggregate=output_payload.get("aggregate"),
        )
        device_ok, device_message = check_baseline_device_match(
            baseline,
            devices=run_payloads[0]["devices"],  # type: ignore[arg-type]
            default_backend=str(run_payloads[0]["default_backend"]),
            mode=str(args.device_check),
            force=bool(args.force),
        )
        if device_message:
            print(device_message, file=sys.stderr)
        if not device_ok:
            return 1
        pass_band = resolve_e2e_pass_band(
            baseline,
            within_pct=args.assert_within_pct,
        )
        passed, failures = compare_e2e_throughput_to_baseline(
            measured,
            pass_band=pass_band,
        )
        output_payload["baseline_path"] = str(args.baseline)
        output_payload["pass_band_applied"] = pass_band
        output_payload["measured_for_gate"] = measured
        output_payload["gate_passed"] = passed
        if failures:
            output_payload["gate_failures"] = failures
        if not passed:
            for reason in failures:
                print(reason, file=sys.stderr)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(output_payload, indent=2) + "\n", encoding="utf-8"
            )
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_payload, sort_keys=True))
    if (
        repeats == 1
        and args.assert_min_env_steps_per_sec is not None
        and float(run_payloads[0]["env_steps_per_sec"])
        < args.assert_min_env_steps_per_sec
    ):
        print(
            "env_steps_per_sec "
            f"{float(run_payloads[0]['env_steps_per_sec']):.3f} < "
            f"{args.assert_min_env_steps_per_sec:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0
