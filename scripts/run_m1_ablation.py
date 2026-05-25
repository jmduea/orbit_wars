#!/usr/bin/env python3
"""Run M1 factored pointer ablation training arms and aggregate JSONL metrics.

Both arms use ``planet_graph_transformer``; only ``pointer_decoder`` differs
(joint flat vs factorized top-K). Default mode runs in-process so JAX compiles
once per decoder path (not once per subprocess).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = REPO_ROOT / "artifacts/m1/baseline_pin.json"
RESULTS_PATH = REPO_ROOT / "docs/m1-factored-pointer-results.md"
METRICS_DIR = REPO_ROOT / "artifacts/m1"


@dataclass(slots=True)
class RunSpec:
    """Single M1 ablation training run."""

    arm: str
    model: str
    pointer_decoder: str
    seed: int
    updates: int


def _load_pin() -> dict:
    return json.loads(PIN_PATH.read_text(encoding="utf-8"))


def _preflight_gpu() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.jax.device import ensure_cuda_jax_if_nvidia_present

    ensure_cuda_jax_if_nvidia_present()
    import jax

    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    print(f"JAX devices: {devices}", flush=True)
    if not gpu_devices:
        print(
            "WARNING: No JAX GPU devices visible. Training will be CPU-bound and "
            "very slow. Install jax[cuda13] via `uv sync` or set "
            "ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1 to bypass the guard.",
            flush=True,
        )


def _metrics_path(spec: RunSpec) -> Path:
    return METRICS_DIR / f"metrics_{spec.arm}_s{spec.seed}.json"


def _compose_overrides(spec: RunSpec, pin: dict) -> list[str]:
    run_name = f"m1-{spec.arm}-s{spec.seed}-u{spec.updates}"
    overrides = [
        f"model={spec.model}",
        f"seed={spec.seed}",
        f"run_name={run_name}",
        *pin["shared_overrides"],
    ]
    overrides = [
        item for item in overrides if not item.startswith("training.total_updates=")
    ]
    overrides.append(f"training.total_updates={spec.updates}")
    return overrides


def _run_training_subprocess(spec: RunSpec, pin: dict) -> Path:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.train",
        *_compose_overrides(spec, pin),
    ]
    print(f"Running (subprocess): {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return _find_log_for_spec(spec, pin)


def _run_training_in_process(spec: RunSpec, pin: dict) -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.config import compose_hydra_train_config
    from src.jax.train import run_jax_training

    overrides = _compose_overrides(spec, pin)
    print(f"Running (in-process): {' '.join(overrides)}", flush=True)
    cfg = compose_hydra_train_config(overrides)
    log_path = run_jax_training(cfg, resume_checkpoint=None)
    return log_path.resolve()


def _find_log_for_spec(spec: RunSpec, pin: dict) -> Path:
    """Locate the JSONL log for a completed run via run manifests."""

    runs_root = REPO_ROOT / "outputs/campaigns/default/runs"
    arm_meta = pin["arms"][spec.arm]
    expected_decoder = arm_meta["pointer_decoder"]
    expected_architecture = "planet_graph_transformer"
    best: tuple[str, Path] | None = None
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if int(manifest.get("seed", -1)) != spec.seed:
            continue
        if str(manifest.get("model_compatibility_family")) != expected_architecture:
            continue
        manifest_decoder = manifest.get("pointer_decoder")
        if manifest_decoder is None or str(manifest_decoder) != expected_decoder:
            continue
        run_name = str(manifest.get("run_name", ""))
        if f"u{spec.updates}" not in run_name:
            continue
        log_rel = manifest.get("paths", {}).get("log_path")
        if not isinstance(log_rel, str):
            continue
        log_path = (REPO_ROOT / log_rel).resolve()
        if not log_path.is_file():
            continue
        created_at = str(manifest.get("created_at", manifest_path.stat().st_mtime))
        if best is None or created_at > best[0]:
            best = (created_at, log_path)
    if best is None:
        raise FileNotFoundError(
            f"No JSONL log found for arm={spec.arm!r} seed={spec.seed} "
            f"architecture={expected_architecture!r} pointer_decoder={expected_decoder!r} "
            f"updates={spec.updates}"
        )
    return best[1]


def _parse_jsonl_metrics(log_path: Path, *, window_start: int, window_end: int) -> dict:
    records: list[dict] = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    window = [
        r for r in records if window_start <= int(r.get("update", 0)) <= window_end
    ]
    if not window:
        window = records[-5:]

    def _mean(key: str) -> float | None:
        vals = [float(r[key]) for r in window if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    max_moves_k = 3
    mean_active = _mean("mean_active_launches_per_turn")
    stop_utilization = (
        mean_active / max_moves_k if mean_active is not None else None
    )

    return {
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "episode_reward_mean": _mean("episode_reward_mean"),
        "average_episode_reward": _mean("average_episode_reward"),
        "rollout_env_steps_per_sec": _mean("rollout_env_steps_per_sec"),
        "trajectory_shield_legal_non_noop_rate": _mean(
            "trajectory_shield_legal_non_noop_rate"
        ),
        "stop_rate": _mean("stop_rate"),
        "mean_active_launches_per_turn": mean_active,
        "stop_utilization_ratio": stop_utilization,
        "overall_win_rate": _mean("overall_win_rate"),
        "updates_logged": len(records),
    }


def _reward_for_r1(metrics: dict) -> float | None:
    reward = metrics.get("episode_reward_mean")
    if reward is not None:
        return reward
    return metrics.get("average_episode_reward")


def _write_results(rows: list[dict], pin: dict) -> None:
    window = pin.get("window_updates", [450, 500])
    lines = [
        "# M1 Factored Pointer Ablation Results",
        "",
        f"**Pinned commit:** `{pin['baseline_commit']}`",
        "**Encoder:** `planet_graph_transformer` (both arms)",
        f"**Feature schema:** v{pin['feature_metadata']['schema_version']} / E={pin['feature_metadata']['edge_feature_dim']}",
        "**Profile:** `training=ablation_m2`, `telemetry=ablation_m1`, `format=mix_2p_4p_16env`",
        "**Status:** Phase 4 ablation **pending**",
        "",
        "## Runs",
        "",
        "| Arm | Decoder | Seed | Updates | Reward ({}–{}) | Env steps/s | Stop util | Log |".format(
            window[0], window[1]
        ),
        "|-----|---------|------|---------|----------------|-------------|-----------|-----|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {arm} | {decoder} | {seed} | {updates} | {reward} | {sps} | {stop} | `{log}` |".format(
                arm=row["arm"],
                decoder=row["pointer_decoder"],
                seed=row["seed"],
                updates=row["updates"],
                reward=_fmt(_reward_for_r1(metrics)),
                sps=_fmt(metrics.get("rollout_env_steps_per_sec")),
                stop=_fmt(metrics.get("stop_utilization_ratio")),
                log=metrics["log_path"],
            )
        )
    lines.extend(
        [
            "",
            "## Gate evaluation",
            "",
            "_Run `uv run python scripts/evaluate_m1_gates.py` after all runs complete._",
            "",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--arms",
        choices=["both", "joint", "factorized"],
        default="both",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=450,
        help="Final-window start update for R1 (inclusive).",
    )
    parser.add_argument(
        "--window-end",
        type=int,
        default=500,
        help="Final-window end update for R1 (inclusive).",
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Spawn a fresh `src.train` subprocess per run (slow; recompiles each time).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs when artifacts/m1/metrics_{arm}_s{seed}.json already exists.",
    )
    parser.add_argument(
        "--no-gpu-preflight",
        action="store_true",
        help="Skip JAX GPU device check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_gpu_preflight:
        _preflight_gpu()

    pin = _load_pin()
    seeds = args.seeds if args.seeds is not None else pin["seeds"]
    specs: list[RunSpec] = []
    if args.arms in {"both", "joint"}:
        arm_meta = pin["arms"]["joint_flat"]
        for seed in seeds:
            specs.append(
                RunSpec(
                    arm="joint_flat",
                    model=arm_meta["model"],
                    pointer_decoder=arm_meta["pointer_decoder"],
                    seed=seed,
                    updates=args.updates,
                )
            )
    if args.arms in {"both", "factorized"}:
        arm_meta = pin["arms"]["factorized_topk"]
        for seed in seeds:
            specs.append(
                RunSpec(
                    arm="factorized_topk",
                    model=arm_meta["model"],
                    pointer_decoder=arm_meta["pointer_decoder"],
                    seed=seed,
                    updates=args.updates,
                )
            )

    run_fn = _run_training_subprocess if args.subprocess else _run_training_in_process
    rows: list[dict] = []
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        metrics_path = _metrics_path(spec)
        if args.skip_existing and metrics_path.is_file():
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(cached)
            print(f"Skipping existing {metrics_path.name}", flush=True)
            continue

        log_path = run_fn(spec, pin)
        metrics = _parse_jsonl_metrics(
            log_path, window_start=args.window_start, window_end=args.window_end
        )
        row = {
            "arm": spec.arm,
            "model": spec.model,
            "pointer_decoder": spec.pointer_decoder,
            "seed": spec.seed,
            "updates": spec.updates,
            "metrics": metrics,
        }
        rows.append(row)
        metrics_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    _write_results(rows, pin)
    print(f"Wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
