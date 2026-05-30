"""Build a VRAM comfort profile from W&B run data.

Prefers measured ``gpu_memory_peak_gb`` telemetry when present. Falls back to the
legacy ``group_envs * rollout_steps`` pressure proxy for older campaigns such as
``sps_experiment`` that predate GPU memory logging.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "benchmarks" / "vram-profile-sps-experiment.json"
DEFAULT_MD = REPO_ROOT / "docs" / "benchmarks" / "vram-profile-sps-experiment.md"
DEFAULT_CALIBRATION = (
    REPO_ROOT / "docs" / "benchmarks" / "vram-profile-calibration.json"
)

TARGET_GPUS: tuple[tuple[str, float], ...] = (
    ("NVIDIA GeForce RTX 5080 (workstation)", 16.0),
    ("NVIDIA Tesla P100 (Kaggle)", 16.0),
)

SAFETY_MARGIN = 0.85
COMFORT_UTILIZATION = 0.90


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Normalized row extracted from one W&B run or local JSONL calibration."""

    run_id: str
    name: str
    state: str
    gpu_name: str | None
    gpu_memory_gb: float | None
    gpu_memory_peak_gb: float | None
    opponent: str | None
    format_label: str
    group_envs: int
    rollout_steps: int
    rollout_microbatch_envs: int
    total_updates: int
    pressure_index: int
    env_steps_per_sec: float | None
    samples_per_sec: float | None
    update_seconds: float | None
    elapsed_seconds: float | None
    compile_seconds_est: float | None
    measurement_source: str


def _parse_name(name: str) -> dict[str, str | int]:
    parsed: dict[str, str | int] = {}
    for key, pattern in (
        ("opponent", r"-(noop|selfplay|random)-"),
        ("env_label", r"-env(\d+)-"),
        ("seed", r"-s(\d+)-"),
    ):
        match = re.search(pattern, name)
        if match:
            parsed[key] = match.group(1)
    update_match = re.search(r"-u(\d+)-", name)
    if update_match:
        parsed["updates"] = int(update_match.group(1))
    if "mix2p4p" in name:
        parsed["format_label"] = "mix2p4p"
    elif "2ponly" in name or "2p-only" in name:
        parsed["format_label"] = "2p_only"
    elif "4ponly" in name or "4p-only" in name:
        parsed["format_label"] = "4p_only"
    else:
        parsed["format_label"] = "unknown"
    return parsed


def _group_env_count(config: dict[str, Any]) -> int:
    groups = config.get("format.rollout_groups")
    fallback = int(config.get("training.num_envs") or 0)
    if not isinstance(groups, list):
        return fallback
    total = 0
    for group in groups:
        if not isinstance(group, dict):
            continue
        total += int(group.get("num_envs", fallback) or 0)
    return total or fallback


def _gpu_metadata(metadata: dict[str, Any]) -> tuple[str | None, float | None]:
    nvidia = metadata.get("gpu_nvidia")
    if isinstance(nvidia, list) and nvidia:
        row = nvidia[0]
        if isinstance(row, dict):
            name = row.get("name")
            memory_total = row.get("memoryTotal")
            memory_gb = None
            if memory_total is not None:
                memory_gb = float(memory_total) / (1024.0**3)
            return (str(name) if name else None, memory_gb)
    gpu_name = metadata.get("gpu")
    return (str(gpu_name) if gpu_name else None, None)


def _estimate_compile_seconds(
    elapsed_seconds: float | None,
    update_seconds: float | None,
    total_updates: int,
) -> float | None:
    if elapsed_seconds is None or update_seconds is None or total_updates <= 0:
        return None
    estimate = elapsed_seconds - (update_seconds * total_updates)
    return max(float(estimate), 0.0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _run_peak_vram_gb(run: Any) -> float | None:
    summary = dict(getattr(run, "summary", {}) or {})
    peak = _optional_float(summary.get("gpu_memory_peak_gb"))
    if peak is not None:
        return peak
    try:
        history = run.history(samples=500, keys=["gpu_memory_peak_gb", "gpu_memory_used_gb"])
    except Exception:
        return None
    if history.empty:
        return None
    peaks: list[float] = []
    for column in ("gpu_memory_peak_gb", "gpu_memory_used_gb"):
        if column in history.columns:
            values = [
                float(value)
                for value in history[column].dropna().tolist()
                if _optional_float(value) is not None
            ]
            if values:
                peaks.append(max(values))
    return max(peaks) if peaks else None


def fetch_run_records(
    *,
    entity: str | None,
    project: str,
    group: str,
    per_page: int = 200,
) -> list[RunRecord]:
    """Download and normalize runs for one W&B group."""

    import wandb  # type: ignore

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path, filters={"group": group}, per_page=per_page))
    records: list[RunRecord] = []
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        metadata = dict(getattr(run, "metadata", {}) or {})
        parsed = _parse_name(str(run.name))
        group_envs = _group_env_count(config)
        rollout_steps = int(config.get("training.rollout_steps") or 0)
        microbatch = int(config.get("training.rollout_microbatch_envs") or 0)
        total_updates = int(
            config.get("training.total_updates") or parsed.get("updates") or 0
        )
        gpu_name, gpu_memory_gb = _gpu_metadata(metadata)
        peak_gb = _run_peak_vram_gb(run)
        elapsed = summary.get("elapsed_seconds")
        update_seconds = summary.get("update_seconds")
        measurement_source = "wandb_gpu_memory_peak_gb" if peak_gb is not None else "pressure_proxy"
        records.append(
            RunRecord(
                run_id=str(run.id),
                name=str(run.name),
                state=str(run.state),
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory_gb,
                gpu_memory_peak_gb=peak_gb,
                opponent=str(parsed.get("opponent")) if parsed.get("opponent") else None,
                format_label=str(parsed.get("format_label", "unknown")),
                group_envs=group_envs,
                rollout_steps=rollout_steps,
                rollout_microbatch_envs=microbatch,
                total_updates=total_updates,
                pressure_index=group_envs * rollout_steps,
                env_steps_per_sec=_optional_float(summary.get("env_steps_per_sec")),
                samples_per_sec=_optional_float(summary.get("samples_per_sec")),
                update_seconds=_optional_float(update_seconds),
                elapsed_seconds=_optional_float(elapsed),
                compile_seconds_est=_estimate_compile_seconds(
                    _optional_float(elapsed),
                    _optional_float(update_seconds),
                    total_updates,
                ),
                measurement_source=measurement_source,
            )
        )
    return records


def load_calibration_jsonl(path: Path) -> list[RunRecord]:
    """Load measured VRAM peaks from a local training JSONL log."""

    if not path.exists():
        return []
    manifest_path = path.parent.parent / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    overrides: dict[str, Any] = {}
    overrides_path = manifest.get("hydra_overrides_path")
    if isinstance(overrides_path, str):
        override_file = Path(overrides_path)
        if override_file.exists():
            import yaml

            raw = yaml.safe_load(override_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and "=" in item:
                        key, value = item.split("=", 1)
                        overrides[key] = value

    updates: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event"):
            continue
        if "update" in row:
            updates.append(row)
    if not updates:
        return []

    last = updates[-1]
    parsed = _parse_name(str(manifest.get("run_name", path.stem)))
    peak = max(
        (
            float(row["gpu_memory_peak_gb"])
            for row in updates
            if _optional_float(row.get("gpu_memory_peak_gb")) is not None
        ),
        default=max(
            (
                float(row["gpu_memory_used_gb"])
                for row in updates
                if _optional_float(row.get("gpu_memory_used_gb")) is not None
            ),
            default=0.0,
        ),
    )
    gpu_name = manifest.get("gpu_name") or last.get("gpu_name")
    gpu_total = _optional_float(
        manifest.get("gpu_memory_total_gb") or last.get("gpu_memory_total_gb")
    )
    rollout_steps = int(overrides.get("training.rollout_steps", 128))
    microbatch = int(overrides.get("training.rollout_microbatch_envs", 16))
    group_envs = int(overrides.get("training.num_envs", 32))
    return [
        RunRecord(
            run_id=str(manifest.get("run_id", path.stem)),
            name=str(manifest.get("run_name", path.stem)),
            state="finished",
            gpu_name=str(gpu_name) if gpu_name else None,
            gpu_memory_gb=gpu_total,
            gpu_memory_peak_gb=peak or None,
            opponent=str(parsed.get("opponent")) if parsed.get("opponent") else None,
            format_label=str(parsed.get("format_label", "mix2p4p")),
            group_envs=group_envs,
            rollout_steps=rollout_steps,
            rollout_microbatch_envs=microbatch,
            total_updates=int(last.get("update", 0)),
            pressure_index=group_envs * rollout_steps,
            env_steps_per_sec=_optional_float(last.get("env_steps_per_sec")),
            samples_per_sec=_optional_float(last.get("samples_per_sec")),
            update_seconds=_optional_float(last.get("update_seconds")),
            elapsed_seconds=_optional_float(last.get("elapsed_seconds")),
            compile_seconds_est=None,
            measurement_source="local_jsonl",
        )
    ]


def _scale_pressure(source_gb: float, target_gb: float, pressure: int) -> int:
    ratio = (target_gb / source_gb) * SAFETY_MARGIN
    return max(int(pressure * ratio), 1)


def _round_down_to(value: int, step: int) -> int:
    if step <= 0:
        return value
    return max(step, (value // step) * step)


def _suggest_rollout_shape(pressure_ceiling: int, group_envs: int) -> tuple[int, int]:
    rollout_steps = max(min(pressure_ceiling // max(group_envs, 1), 1024), 64)
    rollout_steps = _round_down_to(rollout_steps, 64)
    return group_envs, rollout_steps


def _measured_comfort_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    measured = [
        row
        for row in records
        if row.state == "finished" and row.gpu_memory_peak_gb is not None
    ]
    by_gpu: dict[tuple[str | None, float | None], list[RunRecord]] = defaultdict(list)
    for row in measured:
        by_gpu[(row.gpu_name, row.gpu_memory_gb)].append(row)
    rows: list[dict[str, Any]] = []
    for (gpu_name, gpu_total), items in sorted(by_gpu.items()):
        peaks = [float(row.gpu_memory_peak_gb) for row in items if row.gpu_memory_peak_gb]
        max_peak = max(peaks)
        comfort_gb = max_peak / COMFORT_UTILIZATION
        rows.append(
            {
                "gpu_name": gpu_name,
                "gpu_memory_total_gb": gpu_total,
                "run_count": len(items),
                "max_peak_gb": max_peak,
                "comfort_ceiling_gb": comfort_gb,
                "runs": [
                    {
                        "name": row.name,
                        "gpu_memory_peak_gb": row.gpu_memory_peak_gb,
                        "rollout_steps": row.rollout_steps,
                        "rollout_microbatch_envs": row.rollout_microbatch_envs,
                        "group_envs": row.group_envs,
                        "pressure_index": row.pressure_index,
                    }
                    for row in sorted(items, key=lambda item: item.gpu_memory_peak_gb or 0.0, reverse=True)
                ],
            }
        )
    return rows


def build_profile(
    records: list[RunRecord],
    *,
    source_memory_gb: float = 40.0,
) -> dict[str, Any]:
    """Aggregate campaign rows into a JSON-serializable profile."""

    finished = [row for row in records if row.state == "finished"]
    measured_count = sum(1 for row in records if row.gpu_memory_peak_gb is not None)
    gpu_counts = Counter(
        (row.gpu_name, row.gpu_memory_gb) for row in records if row.gpu_name
    )
    by_bucket: dict[tuple[str, str | None, int], list[RunRecord]] = defaultdict(list)
    for row in finished:
        by_bucket[(row.format_label, row.opponent, row.group_envs)].append(row)

    comfort: list[dict[str, Any]] = []
    for (format_label, opponent, group_envs), rows in sorted(by_bucket.items()):
        max_pressure = max(row.pressure_index for row in rows)
        best = max(
            rows,
            key=lambda row: (
                row.env_steps_per_sec or 0.0,
                row.pressure_index,
            ),
        )
        comfort.append(
            {
                "format_label": format_label,
                "opponent": opponent,
                "group_envs": group_envs,
                "max_success_pressure_index": max_pressure,
                "best_run": {
                    "name": best.name,
                    "rollout_steps": best.rollout_steps,
                    "rollout_microbatch_envs": best.rollout_microbatch_envs,
                    "pressure_index": best.pressure_index,
                    "env_steps_per_sec": best.env_steps_per_sec,
                    "compile_seconds_est": best.compile_seconds_est,
                    "gpu_memory_peak_gb": best.gpu_memory_peak_gb,
                },
            }
        )

    measured_comfort = _measured_comfort_rows(records)
    scaled_targets: list[dict[str, Any]] = []
    measured_runs = [
        row for row in records if row.gpu_memory_peak_gb is not None and row.state == "finished"
    ]
    if measured_runs:
        ref_run = max(measured_runs, key=lambda row: row.gpu_memory_peak_gb or 0.0)
        ref_peak = float(ref_run.gpu_memory_peak_gb or 0.0)
        for target_name, target_gb in TARGET_GPUS:
            allowed_peak = target_gb * COMFORT_UTILIZATION
            scale = min(allowed_peak / ref_peak, 1.0) if ref_peak > 0 else 1.0
            if scale >= 0.95:
                envs = ref_run.group_envs
                rollout_steps = ref_run.rollout_steps
                microbatch = ref_run.rollout_microbatch_envs
            else:
                scaled_pressure = max(
                    int(ref_run.pressure_index * scale * SAFETY_MARGIN),
                    4096,
                )
                envs, rollout_steps = _suggest_rollout_shape(
                    scaled_pressure,
                    ref_run.group_envs,
                )
                microbatch = ref_run.rollout_microbatch_envs
            scaled_targets.append(
                {
                    "target_gpu": target_name,
                    "target_memory_gb": target_gb,
                    "reference_gpu": ref_run.gpu_name,
                    "reference_peak_gb": ref_peak,
                    "reference_total_gb": ref_run.gpu_memory_gb,
                    "reference_shape": {
                        "group_envs": ref_run.group_envs,
                        "rollout_steps": ref_run.rollout_steps,
                        "rollout_microbatch_envs": ref_run.rollout_microbatch_envs,
                    },
                    "method": "measured_peak_ratio",
                    "scale_factor": scale,
                    "suggested_group_envs": envs,
                    "suggested_rollout_steps": rollout_steps,
                    "suggested_rollout_microbatch_envs": microbatch,
                }
            )
    else:
        reference = [
            row
            for row in comfort
            if row["format_label"] == "mix2p4p"
            and row["opponent"] == "selfplay"
            and row["group_envs"] == 64
        ]
        ref_pressure = (
            reference[0]["max_success_pressure_index"] if reference else 49152
        )
        for target_name, target_gb in TARGET_GPUS:
            ceiling = _scale_pressure(source_memory_gb, target_gb, int(ref_pressure))
            envs, rollout_steps = _suggest_rollout_shape(ceiling, group_envs=64)
            scaled_targets.append(
                {
                    "target_gpu": target_name,
                    "target_memory_gb": target_gb,
                    "source_memory_gb": source_memory_gb,
                    "method": "pressure_proxy",
                    "safety_margin": SAFETY_MARGIN,
                    "reference_max_pressure_index": ref_pressure,
                    "scaled_pressure_ceiling": ceiling,
                    "suggested_group_envs": envs,
                    "suggested_rollout_steps": rollout_steps,
                    "suggested_rollout_microbatch_envs": 16,
                }
            )

    compile_values = [
        row.compile_seconds_est
        for row in finished
        if row.compile_seconds_est is not None
    ]
    throughput_values = [
        row.env_steps_per_sec for row in finished if row.env_steps_per_sec is not None
    ]
    peak_values = [
        float(row.gpu_memory_peak_gb)
        for row in finished
        if row.gpu_memory_peak_gb is not None
    ]

    return {
        "campaign_group": "sps_experiment",
        "run_count": len(records),
        "finished_count": len(finished),
        "crashed_count": sum(1 for row in records if row.state != "finished"),
        "measured_vram_run_count": measured_count,
        "source_gpu_counts": [
            {"gpu_name": name, "memory_gb": mem, "count": count}
            for (name, mem), count in gpu_counts.most_common()
        ],
        "notes": [
            "Prefer gpu_memory_peak_gb telemetry when present.",
            "Legacy sps_experiment runs fall back to pressure_index = group_envs * rollout_steps.",
            "Scaled targets use measured peak ratio when calibration data exists.",
            "Microbatch must divide every active rollout-group env count.",
        ],
        "aggregate": {
            "compile_seconds_est_mean": _mean(
                [value for value in compile_values if value is not None]
            ),
            "env_steps_per_sec_mean": _mean(
                [value for value in throughput_values if value is not None]
            ),
            "gpu_memory_peak_gb_max": max(peak_values) if peak_values else None,
            "gpu_memory_peak_gb_mean": _mean(peak_values),
        },
        "measured_comfort_by_gpu": measured_comfort,
        "comfort_by_bucket": comfort,
        "scaled_targets": scaled_targets,
        "runs": [asdict(row) for row in records],
    }


def render_markdown(profile: dict[str, Any]) -> str:
    """Render a human-readable benchmark note from profile JSON."""

    measured = profile.get("measured_vram_run_count", 0)
    lines = [
        "# VRAM profile — W&B + measured telemetry",
        "",
        "Closes [#123](https://github.com/jmduea/orbit_wars/issues/123).",
        "",
        "Primary campaign: W&B group `sps_experiment` (A100 40GB/80GB). Newer runs and",
        "local calibration logs include `gpu_memory_peak_gb` from training telemetry.",
        "",
        "## Summary",
        "",
        f"- Runs ingested: **{profile['run_count']}** ({profile['finished_count']} finished)",
        f"- Runs with measured VRAM: **{measured}**",
        "",
    ]
    agg = profile["aggregate"]
    if agg.get("gpu_memory_peak_gb_max") is not None:
        lines.append(
            f"- Peak observed VRAM (measured runs): **{agg['gpu_memory_peak_gb_max']:.2f} GB**"
        )
    else:
        lines.append(
            "- Peak observed VRAM: **n/a** (legacy campaign; see pressure-proxy tables)"
        )

    lines.extend(
        [
            "",
            "## Source hardware",
            "",
            "| GPU | Memory (GB) | Runs |",
            "|-----|-------------|------|",
        ]
    )
    for row in profile["source_gpu_counts"]:
        mem = row["memory_gb"]
        mem_text = f"{mem:.0f}" if mem is not None else "?"
        lines.append(f"| {row['gpu_name']} | {mem_text} | {row['count']} |")

    if profile.get("measured_comfort_by_gpu"):
        lines.extend(
            [
                "",
                "## Measured comfort (preferred)",
                "",
                "| GPU | Runs | Max peak (GB) | Comfort ceiling (90%) |",
                "|-----|------|---------------|------------------------|",
            ]
        )
        for row in profile["measured_comfort_by_gpu"]:
            lines.append(
                f"| {row['gpu_name']} | {row['run_count']} | "
                f"{row['max_peak_gb']:.2f} | {row['comfort_ceiling_gb']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Legacy proxy comfort (sps_experiment)",
            "",
            "Used when `gpu_memory_peak_gb` is absent.",
            "",
            "| Format | Opponent | Group envs | Max pressure | Best rs | Peak GB |",
            "|--------|----------|------------|--------------|---------|---------|",
        ]
    )
    for row in profile["comfort_by_bucket"]:
        best = row["best_run"]
        peak = best.get("gpu_memory_peak_gb")
        peak_text = f"{peak:.2f}" if peak is not None else "—"
        lines.append(
            f"| {row['format_label']} | {row['opponent'] or '—'} | "
            f"{row['group_envs']} | {row['max_success_pressure_index']} | "
            f"{best['rollout_steps']} | {peak_text} |"
        )

    lines.extend(
        [
            "",
            "## Workstation calibration (RTX 5080, measured)",
            "",
            "Local run `vram_profile_calibration/rtx5080-rs128-mb16`:",
            "`format=2p_4p_16env`, `training.rollout_steps=128`,",
            "`training.rollout_microbatch_envs=16`, selfplay, 3 updates.",
            "",
            "- **Peak VRAM: 14.64 GB** (~92% of 16 GB device)",
            "- At this shape the run is near comfort limit; prefer **rs=128** (not rs=256) on 16 GB GPUs",
            "- Full JSONL: `outputs/campaigns/vram_profile_calibration/runs/rtx5080-rs128-mb16/logs/`",
            "",
        ]
    )

    lines.extend(
        [
            "## Scaled targets (16GB class GPUs)",
            "",
            "| Target | Method | Suggested shape | Microbatch |",
            "|--------|--------|-----------------|------------|",
        ]
    )
    for row in profile["scaled_targets"]:
        lines.append(
            f"| {row['target_gpu']} | {row['method']} | "
            f"{row['suggested_group_envs']} env × rs={row['suggested_rollout_steps']} | "
            f"{row['suggested_rollout_microbatch_envs']} |"
        )

    lines.extend(
        [
            "",
            "## Telemetry fields",
            "",
            "- `gpu_memory_used_gb` — driver-reported use after each update",
            "- `gpu_memory_total_gb` — device capacity (GiB)",
            "- `gpu_memory_peak_gb` — running peak since run start",
            "- `gpu_name` — logged once at run start",
            "",
            "## Regenerate",
            "",
            "```bash",
            "uv run python scripts/summarize_sps_vram_profile.py --write-md \\",
            "  --calibration-jsonl path/to/run.jsonl",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _default_wandb_path() -> tuple[str | None, str]:
    from src.config import compose_hydra_train_config

    cfg = compose_hydra_train_config([])
    entity = cfg.telemetry.wandb.entity or None
    project = cfg.telemetry.wandb.project
    return entity, project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="sps_experiment")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-md", action="store_true")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--source-memory-gb", type=float, default=40.0)
    parser.add_argument(
        "--calibration-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional local training JSONL with gpu_memory_* metrics.",
    )
    parser.add_argument(
        "--write-calibration-json",
        type=Path,
        default=None,
        help="Write extracted calibration rows to JSON.",
    )
    args = parser.parse_args()

    default_entity, default_project = _default_wandb_path()
    entity = args.entity if args.entity is not None else default_entity
    project = args.project or default_project

    records = fetch_run_records(entity=entity, project=project, group=args.group)
    for path in args.calibration_jsonl:
        records.extend(load_calibration_jsonl(path))
    profile = build_profile(records, source_memory_gb=args.source_memory_gb)
    profile["wandb"] = {"entity": entity, "project": project, "group": args.group}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps({k: profile[k] for k in profile if k != "runs"}, indent=2))

    if args.write_calibration_json is not None:
        calibration_rows = [asdict(row) for row in records if row.gpu_memory_peak_gb is not None]
        args.write_calibration_json.write_text(
            json.dumps(calibration_rows, indent=2),
            encoding="utf-8",
        )

    if args.write_md:
        args.md_out.write_text(render_markdown(profile), encoding="utf-8")
        print(f"wrote markdown: {args.md_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
