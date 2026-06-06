"""Noop-only smoke runs for Planet Flow sweep shortlist finalists."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.benchmark.planet_flow.shortlist import rank_eligible_entries
from src.jax.preflight import (
    PreflightVerdict,
    _gate_specs,
    evaluate_gate_records,
    gate_evaluation_to_dict,
    latest_run_dir,
)
from src.jax.preflight_calibration import (
    default_calibration_json_path,
    read_jsonl_records,
    run_ow_train,
)

PLANET_FLOW_SMOKE_TRAIN_BASE: tuple[str, ...] = (
    "model=planet_flow_target_heatmap",
    "training=planet_flow",
    "opponents=noop_only",
    "training.total_updates=200",
    "curriculum=off",
    "telemetry.wandb.enabled=false",
    "artifacts=planet_flow_proof",
    "artifacts.artifact_pipeline.enabled=true",
    "telemetry.metric_groups.action_decision=true",
    "task=shield_cheap",
)


@dataclass(frozen=True, slots=True)
class SmokeConfigResult:
    run_id: str
    campaign: str
    train_overrides: tuple[str, ...]
    verdict: str
    reasons: tuple[str, ...]
    run_dir: str | None
    log_path: str | None
    gate: dict[str, object]


def _load_shortlist(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Shortlist JSON must be an object: {path}")
    return payload


def smoke_train_overrides(
    entry: dict[str, object],
    *,
    output_root: Path,
    campaign: str,
) -> list[str]:
    overrides = list(entry.get("train_overrides") or [])
    return [
        f"output.root={output_root.as_posix()}",
        f"output.campaign={campaign}",
        *PLANET_FLOW_SMOKE_TRAIN_BASE,
        *overrides,
    ]


def run_planet_flow_noop_smoke(
    shortlist_path: Path,
    *,
    top_k: int = 3,
    output_root: Path = Path("outputs"),
    repo_root: Path,
    thresholds_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    shortlist = _load_shortlist(shortlist_path)
    eligible = shortlist.get("eligible")
    if not isinstance(eligible, list) or not eligible:
        raise ValueError(
            "Shortlist has no eligible entries; re-run shortlist or sweep v3."
        )
    ranked = rank_eligible_entries(
        [dict(entry) for entry in eligible if isinstance(entry, dict)]
    )
    finalists = ranked[: max(int(top_k), 1)]
    specs = _gate_specs(
        "planet_flow_target_heatmap",
        thresholds_path=thresholds_path or default_calibration_json_path(repo_root),
    )
    beat_noop = specs["beat_noop"]
    results: list[dict[str, object]] = []
    recommended: dict[str, object] | None = None
    for index, entry in enumerate(finalists):
        run_id = str(entry.get("run_id", f"entry_{index}"))
        campaign = f"planet_flow_noop_smoke_{run_id[:12]}"
        overrides = smoke_train_overrides(
            entry,
            output_root=output_root,
            campaign=campaign,
        )
        run_ow_train(
            overrides,
            repo_root=repo_root,
            dry_run=dry_run,
            label=f"planet-flow noop smoke {run_id}",
        )
        if dry_run:
            evaluation_dict: dict[str, object] = {
                "gate_id": "beat_noop",
                "verdict": PreflightVerdict.INCONCLUSIVE.value,
                "reasons": ["dry_run"],
            }
            run_dir_str = None
        else:
            run_dir = latest_run_dir(campaign=campaign, output_root=output_root)
            log_files = sorted((run_dir / "logs").glob("*_jax.jsonl"))
            records = read_jsonl_records(log_files[0]) if log_files else []
            evaluation = evaluate_gate_records(
                beat_noop,
                records,
                campaign=campaign,
                run_dir=run_dir,
                checkpoint=None,
            )
            evaluation_dict = gate_evaluation_to_dict(evaluation)
            run_dir_str = str(run_dir)
        passed = evaluation_dict.get("verdict") == PreflightVerdict.VERIFIED.value
        result = {
            "run_id": run_id,
            "campaign": campaign,
            "train_overrides": list(entry.get("train_overrides") or []),
            "passed": passed,
            "gate": evaluation_dict,
            "run_dir": run_dir_str,
        }
        results.append(result)
        if passed and recommended is None:
            recommended = {
                "run_id": run_id,
                "train_overrides": list(entry.get("train_overrides") or []),
                "campaign": campaign,
            }
    report: dict[str, object] = {
        "shortlist_path": str(shortlist_path),
        "top_k": top_k,
        "results": results,
        "recommended_for_learn_proof": recommended,
        "any_passed": recommended is not None,
    }
    if recommended is not None:
        report["learn_proof_train_overrides"] = list(recommended["train_overrides"])
    return report


def write_smoke_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
