"""Load and run composable preflight gate recipes from ``conf/benchmark/gates/``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.jax.admission_throughput import ThroughputWindow, run_throughput_gate
from src.jax.preflight_gate_loader import (
    GATES_DIR,
    REPO_ROOT,
    build_gate_spec,
    gate_yaml_paths,
    load_gate_yaml,
)

__all__ = [
    "GATES_DIR",
    "REPO_ROOT",
    "DEFAULT_ADMISSION_THROUGHPUT_BASELINE",
    "build_gate_spec",
    "gate_yaml_paths",
    "load_gate_recipe",
    "load_gate_yaml",
    "list_gate_recipes",
    "resolve_learning_gate_id",
    "resolve_throughput_options",
    "run_gate_cli",
]

DEFAULT_ADMISSION_THROUGHPUT_BASELINE = (
    REPO_ROOT / "docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json"
)


def load_gate_recipe(gate_id: str) -> dict[str, object]:
    return load_gate_yaml(gate_id)


def resolve_learning_gate_id(recipe: dict[str, object], gate_id: str) -> str:
    learning_gate = recipe.get("learning_gate")
    if isinstance(learning_gate, str) and learning_gate.strip():
        return learning_gate.strip()
    return gate_id


def _throughput_section(recipe: dict[str, object]) -> dict[str, Any]:
    section = recipe.get("throughput")
    return section if isinstance(section, dict) else {}


def resolve_throughput_options(
    gate_id: str,
    *,
    include_throughput: bool,
    throughput_baseline: Path | None,
    throughput_within_pct: float | None,
) -> tuple[bool, Path | None, float | None, ThroughputWindow]:
    recipe = load_gate_recipe(gate_id)
    section = _throughput_section(recipe)
    enabled = include_throughput or bool(section)
    baseline = throughput_baseline
    if baseline is None and section.get("baseline"):
        baseline = REPO_ROOT / str(section["baseline"])
    within_pct = throughput_within_pct
    if within_pct is None and section.get("within_pct") is not None:
        within_pct = float(section["within_pct"])
    warmup = int(section.get("warmup", 2))
    if section.get("measured_update_count") is not None:
        measured_count = int(section["measured_update_count"])
        window = ThroughputWindow.from_training_benchmark(
            warmup=warmup,
            measured_update_count=measured_count,
        )
    else:
        window = ThroughputWindow(
            warmup=warmup,
            max_measured_update=int(section.get("max_measured_update", warmup + 20)),
        )
    return enabled, baseline, within_pct, window


def list_gate_recipes() -> list[dict[str, object]]:
    recipes: list[dict[str, object]] = []
    for path in gate_yaml_paths():
        gate_id = path.stem
        payload = load_gate_yaml(gate_id)
        recipe_gate_id = payload.get("gate_id") or payload.get("gate") or gate_id
        primitive = payload.get("primitive")
        if gate_id == "win_proof_tournament":
            recipe_gate_id = "win_proof"
            primitive = "uv run ow benchmark tournament-proof"
        recipes.append(
            {
                "gate_id": recipe_gate_id,
                "title": payload.get("title") or payload.get("description"),
                "default_model": payload.get("default_model"),
                "path": payload.get("path"),
                "workflow": payload.get("workflow"),
                "primitive": primitive,
                "learning_gate": payload.get("learning_gate"),
            }
        )
    return recipes


def _append_throughput_section(
    report: dict[str, object],
    *,
    log_path: str | None,
    baseline_path: Path | None,
    within_pct: float | None,
    window: ThroughputWindow,
) -> int:
    if log_path is None:
        report["throughput"] = {
            "verdict": "INCONCLUSIVE",
            "reasons": ["no log_path from learning stage"],
        }
        report["throughput_verdict"] = "INCONCLUSIVE"
        return 1

    try:
        throughput_payload, exit_code = run_throughput_gate(
            Path(log_path),
            baseline_path=baseline_path,
            within_pct=within_pct,
            window=window,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report["throughput"] = {
            "verdict": "INCONCLUSIVE",
            "reasons": [str(exc)],
        }
        report["throughput_verdict"] = "INCONCLUSIVE"
        return 1

    report["throughput"] = throughput_payload
    report["throughput_verdict"] = throughput_payload.get("verdict", "INCONCLUSIVE")
    if baseline_path is not None:
        report["throughput_baseline"] = str(baseline_path)
    return exit_code


def _finalize_admission_report(report: dict[str, object]) -> int:
    learning_passed = str(report.get("verdict", "INCONCLUSIVE")) == "VERIFIED"
    throughput_passed = (
        str(report.get("throughput_verdict", "INCONCLUSIVE")) == "VERIFIED"
    )
    report["admission_passed"] = learning_passed and throughput_passed
    return 0 if report["admission_passed"] else 1


def run_gate_cli(
    gate_id: str,
    *,
    model: str | None = None,
    output_root: Path,
    repo_root: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    thresholds_path: Path | None = None,
    profiles_path: Path | None = None,
    train_overrides: tuple[str, ...] = (),
    out: Path | None = None,
    include_throughput: bool = False,
    throughput_baseline: Path | None = None,
    throughput_within_pct: float | None = None,
) -> int:
    from src.jax.preflight import (
        PreflightVerdict,
        gate_evaluation_to_dict,
        run_preflight_gate,
        write_report,
    )

    recipe = load_gate_recipe(gate_id)
    learning_gate_id = resolve_learning_gate_id(recipe, gate_id)
    recipe_train_overrides: tuple[str, ...] = ()
    raw_recipe_overrides = recipe.get("train_overrides")
    if isinstance(raw_recipe_overrides, list):
        recipe_train_overrides = tuple(str(item) for item in raw_recipe_overrides)
    resolved_model = model or str(
        recipe.get("default_model") or "transformer_factorized_small"
    )
    throughput_enabled, baseline_path, within_pct, window = resolve_throughput_options(
        gate_id,
        include_throughput=include_throughput,
        throughput_baseline=throughput_baseline,
        throughput_within_pct=throughput_within_pct,
    )
    evaluation = run_preflight_gate(
        learning_gate_id,
        model=resolved_model,
        output_root=output_root,
        repo_root=repo_root or REPO_ROOT,
        campaign_gate_id=gate_id if learning_gate_id != gate_id else None,
        dry_run=dry_run,
        verbose=verbose,
        thresholds_path=thresholds_path,
        profiles_path=profiles_path,
        extra_train_overrides=(*recipe_train_overrides, *train_overrides),
    )
    learning_exit = 0 if evaluation.verdict == PreflightVerdict.VERIFIED else 1
    report: dict[str, object] = {
        "gate": gate_id,
        "recipe": recipe.get("path"),
        "model": resolved_model,
        "verdict": evaluation.verdict.value,
        "stage": gate_evaluation_to_dict(evaluation),
    }
    if learning_gate_id != gate_id:
        report["learning_gate"] = learning_gate_id

    if throughput_enabled and not dry_run:
        stage = report.get("stage")
        log_path = stage.get("log_path") if isinstance(stage, dict) else None
        _append_throughput_section(
            report,
            log_path=str(log_path) if log_path else None,
            baseline_path=baseline_path,
            within_pct=within_pct,
            window=window,
        )
        if baseline_path is None:
            print(
                "throughput extract skipped baseline compare: pass --throughput-baseline "
                "or set throughput.baseline in gate YAML",
                file=sys.stderr,
            )
    elif throughput_enabled and dry_run:
        report["throughput"] = {
            "verdict": "INCONCLUSIVE",
            "reasons": ["dry_run"],
        }
        report["throughput_verdict"] = "INCONCLUSIVE"

    if gate_id == "admission" and throughput_enabled:
        exit_code = _finalize_admission_report(report)
    else:
        exit_code = learning_exit

    if out is not None:
        write_report(out, report)
    print(json.dumps(report, indent=2))
    return exit_code
