#!/usr/bin/env python3
"""Build the frozen-defaults config picker HTML from resolved Hydra config."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.config.runtime import compose_hydra_train_config
from src.config.schema import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "docs/tools/config-frozen-defaults-picker.template.html"
OUTPUT_PATH = REPO_ROOT / "docs/tools/config-frozen-defaults-picker.html"

GROUP_LABELS: dict[str, str] = {
    "artifacts": "Artifacts",
    "curriculum": "Curriculum",
    "model": "Model",
    "opponents": "Opponents",
    "output": "Output",
    "reward": "Reward",
    "run": "Run",
    "task": "Task",
    "telemetry": "Telemetry",
    "training": "Training",
}

GROUP_ORDER = [
    "run",
    "model",
    "task",
    "reward",
    "training",
    "curriculum",
    "opponents",
    "telemetry",
    "artifacts",
    "output",
]

RUN_TOP_LEVEL_KEYS = {
    "seed",
    "run_name",
    "heldout_eval_seed_set",
    "print_resolved_config",
    "resume_checkpoint",
    "from_promoted",
}

CHOICES: dict[str, list[str]] = {
    "task.trajectory_shield_mode": ["off", "cheap", "tiered", "exact"],
    "task.trajectory_shield_hit_mode": ["selected_target", "non_friendly"],
    "task.edge_rank_mode": ["snapshot", "intercept_min"],
    "task.ship_action_mode": ["buckets", "continuous_fraction"],
    "reward.terminal_reward_mode": ["binary_win"],
    "model.value_head": ["shared", "format_routed", "distributional"],
    "model.pointer_decoder": ["factorized_topk", "planet_flow_target_heatmap"],
    "artifacts.artifact_pipeline.replay_backend": ["docker", "local"],
    "artifacts.artifact_pipeline.docker_player_count": ["2", "4", "both"],
    "artifacts.promotion.strategy": ["metric", "tournament"],
    "artifacts.promotion.metric_mode": ["max", "min"],
    "artifacts.checkpoint_retention.best_metric_mode": ["max", "min"],
    "opponents.snapshot.selection": ["uniform", "recent_biased"],
    "opponents.snapshot.fallback": ["latest"],
    "output.retention_class": ["compact", "full"],
}

WARN_RULES: list[dict[str, Any]] = [
    {
        "target": "opponents.snapshot.pool_size",
        "requires": "opponents.self_play.enabled",
        "predicate": "gt_zero",
        "message": "pool_size must be > 0 when self-play is enabled",
    },
    {
        "target": "opponents.snapshot.interval_updates",
        "requires": "opponents.self_play.enabled",
        "predicate": "gt_zero",
        "message": "interval_updates must be > 0 when self-play is enabled",
    },
    {
        "target": "opponents.snapshot.pool_size",
        "requires": "opponents.self_play.enabled",
        "predicate": "zero_when_off",
        "invert_requires": True,
        "message": "pool_size must be 0 when self-play is disabled",
    },
    {
        "target": "opponents.snapshot.interval_updates",
        "requires": "opponents.self_play.enabled",
        "predicate": "zero_when_off",
        "invert_requires": True,
        "message": "interval_updates must be 0 when self-play is disabled",
    },
    {
        "target": "curriculum.stages",
        "requires": "curriculum.enabled",
        "predicate": "non_empty_list",
        "message": "curriculum.stages must be non-empty when curriculum is enabled",
    },
]


def _group_id_for_path(path: str) -> str:
    top = path.split(".", 1)[0]
    if top in RUN_TOP_LEVEL_KEYS:
        return "run"
    return top


def _infer_value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


def _flatten_leaves(value: object, *, prefix: str = "") -> list[tuple[str, object]]:
    leaves: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child = value[key]
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_flatten_leaves(child, prefix=path))
        return leaves
    if isinstance(value, list):
        leaves.append((prefix, value))
        return leaves
    leaves.append((prefix, value))
    return leaves


def _field_docstrings() -> dict[str, str]:
    docs: dict[str, str] = {}

    def walk(cls: type, prefix: str = "") -> None:
        for field in dataclasses.fields(cls):
            path = f"{prefix}.{field.name}" if prefix else field.name
            if field_type := field.type:
                if dataclasses.is_dataclass(field_type):
                    walk(field_type, path)
                    continue
            field_doc = str(field.metadata.get("doc", "") or "").strip()
            if field_doc:
                docs[path] = field_doc

    walk(TrainConfig)
    return docs


def _describe(path: str, value: object, docs: dict[str, str]) -> str:
    if path in docs:
        return docs[path]
    if path in CHOICES:
        return "One of: " + ", ".join(CHOICES[path]) + "."
    type_name = _infer_value_type(value)
    if type_name in {"list", "dict"}:
        return f"Structured {type_name} (edit as JSON)."
    return f"Resolved default ({type_name})."


def build_groups() -> list[dict[str, Any]]:
    cfg = compose_hydra_train_config()
    payload = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
    assert isinstance(payload, dict)

    docs = _field_docstrings()
    grouped: dict[str, list[dict[str, Any]]] = {
        group_id: [] for group_id in GROUP_ORDER
    }

    for path, value in _flatten_leaves(payload):
        group_id = _group_id_for_path(path)
        if group_id not in grouped:
            grouped[group_id] = []
        entry: dict[str, Any] = {
            "key": path,
            "desc": _describe(path, value, docs),
            "type": _infer_value_type(value),
            "value": value,
        }
        if path in CHOICES:
            entry["choices"] = CHOICES[path]
        grouped[group_id].append(entry)

    groups: list[dict[str, Any]] = []
    for group_id in GROUP_ORDER:
        fields = grouped.get(group_id, [])
        if not fields:
            continue
        groups.append(
            {
                "id": group_id,
                "label": GROUP_LABELS.get(group_id, group_id.title()),
                "fields": fields,
            }
        )
    return groups


def render_html(groups: list[dict[str, Any]]) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data_blob = json.dumps(
        {"groups": groups, "warnRules": WARN_RULES},
        indent=2,
        sort_keys=True,
    )
    if "__CONFIG_DATA__" not in template:
        raise ValueError("template is missing __CONFIG_DATA__ placeholder")
    return template.replace("__CONFIG_DATA__", data_blob)


def main() -> None:
    groups = build_groups()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_html(groups), encoding="utf-8")
    field_count = sum(len(group["fields"]) for group in groups)
    print(f"wrote {OUTPUT_PATH} ({field_count} fields, {len(groups)} groups)")


if __name__ == "__main__":
    main()
