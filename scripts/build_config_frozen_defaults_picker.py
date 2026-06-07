#!/usr/bin/env python3
"""Build the frozen-defaults config picker HTML from resolved Hydra config."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from src.config.runtime import compose_hydra_train_config
from src.config.schema import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"
TEMPLATE_PATH = REPO_ROOT / "docs/tools/config-frozen-defaults-picker.template.html"
OUTPUT_PATH = REPO_ROOT / "docs/tools/config-frozen-defaults-picker.html"

HYDRA_CONFIG_GROUPS = (
    "model",
    "task",
    "reward",
    "training",
    "curriculum",
    "opponents",
    "telemetry",
    "artifacts",
)

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

# Leaves omitted from the operator picker (still valid via Hydra overrides / tests).
EXCLUDE_PATHS: frozenset[str] = frozenset(
    {
        "task.env_parity_mode",  # opt-in task=kaggle_parity only; not a frozen train default
        "output.run_id",  # ${orbit_run_id:${seed}} — changes every compose/build
    }
)

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

NAMED_PRESETS: list[dict[str, Any]] = [
    {
        "id": "repo_default",
        "label": "Repo default (conf/config.yaml)",
        "overrides": [],
    },
    {
        "id": "beat_noop_core",
        "label": "Beat noop learning proof (core recipe)",
        "overrides": [
            "model=transformer_factorized_small",
            "task=shield_cheap",
            "training=2p_16",
            "training.rollout_steps=128",
            "opponents=noop_only",
            "curriculum=off",
            "telemetry.wandb.enabled=false",
            "artifacts.artifact_pipeline.enabled=false",
            "telemetry.metric_groups.action_decision=true",
            "seed=42",
            "training.log_every=1",
        ],
    },
    {
        "id": "admission_locked",
        "label": "Admission locked (operator 2026-06-05)",
        "overrides": [
            "model=transformer_factorized_small",
            "task=shield_cheap",
            "training=2p4p_32_split",
            "training.rollout_steps=256",
            "task.candidate_count=3",
            "opponents=noop_only",
            "curriculum=off",
            "telemetry.wandb.enabled=true",
            "telemetry.wandb.group=preflight",
            "artifacts.artifact_pipeline.enabled=false",
            "artifacts.replay.enabled=false",
            "telemetry.metric_groups.action_decision=true",
            "seed=42",
            "training.log_every=1",
        ],
    },
    {
        "id": "ssot_pipeline",
        "label": "SSOT training pipeline",
        "overrides": ["artifacts=ssot_pipeline"],
    },
    {
        "id": "hybrid_promotion",
        "label": "Hybrid promotion funnel",
        "overrides": ["artifacts=hybrid_promotion"],
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


def _flatten_leaf_map(value: object, *, prefix: str = "") -> dict[str, object]:
    return dict(_flatten_leaves(value, prefix=prefix))


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


def _composition_key(overrides: list[str]) -> str:
    """Canonical lookup key for a Hydra override bundle."""

    group_overrides = [
        item
        for item in overrides
        if "=" in item and "." not in item.split("=", maxsplit=1)[0]
    ]
    leaf_overrides = [item for item in overrides if item not in group_overrides]
    parts = sorted(group_overrides) + sorted(leaf_overrides)
    return "|".join(parts)


def _parse_config_yaml_defaults(*, conf_root: Path = CONF_ROOT) -> dict[str, str]:
    raw = yaml.safe_load((conf_root / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("conf/config.yaml must be a mapping.")
    selections: dict[str, str] = {}
    for entry in raw.get("defaults", []):
        if isinstance(entry, dict):
            for group, selection in entry.items():
                if group != "_self_":
                    selections[str(group)] = str(selection)
    return selections


def _list_group_options(group: str, *, conf_root: Path = CONF_ROOT) -> list[str]:
    group_dir = conf_root / group
    if not group_dir.is_dir():
        return []
    options = sorted(
        path.stem
        for path in group_dir.glob("*.yaml")
        if path.name not in {"base.yaml"}
    )
    return options


def _resolve_yaml_chain(
    group: str,
    selection: str,
    *,
    conf_root: Path = CONF_ROOT,
    visited: set[str] | None = None,
) -> list[Path]:
    """Return yaml files in Hydra merge order (earlier = lower priority)."""

    visited = visited or set()
    visit_key = f"{group}:{selection}"
    if visit_key in visited:
        return []
    visited.add(visit_key)

    path = conf_root / group / f"{selection}.yaml"
    if not path.is_file():
        return []

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return [path]

    chain: list[Path] = []
    defaults = raw.get("defaults", [])
    if isinstance(defaults, list):
        for entry in defaults:
            if entry == "_self_":
                continue
            if isinstance(entry, str):
                if entry == "base":
                    base_path = conf_root / group / "base.yaml"
                    if base_path.is_file():
                        chain.append(base_path)
                else:
                    chain.extend(
                        _resolve_yaml_chain(
                            group,
                            entry,
                            conf_root=conf_root,
                            visited=visited,
                        )
                    )
            elif isinstance(entry, dict):
                for nested_group, nested_selection in entry.items():
                    if nested_group == "_self_":
                        continue
                    chain.extend(
                        _resolve_yaml_chain(
                            str(nested_group),
                            str(nested_selection),
                            conf_root=conf_root,
                            visited=visited,
                        )
                    )
    chain.append(path)
    return chain


def _flatten_yaml_values(
    data: object, *, prefix: str = ""
) -> dict[str, object]:
    values: dict[str, object] = {}
    if not isinstance(data, dict):
        return values
    for key, value in data.items():
        if key == "defaults":
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            values[path] = value
            values.update(_flatten_yaml_values(value, prefix=path))
        else:
            values[path] = value
    return values


def _group_relative_path(path: str, group: str) -> str | None:
    prefix = f"{group}."
    if path == group:
        return ""
    if path.startswith(prefix):
        return path[len(prefix) :]
    return None


def _schema_leaf_map() -> dict[str, object]:
    schema = OmegaConf.structured(TrainConfig)
    payload = OmegaConf.to_container(schema, resolve=True)
    assert isinstance(payload, dict)
    return _flatten_leaf_map(payload)


def _root_config_leaf_map(*, conf_root: Path = CONF_ROOT) -> dict[str, object]:
    raw = yaml.safe_load((conf_root / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    values: dict[str, object] = {}
    for key, value in raw.items():
        if key in {"defaults", "hydra"}:
            continue
        if isinstance(value, dict):
            values[key] = value
            values.update(_flatten_yaml_values(value, prefix=str(key)))
        else:
            values[str(key)] = value
    return values


def _active_group_selections(overrides: list[str]) -> dict[str, str]:
    selections = _parse_config_yaml_defaults()
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", maxsplit=1)
        if "." not in key:
            selections[key] = value
    return selections


def _resolve_field_sources(
    values: dict[str, object],
    overrides: list[str],
    *,
    conf_root: Path = CONF_ROOT,
) -> dict[str, str]:
    """Best-effort provenance for each resolved leaf."""

    schema_values = _schema_leaf_map()
    root_values = _root_config_leaf_map(conf_root=conf_root)
    selections = _active_group_selections(overrides)

    group_yaml_values: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for group in HYDRA_CONFIG_GROUPS:
        selection = selections.get(group, "default")
        chain = _resolve_yaml_chain(group, selection, conf_root=conf_root)
        entries: list[tuple[str, dict[str, object]]] = []
        for yaml_path in chain:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            rel = f"conf/{group}/{yaml_path.name}"
            entries.append((rel, _flatten_yaml_values(raw)))
        group_yaml_values[group] = entries

    leaf_override_sources: dict[str, str] = {}
    for override in overrides:
        if "=" not in override or "." not in override.split("=", maxsplit=1)[0]:
            continue
        leaf_override_sources[override.split("=", maxsplit=1)[0]] = (
            f"override: {override}"
        )

    sources: dict[str, str] = {}
    for path, value in values.items():
        if path in EXCLUDE_PATHS:
            continue
        if path in leaf_override_sources:
            sources[path] = leaf_override_sources[path]
            continue

        top = path.split(".", 1)[0]
        if top in RUN_TOP_LEVEL_KEYS or top == "output":
            rel = _group_relative_path(path, top) if top == "output" else path
            declared = False
            for root_path, root_value in root_values.items():
                if root_path == path or (
                    rel is not None and root_path == rel and top == "output"
                ):
                    sources[path] = "conf/config.yaml"
                    declared = True
                    break
            if declared:
                continue
            if path in schema_values and values_equal(value, schema_values[path]):
                sources[path] = "schema.py default"
            else:
                sources[path] = "composed"
            continue

        if top in HYDRA_CONFIG_GROUPS:
            rel_path = _group_relative_path(path, top)
            found = False
            for rel_file, flat in reversed(group_yaml_values.get(top, [])):
                if rel_path is not None and rel_path in flat:
                    sources[path] = rel_file
                    found = True
                    break
            if found:
                continue
            if path in schema_values and values_equal(value, schema_values[path]):
                sources[path] = "schema.py default"
            else:
                sources[path] = "composed"
            continue

        if path in schema_values and values_equal(value, schema_values[path]):
            sources[path] = "schema.py default"
        else:
            sources[path] = "composed"

    return sources


def values_equal(a: object, b: object) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _compose_leaf_values(overrides: list[str]) -> dict[str, object] | None:
    try:
        cfg = compose_hydra_train_config(overrides)
    except Exception:
        return None
    payload = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
    assert isinstance(payload, dict)
    values = _flatten_leaf_map(payload)
    return {path: value for path, value in values.items() if path not in EXCLUDE_PATHS}


def build_field_catalog() -> list[dict[str, Any]]:
    """Field metadata from the repo-default composition."""

    values = _compose_leaf_values([])
    docs = _field_docstrings()
    grouped: dict[str, list[dict[str, Any]]] = {group_id: [] for group_id in GROUP_ORDER}

    for path, value in sorted(values.items()):
        group_id = _group_id_for_path(path)
        if group_id not in grouped:
            grouped[group_id] = []
        entry: dict[str, Any] = {
            "key": path,
            "desc": _describe(path, value, docs),
            "type": _infer_value_type(value),
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


def build_config_group_catalog() -> dict[str, Any]:
    defaults = _parse_config_yaml_defaults()
    catalog: dict[str, Any] = {}
    for group in HYDRA_CONFIG_GROUPS:
        options = []
        for option_id in _list_group_options(group):
            chain = _resolve_yaml_chain(group, option_id)
            options.append(
                {
                    "id": option_id,
                    "files": [f"conf/{group}/{path.name}" for path in chain],
                }
            )
        catalog[group] = {
            "defaultSelection": defaults.get(group, "default"),
            "options": options,
        }
    return catalog


def build_composition_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    skipped: list[str] = []

    def add_composition(
        *,
        key: str,
        label: str,
        overrides: list[str],
    ) -> bool:
        if key in index:
            return True
        values = _compose_leaf_values(overrides)
        if values is None:
            skipped.append(label)
            return False
        index[key] = {
            "label": label,
            "overrides": overrides,
            "values": values,
            "sources": _resolve_field_sources(values, overrides),
        }
        return True

    add_composition(key="", label="Repo default", overrides=[])

    for group in HYDRA_CONFIG_GROUPS:
        default_selection = _parse_config_yaml_defaults().get(group, "default")
        for option_id in _list_group_options(group):
            if option_id == default_selection:
                continue
            overrides = [f"{group}={option_id}"]
            key = _composition_key(overrides)
            add_composition(
                key=key,
                label=f"{group}={option_id}",
                overrides=overrides,
            )

    for preset in NAMED_PRESETS:
        overrides = list(preset["overrides"])
        preset_id = str(preset["id"])
        if preset_id == "repo_default":
            continue
        preset_key = _composition_key(overrides)
        if preset_key in index:
            entry = index[preset_key]
            index[preset_id] = {
                "label": preset["label"],
                "overrides": overrides,
                "values": entry["values"],
                "sources": entry["sources"],
            }
            continue
        if add_composition(key=preset_id, label=preset["label"], overrides=overrides):
            if preset_key and preset_key != preset_id:
                entry = index[preset_id]
                index[preset_key] = entry

    if skipped:
        print(f"skipped {len(skipped)} invalid compositions (e.g. {skipped[0]})")

    return index


def build_groups() -> list[dict[str, Any]]:
    """Backward-compatible helper: repo-default groups with values embedded."""

    catalog = build_field_catalog()
    values = build_composition_index()[""]["values"]
    groups: list[dict[str, Any]] = []
    for group in catalog:
        fields = []
        for field in group["fields"]:
            enriched = dict(field)
            enriched["value"] = values[field["key"]]
            fields.append(enriched)
        groups.append({**group, "fields": fields})
    return groups


def build_picker_payload() -> dict[str, Any]:
    return {
        "groups": build_field_catalog(),
        "configGroups": build_config_group_catalog(),
        "compositionIndex": build_composition_index(),
        "presets": [
            {"id": preset["id"], "label": preset["label"]}
            for preset in NAMED_PRESETS
        ],
        "warnRules": WARN_RULES,
    }


def render_html(payload: dict[str, Any]) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data_blob = json.dumps(payload, indent=2, sort_keys=True)
    if "__CONFIG_DATA__" not in template:
        raise ValueError("template is missing __CONFIG_DATA__ placeholder")
    return template.replace("__CONFIG_DATA__", data_blob)


def main() -> None:
    payload = build_picker_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_html(payload), encoding="utf-8")
    field_count = sum(len(group["fields"]) for group in payload["groups"])
    composition_count = len(payload["compositionIndex"])
    print(
        f"wrote {OUTPUT_PATH} ({field_count} fields, "
        f"{len(payload['groups'])} groups, {composition_count} compositions)"
    )


if __name__ == "__main__":
    main()
