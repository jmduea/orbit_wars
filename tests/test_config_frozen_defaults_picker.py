"""Guards for the frozen-defaults config picker template and build script."""

from __future__ import annotations

from pathlib import Path

import scripts.build_config_frozen_defaults_picker as picker

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs/tools/config-frozen-defaults-picker.template.html"
OUTPUT = REPO_ROOT / "docs/tools/config-frozen-defaults-picker.html"


def test_bool_toggle_has_clickable_box() -> None:
    """Label toggles need block layout or width/height collapse to zero."""
    css = TEMPLATE.read_text(encoding="utf-8")
    toggle_block = css.split(".toggle {", 1)[1].split("}", 1)[0]
    assert "display: inline-block" in toggle_block


def test_build_groups_include_bool_fields() -> None:
    groups = picker.build_groups()
    fields = [f for g in groups for f in g["fields"]]
    bool_fields = [f for f in fields if f["type"] == "bool"]
    assert bool_fields
    normalize = next(f for f in bool_fields if f["key"] == "model.normalize_observations")
    assert isinstance(normalize["value"], bool)


def test_template_has_collapsible_group_markup() -> None:
    """Each group section should expose collapse affordances and aria state."""
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "group-chevron" in html
    assert "group-body" in html
    assert "aria-expanded" in html
    assert "groupExpanded" in html
    assert "toggleGroup" in html
    assert ".group.collapsed .group-body" in html


def test_picker_payload_includes_compositions_and_provenance() -> None:
    payload = picker.build_picker_payload()
    assert payload["compositionIndex"]
    assert "" in payload["compositionIndex"]
    assert "beat_noop_core" in payload["compositionIndex"]
    assert payload["configGroups"]["model"]["options"]
    default_sources = payload["compositionIndex"][""]["sources"]
    assert default_sources["seed"] == "conf/config.yaml"
    assert "model=" in "".join(payload["compositionIndex"])


def test_single_override_changes_model_hidden_size() -> None:
    payload = picker.build_picker_payload()
    key = picker._composition_key(["model=transformer_factorized_small"])
    entry = payload["compositionIndex"][key]
    default = payload["compositionIndex"][""]["values"]
    assert entry["values"]["model.hidden_size"] != default["model.hidden_size"]


def test_generated_html_embeds_selector_ui() -> None:
    picker.main()
    html = OUTPUT.read_text(encoding="utf-8")
    assert "compositionIndex" in html
    assert "presetSelect" in html
    assert "flag-source" in html
    assert "beat_noop_core" in html
