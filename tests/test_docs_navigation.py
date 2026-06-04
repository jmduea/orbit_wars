"""Regression: docs navigation links resolve to existing paths."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
README = DOCS_ROOT / "README.md"
ONBOARDING = DOCS_ROOT / "ONBOARDING.md"

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HAND_MAINTAINED_BLOCK_RE = re.compile(
    r"<!-- hand-maintained:documentation -->\s*(.*?)\s*<!-- /hand-maintained -->",
    re.DOTALL,
)
STALE_ONBOARDING_DOCS = (
    "experiments.md",
    "baseline_sweep.md",
    "config_migration.md",
)


def _extract_markdown_links(markdown: str) -> list[str]:
    links: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(markdown):
        href = match.group(1).strip()
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if "#" in href:
            href = href.split("#", 1)[0]
        if href:
            links.append(href)
    return links


def _hand_maintained_blocks(markdown: str) -> list[str]:
    blocks = [match.group(1) for match in HAND_MAINTAINED_BLOCK_RE.finditer(markdown)]
    return blocks


def _assert_link_resolves(base_dir: Path, href: str) -> None:
    target = (base_dir / href).resolve()
    if href.endswith("/"):
        assert target.is_dir(), f"missing directory for {href!r}: {target}"
        return
    assert target.exists(), f"missing path for {href!r}: {target}"


@pytest.mark.parametrize("href", _extract_markdown_links(README.read_text(encoding="utf-8")))
def test_docs_readme_links_resolve(href: str) -> None:
    _assert_link_resolves(DOCS_ROOT, href)


def test_docs_readme_start_here_excludes_brain_dump() -> None:
    text = README.read_text(encoding="utf-8")
    start_here = text.split("## Agent policy chain", maxsplit=1)[0]
    assert "brain_dump" not in start_here


def test_onboarding_hand_maintained_doc_links_resolve() -> None:
    onboarding = ONBOARDING.read_text(encoding="utf-8")
    blocks = _hand_maintained_blocks(onboarding)
    assert blocks, "expected hand-maintained documentation blocks in ONBOARDING.md"
    for block in blocks:
        for href in _extract_markdown_links(block):
            _assert_link_resolves(DOCS_ROOT, href)


def test_onboarding_regenerating_mentions_hand_maintained_docs() -> None:
    text = ONBOARDING.read_text(encoding="utf-8")
    regen = text.split("## Regenerating this guide", maxsplit=1)[-1]
    assert "Hand-maintained sections" in regen
    assert "hand-maintained:documentation" in regen
    assert "docs/README.md" in regen or "README.md" in regen


@pytest.mark.parametrize("stale_doc", STALE_ONBOARDING_DOCS)
def test_onboarding_has_no_stale_documentation_links(stale_doc: str) -> None:
    text = ONBOARDING.read_text(encoding="utf-8")
    assert stale_doc not in text


def test_agents_md_points_to_docs_readme() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    intro = agents.split("\n## ", maxsplit=1)[0]
    assert intro.count("docs/README.md") == 1
