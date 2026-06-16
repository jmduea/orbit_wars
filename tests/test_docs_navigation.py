"""Regression: docs navigation links resolve to existing paths."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
README = DOCS_ROOT / "README.md"
BENCHMARKS_README = DOCS_ROOT / "benchmarks" / "README.md"
ONBOARDING = DOCS_ROOT / "ONBOARDING.md"

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
BACKTICK_PATH_RE = re.compile(r"`((?:docs/|\.\./)[^`]+)`")
HAND_MAINTAINED_BLOCK_RE = re.compile(
    r"<!-- hand-maintained:documentation -->\s*(.*?)\s*<!-- /hand-maintained -->",
    re.DOTALL,
)
STALE_ONBOARDING_DOCS = (
    "experiments.md",
    "baseline_sweep.md",
    "config_migration.md",
)
STALE_DOC_PREFIXES = (
    "docs/archive/",
    "docs/plans/",
    "docs/ideation/",
)

DOC_MARKDOWN_SOURCES = [
    *sorted(DOCS_ROOT.rglob("*.md")),
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CONCEPTS.md",
    REPO_ROOT / "COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md",
]


def _extract_markdown_links(markdown: str) -> list[str]:
    links: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(markdown):
        href = match.group(1).strip()
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if href.startswith("/"):
            continue
        if re.fullmatch(r"[0-9a-f-]{36}", href):
            continue
        if "#" in href:
            href = href.split("#", 1)[0]
        if href:
            links.append(href)
    return links


def _extract_backtick_doc_paths(markdown: str) -> list[str]:
    paths: list[str] = []
    for match in BACKTICK_PATH_RE.finditer(markdown):
        path = match.group(1).strip()
        if "\n" in path:
            continue
        if not (path.endswith(".md") or path.endswith("/")):
            continue
        if path.startswith(("http://", "https://")):
            continue
        if "#" in path:
            path = path.split("#", 1)[0]
        if path:
            paths.append(path)
    return paths


def _collect_backtick_path_cases() -> list[tuple[Path, str]]:
    cases: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in DOC_MARKDOWN_SOURCES:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        for raw in _extract_backtick_doc_paths(text):
            if raw.startswith("../"):
                base_dir = source.parent
                href = raw
            elif raw.startswith("docs/"):
                base_dir = REPO_ROOT
                href = raw
            else:
                continue
            if href.endswith((".py", ".yaml", ".yml", ".json", ".html", ".sh")):
                continue
            if href in {"docs/Issues.md", "docs/brain_dump.md"}:
                continue
            key = (str(source.relative_to(REPO_ROOT)), href)
            if key in seen:
                continue
            seen.add(key)
            cases.append((base_dir, href))
    return cases


def _hand_maintained_blocks(markdown: str) -> list[str]:
    blocks = [match.group(1) for match in HAND_MAINTAINED_BLOCK_RE.finditer(markdown)]
    return blocks


def _assert_link_resolves(base_dir: Path, href: str) -> None:
    target = (base_dir / href).resolve()
    if href.endswith("/"):
        assert target.is_dir(), f"missing directory for {href!r}: {target}"
        return
    assert target.exists(), f"missing path for {href!r}: {target}"


def _collect_doc_link_cases() -> list[tuple[Path, str]]:
    cases: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in DOC_MARKDOWN_SOURCES:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        base_dir = source.parent
        for href in _extract_markdown_links(text):
            key = (str(source.relative_to(REPO_ROOT)), href)
            if key in seen:
                continue
            seen.add(key)
            cases.append((base_dir, href))
    return cases


@pytest.mark.parametrize(
    "href",
    _extract_markdown_links(BENCHMARKS_README.read_text(encoding="utf-8")),
)
def test_docs_benchmarks_readme_links_resolve(href: str) -> None:
    _assert_link_resolves(DOCS_ROOT / "benchmarks", href)


@pytest.mark.parametrize("base_dir,href", _collect_doc_link_cases())
def test_documentation_markdown_links_resolve(base_dir: Path, href: str) -> None:
    _assert_link_resolves(base_dir, href)


@pytest.mark.parametrize("base_dir,href", _collect_backtick_path_cases())
def test_documentation_backtick_paths_resolve(base_dir: Path, href: str) -> None:
    _assert_link_resolves(base_dir, href)


@pytest.mark.parametrize("source", DOC_MARKDOWN_SOURCES)
def test_documentation_has_no_stale_archive_or_plan_references(source: Path) -> None:
    if not source.exists():
        pytest.skip(f"missing {source}")
    text = source.read_text(encoding="utf-8")
    for prefix in STALE_DOC_PREFIXES:
        assert prefix not in text, (
            f"{source.relative_to(REPO_ROOT)} still references {prefix}"
        )


def test_documentation_script_backtick_paths_exist() -> None:
    script_pat = re.compile(r"`(scripts/[^`]+\.py)`")
    missing: list[str] = []
    for source in DOC_MARKDOWN_SOURCES:
        if not source.exists():
            continue
        for match in script_pat.finditer(source.read_text(encoding="utf-8")):
            script = match.group(1)
            if script == "scripts/*.py":
                continue
            if not (REPO_ROOT / script).is_file():
                missing.append(f"{source.relative_to(REPO_ROOT)}: {script}")
    assert not missing, "missing script paths in docs:\n" + "\n".join(missing)


def test_docs_readme_start_here_excludes_brain_dump() -> None:
    text = README.read_text(encoding="utf-8")
    start_here = text.split("## Agent policy chain", maxsplit=1)[0]
    assert "brain_dump" not in start_here


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


def test_docs_archive_directory_removed() -> None:
    assert not (DOCS_ROOT / "archive").exists()


def test_canonical_brainstorms_present() -> None:
    for name in (
        "2026-06-03-training-pipeline-ssot-requirements.md",
        "2026-06-03-gate5-unified-tournament-requirements.md",
    ):
        assert (DOCS_ROOT / "brainstorms" / name).is_file()
