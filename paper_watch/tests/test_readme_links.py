"""Data-free checks that the paper-watch README and notes link to files that exist."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s]+)\)")
DOCS = [ROOT / "README.md", *sorted((ROOT / "notes").glob("*.md"))]


def _relative_links(path):
    for target in LINK.findall(path.read_text()):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        yield target.split("#", 1)[0]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    missing = [t for t in _relative_links(doc) if not (doc.parent / t).exists()]
    assert not missing, f"{doc.name} links to missing files: {missing}"


def test_readme_links_blocked_finds_note():
    assert "notes/blocked_finds.md" in set(_relative_links(ROOT / "README.md"))
