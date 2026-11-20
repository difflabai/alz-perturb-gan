#!/usr/bin/env python3
"""Check that relative links in tracked Markdown files point at existing paths.

Usage: python -B scripts/ci/check_md_links.py [FILE.md ...]

With no arguments it checks every Markdown file tracked by git. External links
(http, https, mailto and similar), pure #anchors and links inside code spans or
fenced code blocks are ignored. Anchors are not validated; only the target
path must exist. A target may name a gitignored local artifact (runs/, data/,
artifacts/); such links are reported separately and do not fail the check,
because the repository documents those machine-local paths on purpose.
Exit status is 1 when any other relative link is broken.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
# Gitignored machine-local trees (.gitignore). Links into them document local
# artifacts and cannot resolve in a clean clone or CI.
LOCAL_ONLY = ("runs/", "data/", "artifacts/")

# The destination is either <...> (which may contain spaces) or a bare run without spaces.
INLINE = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*(?:<([^>\n]+)>|([^)\s]+))(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(`+)(?:.(?!\1))*?.?\1", re.S)
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def tracked_markdown(root: Path = ROOT) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "*.md"], cwd=root, check=True,
                         capture_output=True).stdout.decode()
    return [root / name for name in out.split("\0") if name]


def links(text: str):
    """Yield (line number, target) for Markdown links outside code."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0] * len(marker)
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        stripped = CODE_SPAN.sub("", line)
        for found in INLINE.finditer(stripped):
            yield number, found.group(1) or found.group(2)
        ref = REFERENCE.match(stripped)
        if ref:
            yield number, ref.group(1)


def resolve(source: Path, target: str, root: Path = ROOT) -> Path | None:
    """Return the filesystem path a relative link names, or None to ignore it."""
    if not target or target.startswith("#") or SCHEME.match(target) or target.startswith("//"):
        return None
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path:
        return None
    base = root if path.startswith("/") else source.parent
    return (base / path.lstrip("/")).resolve()


def check(files: list[Path], root: Path = ROOT):
    broken, local = [], []
    for source in files:
        for number, target in links(source.read_text(encoding="utf-8", errors="replace")):
            path = resolve(source, target, root)
            if path is None or path.exists():
                continue
            where = f"{source.relative_to(root)}:{number}: {target}"
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = ""
            (local if relative.startswith(LOCAL_ONLY) else broken).append(where)
    return broken, local


def main(argv: list[str]) -> int:
    files = [Path(arg).resolve() for arg in argv] if argv else tracked_markdown()
    broken, local = check(files)
    for where in local:
        print(f"local-only (ignored): {where}")
    for where in broken:
        print(f"BROKEN: {where}")
    print(f"checked {len(files)} Markdown files: {len(broken)} broken relative links, "
          f"{len(local)} links into gitignored local trees")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
