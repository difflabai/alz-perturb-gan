#!/usr/bin/env python3
"""Compare the running interpreter with an environment recipe in envs/.

Usage:
    python -B scripts/check_env.py envs/microglia-research-py311.txt
    python -B scripts/check_env.py --allow-local-variant envs/microglia-research-py311.txt

The recipe is a pip requirements file. ``name==version`` lines must match the
installed distribution exactly (``--allow-local-variant`` ignores a ``+local``
label such as ``+cu130`` versus ``+cpu``). Range lines are checked when the
``packaging`` module is importable. ``-r other.txt`` includes are followed.
Pip options and ``name @ url`` lines without a version are reported but not
compared.

Lines starting with ``# check_env:`` carry checks pip cannot express:

    # check_env: python==3.11
    # check_env: native particlegan==0.10.0 target=<dir> commit=<sha>

A ``native`` check reads the distribution metadata inside ``target`` (relative
to the repository root) instead of the interpreter's site-packages, and reports
which commit the target was installed from when that is recorded. The model
loaders still verify the pinned native source hashes themselves; this script
is an early, readable check, not a replacement for those gates.

Exit status: 0 when every compared requirement matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$")
DIRECTIVE = re.compile(r"^#\s*check_env:\s*(.+)$")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def strip_local(version: str) -> str:
    return version.split("+", 1)[0]


def parse_recipe(path: Path, seen: set[Path] | None = None):
    """Yield ('req', name, spec, line) and ('directive', text, line) entries."""
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        directive = DIRECTIVE.match(line)
        if directive:
            yield ("directive", directive.group(1).strip(), f"{path.name}: {line}")
            continue
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            yield from parse_recipe(path.parent / line.split(None, 1)[1], seen)
            continue
        if line.startswith("-"):
            yield ("option", line, f"{path.name}: {line}")
            continue
        match = REQUIREMENT.match(line.split(";", 1)[0].strip())
        if match:
            yield ("req", match.group(1), match.group(3).strip(), f"{path.name}: {line}")


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def direct_url_commit(dist) -> str | None:
    try:
        info = json.loads(dist.read_text("direct_url.json") or "null") or {}
    except (OSError, ValueError):
        return None
    return (info.get("vcs_info") or {}).get("commit_id")


def version_matches(installed: str, spec: str, allow_local: bool) -> tuple[bool | None, str]:
    if spec.startswith("==") and "," not in spec and "*" not in spec:
        wanted = spec[2:].strip()
        if allow_local:
            return strip_local(installed) == strip_local(wanted), wanted
        return installed == wanted, wanted
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        return None, spec + " (packaging not installed; not compared)"
    return SpecifierSet(spec).contains(installed, prereleases=True), spec


def check_requirement(name: str, spec: str, allow_local: bool) -> tuple[bool | None, str]:
    installed = installed_version(name)
    if installed is None:
        return False, f"{name}: not installed (wanted {spec or 'any'})"
    if not spec:
        return True, f"{name}: {installed}"
    if spec.startswith("@"):
        wanted_commit = spec.rsplit("@", 1)[1] if spec.count("@") > 1 else ""
        commit = direct_url_commit(metadata.distribution(name))
        if re.fullmatch(r"[0-9a-f]{7,40}", wanted_commit):
            ok = commit is not None and (commit.startswith(wanted_commit) or wanted_commit.startswith(commit))
            return ok, f"{name}: {installed} from commit {commit or 'unrecorded'} (wanted {wanted_commit[:12]})"
        return None, f"{name}: {installed} (wanted {spec}; not compared)"
    ok, wanted = version_matches(installed, spec, allow_local)
    if ok is None:
        return None, f"{name}: {installed} (wanted {wanted}; not compared)"
    return ok, f"{name}: {installed} (wanted {wanted})"


def check_native(text: str, root: Path) -> tuple[bool, str]:
    fields = text.split()
    name, _, wanted = fields[1].partition("==")
    options = dict(field.split("=", 1) for field in fields[2:] if "=" in field)
    target = (root / options.get("target", "")).resolve()
    dists = [d for d in metadata.distributions(path=[str(target)])
             if canonical(d.metadata["Name"] or "") == canonical(name)]
    if not dists:
        return False, f"{name} (native): no distribution in {target}"
    dist = dists[0]
    version = dist.version
    commit = direct_url_commit(dist)
    origin = "direct_url.json"
    source = target.parent / "source"
    if commit is None and (source / ".git").exists():
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        commit = result.stdout.strip() or None
        origin = f"{source} HEAD"
    ok = version == wanted and (target / name.replace("-", "_")).is_dir()
    note = f"{name} (native): {version} from {target} (wanted {wanted})"
    wanted_commit = options.get("commit")
    if wanted_commit:
        if commit is None:
            note += "; install commit not recorded in the target"
        else:
            same = commit.startswith(wanted_commit) or wanted_commit.startswith(commit)
            ok = ok and same
            note += f"; commit {commit[:12]} via {origin} (wanted {wanted_commit[:12]})"
    return ok, note


def check_python(text: str) -> tuple[bool, str]:
    wanted = text.split("==", 1)[1].strip()
    actual = platform.python_version()
    ok = actual == wanted or actual.startswith(wanted + ".")
    return ok, f"python: {actual} (wanted {wanted})"


def run(recipe: Path, root: Path = ROOT, allow_local: bool = False):
    rows = []
    for entry in parse_recipe(recipe):
        kind = entry[0]
        if kind == "req":
            _, name, spec, _ = entry
            rows.append(check_requirement(name, spec, allow_local))
        elif kind == "directive":
            text = entry[1]
            if text.startswith("python=="):
                rows.append(check_python(text))
            elif text.startswith("native "):
                rows.append(check_native(text, root))
            else:
                rows.append((False, f"unknown check_env directive: {text}"))
        else:
            rows.append((None, f"pip option (not compared): {entry[1]}"))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--allow-local-variant", action="store_true",
                        help="ignore +local version labels (for example +cu130 vs +cpu)")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root for native targets (default: this checkout)")
    args = parser.parse_args(argv)
    rows = run(args.recipe, args.root, args.allow_local_variant)
    labels = {True: "ok  ", False: "FAIL", None: "info"}
    for ok, note in rows:
        print(f"{labels[ok]} {note}")
    failures = sum(ok is False for ok, _ in rows)
    print(f"{args.recipe}: {failures} mismatches in {sum(ok is not None for ok, _ in rows)} checks "
          f"({sys.executable})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
