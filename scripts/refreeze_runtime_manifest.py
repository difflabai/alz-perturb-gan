#!/usr/bin/env python3
"""Check or re-pin the microglia runtime source manifest.

--check verifies every pinned source against the checkout and changes nothing.
Without --check, the tool recomputes the digests for exactly the pinned key set
(it never drops a key; --add pins an extra path; --relocate OLD=NEW renames a key
only when the file at NEW has exactly the digest pinned for OLD, e.g. when an
out-of-repo source is vendored byte for byte), archives the previous manifest
byte for byte under analysis/default_empf_20261117/previous/, and records the
archive, its digest, the reason and the changed keys in the new manifest.

Gitignored artifacts (the native ParticleGAN releases) are absent from a fresh
worktree. --artifact-root names another checkout to read those missing files
from; a file read that way must keep its pinned digest, so this path can only
confirm artifacts, never re-pin them.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'analysis/default_empf_20261117/runtime_source_manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(name, root, artifact_root=None):
    """Return (path, from_artifact_root) for a manifest key, or (None, False) if absent."""
    if Path(name).is_absolute():
        path = Path(name)
        return (path, False) if path.exists() else (None, False)
    path = root / name
    if path.exists():
        return path, False
    if artifact_root is not None and (artifact_root / name).exists():
        return artifact_root / name, True
    return None, False


def scan(manifest, root, artifact_root=None):
    ok, missing, changed, borrowed = [], [], {}, []
    for name, digest in manifest['sources'].items():
        path, from_artifacts = locate(name, root, artifact_root)
        if path is None:
            missing.append(name)
            continue
        if from_artifacts:
            borrowed.append(name)
        now = sha(path)
        if now == digest:
            ok.append(name)
        else:
            changed[name] = now
    old = manifest['archived_previous_entrypoint']
    entry_ok = (root / old['archive']).exists() and sha(root / old['archive']) == old['sha256']
    return ok, missing, changed, borrowed, entry_ok


def relocate_keys(manifest, root, relocate):
    """Return a copy of manifest with each OLD key renamed to NEW, refusing any digest change."""
    sources = dict(manifest['sources'])
    moved = {}
    for old, new in relocate:
        if old not in sources:
            raise ValueError(f'cannot relocate an unpinned key: {old}')
        if new in sources or new in moved.values():
            raise ValueError(f'relocation target already pinned: {new}')
        if Path(new).is_absolute() or not (root / new).is_file():
            raise ValueError(f'relocation target must be an existing repository-relative file: {new}')
        if sha(root / new) != sources[old]:
            raise ValueError(f'relocation target differs from the pinned digest of {old}: {new}')
        # Keep the key order stable apart from the renamed key.
        sources = {(new if k == old else k): v for k, v in sources.items()}
        moved[old] = new
    return dict(manifest, sources=sources), moved


def refreeze(manifest_path, root, reason, add=(), artifact_root=None, now=None, relocate=()):
    manifest_path, root = Path(manifest_path), Path(root)
    raw = manifest_path.read_bytes()
    pinned = json.loads(raw)
    manifest, relocated = relocate_keys(pinned, root, relocate)
    ok, missing, changed, borrowed, entry_ok = scan(manifest, root, artifact_root)
    if missing:
        raise ValueError(f'{len(missing)} pinned sources are missing; refusing to drop keys: {missing[:5]}')
    bad_borrowed = sorted(set(borrowed) & set(changed))
    if bad_borrowed:
        raise ValueError(f'artifact-root files differ from their pins; refusing to re-pin them: {bad_borrowed[:5]}')
    if not entry_ok:
        raise ValueError('archived previous public entrypoint changed')
    added = {}
    for name in add:
        if name in manifest['sources']:
            raise ValueError(f'already pinned: {name}')
        path, _ = locate(name, root)
        if path is None:
            raise ValueError(f'cannot pin missing path: {name}')
        added[name] = sha(path)
    if not changed and not added and not relocated:
        return None
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime('%Y%m%dT%H%M%SZ')
    archive = manifest_path.parent / 'previous' / f'runtime_source_manifest.{stamp}.json'
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(raw)
    new = dict(manifest)
    sources = dict(manifest['sources'])
    sources.update(changed)
    sources.update(added)
    new['sources'] = sources
    new['previous_runtime_manifest'] = {
        'archive': str(archive.relative_to(root)) if archive.is_relative_to(root) else str(archive),
        'sha256': hashlib.sha256(raw).hexdigest(), 'reason': reason, 'refrozen_utc': stamp,
        'changed': sorted(changed), 'added': sorted(added)}
    if relocated:
        new['previous_runtime_manifest']['relocated'] = relocated
    tmp = manifest_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(new, indent=2) + '\n')
    if set(json.loads(tmp.read_text())['sources']) != set(manifest['sources']) | set(added):
        tmp.unlink()
        raise ValueError('refrozen key set differs from the pinned key set')
    tmp.replace(manifest_path)
    result = {'archive': str(archive), 'changed': sorted(changed), 'added': sorted(added)}
    if relocated:
        result['relocated'] = relocated
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true', help='verify the checkout against the manifest; write nothing')
    p.add_argument('--reason', help='why the sources changed (required to re-pin)')
    p.add_argument('--add', action='append', default=[], help='pin an additional repository-relative path')
    p.add_argument('--relocate', action='append', default=[], metavar='OLD=NEW',
                   help='rename pinned key OLD to repository-relative NEW; NEW must have the digest pinned for OLD')
    p.add_argument('--artifact-root', type=Path, help='checkout to read missing gitignored artifacts from')
    p.add_argument('--manifest', type=Path, default=MANIFEST)
    args = p.parse_args(argv)
    artifact_root = args.artifact_root.resolve() if args.artifact_root else None
    if args.check:
        manifest = json.loads(args.manifest.read_text())
        ok, missing, changed, borrowed, entry_ok = scan(manifest, ROOT, artifact_root)
        total = len(manifest['sources'])
        print(f'{len(ok)}/{total} sources OK' + (f' ({len(borrowed)} read from {artifact_root})' if borrowed else ''))
        for name in missing:
            print(f'MISSING {name}')
        for name in changed:
            print(f'CHANGED {name}')
        if not entry_ok:
            print('CHANGED archived previous entrypoint')
        return 0 if len(ok) == total and entry_ok else 1
    if not args.reason:
        p.error('--reason is required to re-pin the manifest')
    relocate = []
    for item in args.relocate:
        old, sep, new = item.partition('=')
        if not sep or not old or not new:
            p.error(f'--relocate expects OLD=NEW, got {item!r}')
        relocate.append((old, new))
    result = refreeze(args.manifest, ROOT, args.reason, args.add, artifact_root, relocate=relocate)
    if result is None:
        print('manifest already matches the checkout; nothing written')
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f'refreeze: {exc}')
