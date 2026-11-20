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

Source-epoch lease: every scripts/microglia_runtime.py process (and every
in-process consumer from its first check_sources()/activate()) holds a shared
lock on the source-epoch lease for its lifetime and records itself under
holders/. The lease directory is --epoch-dir, else $MICROGLIA_EPOCH_DIR, else
runs/.source-epoch of this checkout -- the same rule the runtime uses, so both
always lock the same file; a --manifest outside this checkout needs an explicit
--epoch-dir. A re-pin takes that lock exclusively and without waiting; while a
runtime process holds it the re-pin refuses and lists the live holders (pid,
command, digest, fit). --wait blocks until they have finished instead; while it
waits it holds the intent gate (gate.lock) exclusively, so runtimes that start
meanwhile queue behind it (and exit 75 after MICROGLIA_EPOCH_WAIT_S) instead
of starving it. The previous manifest is archived as
previous/runtime_source_manifest.<UTC>-<sha12 of its bytes>.json, so two re-pins
within one second never collide; older <UTC>.json archives stay readable.

Exit codes: 0 done (or nothing to do), 1 error / --check found differences,
2 usage, 75 runtime processes hold the source-epoch lease.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))
from alz_perturb_gan import durable_state as ds  # noqa: E402

MANIFEST = ROOT / 'analysis/default_empf_20261117/runtime_source_manifest.json'
EX_BUSY = 75
EPOCH_DIR_ENV = 'MICROGLIA_EPOCH_DIR'


class EpochBusy(RuntimeError):
    """Runtime processes hold the shared source-epoch lease; `holders` are the live holder records."""

    def __init__(self, directory, holders):
        self.directory, self.holders = Path(directory), holders
        listed = '; '.join(f"pid {h.get('pid')} ({' '.join(map(str, h.get('cmd') or []))}; digest "
                           f"{str(h.get('digest'))[:12]}; fit {h.get('fit')})" for h in holders) or \
            'no live holder record yet (a process is starting)'
        super().__init__(f'{len(holders)} runtime process(es) hold the source-epoch lease {self.directory/"lease"}; '
                         f'refusing to re-pin while they use the pinned sources (rerun with --wait to block): {listed}')


def epoch_dir(root=ROOT):
    """$MICROGLIA_EPOCH_DIR, else <root>/runs/.source-epoch: microglia_runtime.epoch_dir resolves it the same way."""
    return Path(os.environ.get(EPOCH_DIR_ENV) or Path(root) / 'runs/.source-epoch')


epoch_dir_of = epoch_dir  # earlier name; also used where a parameter shadows epoch_dir


def live_holders(directory):
    """Holder records whose process is still alive (same pid and start time)."""
    live = []
    for path in sorted((Path(directory) / 'holders').glob('*.json')):
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(info, dict) and info.get('pid') is not None and \
                ds.pid_starttime(info['pid']) == info.get('pid_starttime'):
            live.append({**info, 'record': path.name})
    return live


def exclusive_epoch(directory, wait=False):
    """The exclusive source-epoch lease (a FileLock to release), or EpochBusy listing the live holders.

    With the exclusive lock held no runtime process holds the shared one, so every holder record left
    behind belongs to a dead process and is removed.
    """
    directory = Path(directory)
    (directory / 'holders').mkdir(parents=True, exist_ok=True)
    # The intent gate first: runtimes pass it shared before taking the lease, so while this (possibly waiting)
    # refreeze holds it no new runtime can join the shared holders, and the existing ones drain.
    gate = ds.FileLock(directory / 'gate', exclusive=True, blocking=wait, purpose='refreeze runtime manifest (intent)')
    try:
        gate.acquire()
    except ds.Busy:
        raise EpochBusy(directory, live_holders(directory)) from None
    try:
        lock = ds.FileLock(directory / 'lease', exclusive=True, blocking=wait, purpose='refreeze runtime manifest')
        try:
            lock.acquire()
        except ds.Busy:
            raise EpochBusy(directory, live_holders(directory)) from None
    finally:
        gate.release()  # new runtimes now block on the lease itself until this refreeze releases it
    for path in (directory / 'holders').glob('*.json'):
        path.unlink(missing_ok=True)
    return lock


def archive_name(raw, stamp):
    """previous/ name of an archived manifest: <UTC>-<sha12 of its bytes>, unique even within one second."""
    return f'runtime_source_manifest.{stamp}-{hashlib.sha256(raw).hexdigest()[:12]}.json'


def find_archive(directory, digest):
    """The archived manifest with sha256 `digest`, under either name form (<UTC>.json or <UTC>-<sha12>.json)."""
    for path in sorted(Path(directory).glob('runtime_source_manifest.*.json')):
        if sha(path) == digest:
            return path
    return None


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
        inside = (root / new).resolve().is_relative_to(Path(root).resolve())
        if Path(new).is_absolute() or not inside or not (root / new).is_file():
            raise ValueError(f'relocation target must be an existing repository-relative file: {new}')
        if sha(root / new) != sources[old]:
            raise ValueError(f'relocation target differs from the pinned digest of {old}: {new}')
        # Keep the key order stable apart from the renamed key.
        sources = {(new if k == old else k): v for k, v in sources.items()}
        moved[old] = new
    return dict(manifest, sources=sources), moved


def refreeze(manifest_path, root, reason, add=(), artifact_root=None, now=None, relocate=(), epoch_dir=None,
             wait=False):
    """Re-pin under the exclusive source-epoch lease (EpochBusy while runtime processes hold it)."""
    lock = exclusive_epoch(epoch_dir if epoch_dir is not None else epoch_dir_of(root), wait)
    try:
        return _refreeze(manifest_path, root, reason, add, artifact_root, now, relocate)
    finally:
        lock.release()


def _refreeze(manifest_path, root, reason, add, artifact_root, now, relocate):
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
    archive = manifest_path.parent / 'previous' / archive_name(raw, stamp)
    if archive.exists() and archive.read_bytes() != raw:
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    ds.atomic_write_bytes(archive, raw)
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
    data = (json.dumps(new, indent=2) + '\n').encode()
    if set(json.loads(data)['sources']) != set(manifest['sources']) | set(added):
        raise ValueError('refrozen key set differs from the pinned key set')
    ds.atomic_write_bytes(manifest_path, data)
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
    p.add_argument('--wait', action='store_true',
                   help='block until no runtime process holds the source-epoch lease (default: refuse and list them)')
    p.add_argument('--epoch-dir', type=Path,
                   help=f'source-epoch lease directory (default: ${EPOCH_DIR_ENV}, else runs/.source-epoch of this '
                        'checkout; required with a --manifest outside this checkout)')
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
    if args.epoch_dir is None and not args.manifest.resolve().is_relative_to(ROOT.resolve()):
        p.error(f'--manifest {args.manifest} is outside this checkout ({ROOT}); pass --epoch-dir naming the '
                'source-epoch lease directory of the runtimes that use it')
    lease_dir = args.epoch_dir if args.epoch_dir is not None else epoch_dir(ROOT)
    relocate = []
    for item in args.relocate:
        old, sep, new = item.partition('=')
        if not sep or not old or not new:
            p.error(f'--relocate expects OLD=NEW, got {item!r}')
        relocate.append((old, new))
    try:
        result = refreeze(args.manifest, ROOT, args.reason, args.add, artifact_root, relocate=relocate,
                          epoch_dir=lease_dir, wait=args.wait)
    except EpochBusy as busy:
        print(f'refreeze: {busy}', file=sys.stderr)
        return EX_BUSY
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
