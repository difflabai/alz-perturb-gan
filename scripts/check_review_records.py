#!/usr/bin/env python3
"""Compare the digests that review records bind with the live files; report STALE ones. Never writes.

analysis/default_empf_20261117/independent_application_review.json and verification.json record the
sha256 of the files they reviewed (the application sources, the runtime source manifest, the config, the
selection records, the generation receipt and the verified archive). When one of those files changes
afterwards, the record still reads as a pass for bytes that no longer exist. This tool lists every bound
digest as:

  OK      the live file has the recorded digest
  STALE   the live file exists and differs: the record no longer covers it
  ABSENT  the file is not in this checkout (gitignored local artifacts under runs/ or artifacts/)

It is a warning, not a gate: scripts/microglia.py --dry-run includes its summary under
preflight.review_records, and tests run it. The records are never rewritten; a new review writes a new
record. Exit code: 0, or 1 with --strict when anything is STALE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = 'analysis/default_empf_20261117'
RECORDS = (f'{RECORD_DIR}/independent_application_review.json', f'{RECORD_DIR}/verification.json')
MANIFEST = f'{RECORD_DIR}/runtime_source_manifest.json'
# Top-level digest keys whose file is implied by the key name.
IMPLIED = {
    'runtime_source_manifest_sha256': MANIFEST,
    'config_sha256': 'configs/microglia_default.json',
    'selection_sha256': f'{RECORD_DIR}/selection.json',
    'selected_model_sha256': f'{RECORD_DIR}/selected_model.json',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def bound_digests(record):
    """(json path, repository path, recorded sha256) for every file digest the record binds."""
    found = []

    def walk(node, where):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            here = f'{where}.{key}' if where else key
            if key.startswith('reviewed_') and key.endswith('_sha256') and isinstance(value, dict):
                found.extend((f'{here}.{name}', name, digest) for name, digest in value.items() if _is_digest(digest))
            elif key in IMPLIED and _is_digest(value) and not where:
                found.append((here, IMPLIED[key], value))
            elif key.endswith('_sha256') and _is_digest(value):
                stem = key[:-len('_sha256')]
                path = node.get(stem + '_path')
                if isinstance(path, str):
                    found.append((here, path, value))
            elif isinstance(value, dict):
                path = value.get('path') or value.get('local_path')
                if isinstance(path, str) and _is_digest(value.get('sha256')):
                    found.append((f'{here}.sha256', path, value['sha256']))
                walk(value, here)
    walk(record, '')
    return found


def check(root=ROOT, records=RECORDS):
    """[{record, field, path, recorded, live, status}] for every bound digest of every record."""
    root = Path(root)
    rows = []
    for name in records:
        record = json.loads((root / name).read_text())
        seen = set()
        for field, path, recorded in bound_digests(record):
            if (path, recorded) in seen:
                continue
            seen.add((path, recorded))
            target = root / path
            live = _sha(target) if target.is_file() else None
            status = 'ABSENT' if live is None else 'OK' if live == recorded else 'STALE'
            rows.append({'record': name, 'field': field, 'path': path, 'recorded': recorded, 'live': live,
                         'status': status})
    return rows


def summary(rows):
    stale = sorted({row['path'] for row in rows if row['status'] == 'STALE'})
    return {'ok': not stale, 'checked': len(rows), 'stale': stale,
            'absent': sorted({row['path'] for row in rows if row['status'] == 'ABSENT'}),
            'note': 'warning only: review records bind digests of files that changed since the review'
                    if stale else 'every bound digest that is present matches'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--json', action='store_true', help='print every row as JSON')
    p.add_argument('--strict', action='store_true', help='exit 1 when any bound digest is STALE')
    args = p.parse_args(argv)
    rows = check(args.root)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            if row['status'] != 'OK':
                print(f"{row['status']:6} {row['path']}  ({row['record']}: {row['field']})")
        result = summary(rows)
        print(f"{result['checked']} bound digests; {len(result['stale'])} stale files, "
              f"{len(result['absent'])} absent here")
    return 1 if args.strict and not summary(rows)['ok'] else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f'check_review_records: {exc}')
