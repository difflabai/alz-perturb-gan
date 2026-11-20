#!/usr/bin/env python3
"""Build and check the record-authority registries.

Three record lifecycles in this repository were implicit: which handoff file is
authoritative, where the default model stands, and how the legacy CAL_GPU0_*
job records ended. This script makes each one an explicit, checked state:

* analysis/RECORD_REGISTRY.json (generated): every handoff document
  (``*CONTINUATION*.txt``, ``*TAKEOVER*.txt``, ``*HANDOFF*``) and its state,
  CURRENT or SUPERSEDED(by, date). Exactly one document is CURRENT (STATUS.md).
* analysis/CAL_GPU0_V1_INDEX.json (generated): each legacy CAL_GPU0_V1 attempt,
  its records, its resumption chain (``_resumeN``) and one terminal state.
* analysis/default_empf_20261117/default_lifecycle.json (hand-appended log):
  the default-model promotion lifecycle. This script only validates it.

Nothing is moved or rewritten; the generated files are derived from file names,
banners and status fields, so two runs give byte-identical output.

    python3 scripts/hygiene/build_record_registry.py          # rewrite generated files
    python3 scripts/hygiene/build_record_registry.py --check  # exit 1 if stale or invalid

Only the Python standard library and git are needed.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMAND = 'python3 scripts/hygiene/build_record_registry.py'

# --------------------------------------------------------------------------
# (1) Handoff-document authority
# --------------------------------------------------------------------------
REGISTRY = 'analysis/RECORD_REGISTRY.json'
DOC_STATES = ('CURRENT', 'SUPERSEDED')
DOC_TRANSITIONS = (('CURRENT', 'supersede', 'SUPERSEDED'),)
HANDOFF_PATTERNS = ('*CONTINUATION*.txt', '*TAKEOVER*.txt', '*HANDOFF*')
CURRENT_DOC = 'STATUS.md'
# Bannered historical handoffs whose names do not match HANDOFF_PATTERNS.
EXTRA_SUPERSEDED = ('analysis/particlegan_upgrade_20261117/README.txt',)
BANNER = re.compile(r'^SUPERSEDED by (?P<by>\S+) \((?P<date>\d{4}-\d{2}-\d{2})\)\.')
# The banner must say that everything below it, including any "CURRENT",
# "AUTHORITATIVE", "ACTIVE" or "LIVE" paragraph, is historical.
BANNER_HISTORICAL = 'Historical record, kept unchanged below'
BANNER_TEXT = ('SUPERSEDED by STATUS.md (2026-11-20). Historical record, kept unchanged '
               'below; its statements about the current default, live jobs and next '
               'steps are out of date.')
SKIP_DIRS = {'.git', 'runs', 'artifacts', 'data', 'node_modules', '__pycache__',
             '.pytest_cache', '.venv', 'dist', 'build'}


def _git_files(root):
    """Tracked plus untracked-but-not-ignored files, or None outside a git checkout."""
    root = Path(root)
    try:
        top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if Path(top).resolve() != root.resolve():
        return None
    out = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others',
                          '--exclude-standard'], capture_output=True, check=True).stdout
    return sorted({p for p in out.decode('utf-8').split('\0') if p and (root/p).is_file()})


def _walk_files(root):
    root = Path(root)
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        for name in files:
            found.append((Path(base)/name).relative_to(root).as_posix())
    return sorted(found)


def handoff_files(root=ROOT):
    """Every handoff document under root, as sorted repo-relative paths."""
    files = _git_files(root)
    if files is None:
        files = _walk_files(root)
    return sorted(p for p in files
                  if any(fnmatch.fnmatchcase(p.rsplit('/', 1)[-1], pat) for pat in HANDOFF_PATTERNS))


def read_banner(path):
    try:
        with open(path, encoding='utf-8') as fh:
            first = fh.readline().rstrip('\n')
    except FileNotFoundError:
        return None
    m = BANNER.match(first)
    if not m or BANNER_HISTORICAL not in first:
        return None
    return {'by': m.group('by'), 'date': m.group('date')}


def build_doc_registry(root=ROOT, current=CURRENT_DOC):
    """Derive the document registry from banners. Raises on an unclassifiable handoff."""
    root = Path(root)
    docs, errors = {}, []
    for rel in sorted(set(handoff_files(root)) | {p for p in EXTRA_SUPERSEDED if (root/p).exists()}):
        if rel == current:
            continue
        banner = read_banner(root/rel)
        if banner is None:
            errors.append(f'{rel}: handoff has no SUPERSEDED banner stating that the text '
                          f'below is historical, and is not the CURRENT document {current}')
            continue
        if banner['by'] != current:
            errors.append(f'{rel}: superseded by {banner["by"]}, which is not CURRENT ({current})')
        docs[rel] = {'state': 'SUPERSEDED', 'by': banner['by'], 'date': banner['date']}
    if not (root/current).exists():
        errors.append(f'CURRENT document {current} does not exist')
    docs[current] = {'state': 'CURRENT', 'since': '2026-11-20',
                     'evidence': {'commit': '2b7ca0f', 'path': current,
                                  'note': 'commit that made STATUS.md the single current index'}}
    if errors:
        raise ValueError('\n'.join(errors))
    return {
        'schema': 1,
        'generated_by': COMMAND,
        'purpose': ('Which handoff document is authoritative. Exactly one document is '
                    'CURRENT; every other handoff carries a SUPERSEDED banner as its first '
                    'line, and any CURRENT/AUTHORITATIVE/ACTIVE/LIVE text below that '
                    'banner is historical.'),
        'states': list(DOC_STATES),
        'transitions': [list(t) for t in DOC_TRANSITIONS],
        'patterns': list(HANDOFF_PATTERNS),
        'banner': BANNER_TEXT,
        'current': current,
        'documents': {k: docs[k] for k in sorted(docs)},
    }


def check_doc_registry(registry, root=ROOT):
    """Errors for a document registry against the files under root."""
    errors = []
    docs = registry.get('documents', {})
    current = [p for p, d in docs.items() if d.get('state') == 'CURRENT']
    if len(current) != 1:
        errors.append(f'exactly one CURRENT document required, found {current}')
    for p, d in docs.items():
        if d.get('state') not in DOC_STATES:
            errors.append(f'{p}: unknown state {d.get("state")!r}')
        if d.get('state') == 'SUPERSEDED':
            if d.get('by') not in current:
                errors.append(f'{p}: superseded by {d.get("by")!r}, which is not CURRENT')
            if read_banner(Path(root)/p) is None:
                errors.append(f'{p}: registered SUPERSEDED but its first line has no banner')
    for p in handoff_files(root):
        if p not in docs:
            errors.append(f'{p}: handoff not in {REGISTRY}')
    return errors


# --------------------------------------------------------------------------
# (2) Default-model promotion lifecycle
# --------------------------------------------------------------------------
LIFECYCLE = 'analysis/default_empf_20261117/default_lifecycle.json'
LIFECYCLE_STATES = ('DECLARED', 'PRELAUNCH_PASSED', 'RUNNING', 'COMPLETED', 'AUDITED',
                    'OWNER_SELECTED_EXPERIMENTAL', 'UNDER_CONFIRMATION', 'CONFIRMED_QUALIFIED',
                    'CONFIRMATION_FAILED', 'PROMOTED', 'REJECTED', 'REVERTED', 'REPLACED')
# (from, event, to). None as `from` is the first entry of a subject.
LIFECYCLE_TRANSITIONS = (
    (None, 'declare', 'DECLARED'),
    (None, 'import_legacy_default', 'PROMOTED'),  # history that predates this log
    ('DECLARED', 'prelaunch_pass', 'PRELAUNCH_PASSED'),
    ('DECLARED', 'reject', 'REJECTED'),
    ('PRELAUNCH_PASSED', 'launch', 'RUNNING'),
    ('PRELAUNCH_PASSED', 'reject', 'REJECTED'),
    ('RUNNING', 'complete', 'COMPLETED'),
    ('RUNNING', 'reject', 'REJECTED'),
    ('COMPLETED', 'audit', 'AUDITED'),
    ('COMPLETED', 'reject', 'REJECTED'),
    ('AUDITED', 'reject', 'REJECTED'),
    ('COMPLETED', 'owner_override', 'OWNER_SELECTED_EXPERIMENTAL'),
    ('AUDITED', 'owner_override', 'OWNER_SELECTED_EXPERIMENTAL'),
    ('OWNER_SELECTED_EXPERIMENTAL', 'start_confirmation', 'UNDER_CONFIRMATION'),
    ('OWNER_SELECTED_EXPERIMENTAL', 'revert', 'REVERTED'),
    ('UNDER_CONFIRMATION', 'pass', 'CONFIRMED_QUALIFIED'),
    ('UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
    ('CONFIRMATION_FAILED', 'owner_keep', 'OWNER_SELECTED_EXPERIMENTAL'),  # must be flagged
    ('CONFIRMATION_FAILED', 'revert', 'REVERTED'),
    ('CONFIRMED_QUALIFIED', 'promote', 'PROMOTED'),
    ('PROMOTED', 'replace', 'REPLACED'),
    ('PROMOTED', 'revert', 'REVERTED'),
    ('OWNER_SELECTED_EXPERIMENTAL', 'replace', 'REPLACED'),
)
TERMINAL_LIFECYCLE = ('REJECTED', 'REVERTED', 'REPLACED')
# Read-only mapping from lifecycle state to the vocabularies of
# configs/microglia_default.json ("status") and selection.json ("status" and
# flags). States not listed here never hold the default. None means "not
# constrained by this state".
DEFAULT_HOLDING = {
    'OWNER_SELECTED_EXPERIMENTAL': {
        'config_status': 'owner-selected-experimental',
        'selection_status': 'owner_selected_experimental_default',
        'population_qualified': False, 'prospective_promotion_rule_passed': False},
    'UNDER_CONFIRMATION': {
        'config_status': 'owner-selected-experimental',
        'selection_status': 'owner_selected_experimental_default',
        'population_qualified': False, 'prospective_promotion_rule_passed': False},
    'CONFIRMATION_FAILED': {
        'config_status': 'owner-selected-experimental',
        'selection_status': 'owner_selected_experimental_default',
        'population_qualified': False, 'prospective_promotion_rule_passed': False},
    'CONFIRMED_QUALIFIED': {
        'config_status': 'owner-selected-experimental',
        'selection_status': 'owner_selected_experimental_default',
        'population_qualified': None, 'prospective_promotion_rule_passed': None},
    # The 0.9.7 count-shape default used "development-selected" while promoted.
    'PROMOTED': {'config_status': 'development-selected', 'selection_status': None,
                 'population_qualified': None, 'prospective_promotion_rule_passed': None},
}
ENTRY_KEYS = {'seq', 'subject', 'at', 'event', 'from', 'to', 'evidence', 'actor', 'prev'}
OPTIONAL_KEYS = {'note', 'flags'}
TIME = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
SHA = re.compile(r'^[0-9a-f]{64}$')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def entry_digest(entry):
    return hashlib.sha256(canonical(entry).encode('utf-8')).hexdigest()


def allowed(frm, event, to):
    return (frm, event, to) in LIFECYCLE_TRANSITIONS


def validate_lifecycle(log):
    """Errors for an append-only lifecycle log (structure, chain and transitions)."""
    errors = []
    entries = log.get('entries')
    if not isinstance(entries, list) or not entries:
        return ['lifecycle log has no entries']
    if list(log.get('states', [])) != list(LIFECYCLE_STATES):
        errors.append('declared states differ from the builder enum')
    declared = {tuple(t) if t[0] is not None else (None, t[1], t[2])
                for t in log.get('transitions', [])}
    if declared != set(LIFECYCLE_TRANSITIONS):
        errors.append('declared transitions differ from the builder table')
    state, prev, last_at = {}, None, ''
    for i, e in enumerate(entries):
        where = f'entry {i}'
        if not isinstance(e, dict):
            errors.append(f'{where}: not an object')
            continue
        keys = set(e)
        if not ENTRY_KEYS <= keys or keys - ENTRY_KEYS - OPTIONAL_KEYS:
            errors.append(f'{where}: keys {sorted(keys)}')
            continue
        if e['seq'] != i:
            errors.append(f'{where}: seq {e["seq"]} out of order')
        if e['prev'] != prev:
            errors.append(f'{where}: prev digest does not chain to entry {i - 1}')
        prev = entry_digest(e)
        if not isinstance(e['at'], str) or not TIME.match(e['at']):
            errors.append(f'{where}: bad time {e["at"]!r}')
        elif e['at'] < last_at:
            errors.append(f'{where}: time goes backwards')
        else:
            last_at = e['at']
        if e['to'] not in LIFECYCLE_STATES or (e['from'] is not None and e['from'] not in LIFECYCLE_STATES):
            errors.append(f'{where}: unknown state')
        cur = state.get(e['subject'])
        if e['from'] != cur:
            errors.append(f'{where}: from {e["from"]} but {e["subject"]} is in {cur}')
        if not allowed(e['from'], e['event'], e['to']):
            errors.append(f'{where}: undeclared transition {e["from"]} -{e["event"]}-> {e["to"]}')
        if e['event'] == 'owner_keep' and 'confirmation_failed' not in e.get('flags', []):
            errors.append(f'{where}: owner_keep after a failed confirmation must be flagged')
        if not isinstance(e['actor'], str) or not e['actor']:
            errors.append(f'{where}: no actor')
        ev = e['evidence']
        if not isinstance(ev, list) or not ev:
            errors.append(f'{where}: no evidence')
        else:
            for item in ev:
                if not isinstance(item, dict) or not isinstance(item.get('path'), str) \
                        or not SHA.match(str(item.get('sha256', ''))):
                    errors.append(f'{where}: evidence needs path and sha256: {item}')
        state[e['subject']] = e['to']
    return errors


def latest_states(log):
    state = {}
    for e in log['entries']:
        state[e['subject']] = e['to']
    return state


def check_default_consistency(log, config, selection, records):
    """Errors when the latest lifecycle state disagrees with config/selection.

    `records` maps a repo path to its parsed JSON, for the superseded-record
    notes (e.g. completion/status.json default_changed:false)."""
    errors = []
    state = latest_states(log)
    holders = sorted(s for s, st in state.items() if st in DEFAULT_HOLDING)
    if holders != [config.get('name')]:
        errors.append(f'default holders {holders} != configured default {config.get("name")!r}')
        return errors
    expect = DEFAULT_HOLDING[state[config['name']]]
    if config.get('status') != expect['config_status']:
        errors.append(f'config status {config.get("status")!r} != {expect["config_status"]!r} '
                      f'for lifecycle state {state[config["name"]]}')
    if expect['selection_status'] is not None and selection.get('status') != expect['selection_status']:
        errors.append(f'selection status {selection.get("status")!r} != {expect["selection_status"]!r}')
    for flag in ('population_qualified', 'prospective_promotion_rule_passed'):
        if expect[flag] is not None and selection.get(flag) is not expect[flag]:
            errors.append(f'selection {flag}={selection.get(flag)!r}, lifecycle expects {expect[flag]}')
    if config.get('selection', {}).get('population_qualified') != selection.get('population_qualified'):
        errors.append('config and selection disagree on population_qualified')
    by_seq = {e['seq']: e for e in log['entries']}
    for note in log.get('superseded_records', []):
        target = by_seq.get(note.get('superseded_by_seq'))
        if target is None or target['event'] != 'owner_override' or target['subject'] != config['name']:
            errors.append(f'{note.get("path")}: superseded_by_seq does not name the owner_override entry')
        rec = records.get(note.get('path'))
        if rec is None:
            errors.append(f'{note.get("path")}: superseded record missing')
        elif rec.get(note['field']) != note['recorded_value']:
            errors.append(f'{note["path"]}: {note["field"]}={rec.get(note["field"])!r}, '
                          f'registry note says {note["recorded_value"]!r}')
    return errors


# --------------------------------------------------------------------------
# (3) Legacy CAL_GPU0_V1 job records
# --------------------------------------------------------------------------
CAL_INDEX = 'analysis/CAL_GPU0_V1_INDEX.json'
CAL_RECORD = re.compile(r'^CAL_GPU0_V1_(?P<attempt>.+?_f\d+_[a-z0-9]+_s\d+)-(?P<kind>[A-Za-z0-9-]+)\.json$')
RESUME = re.compile(r'_resume(?P<n>\d+)(?=_f\d+_[a-z0-9]+_s\d+$)')
# Resumed attempts drop the base attempt's "_v1" (routed_e22_v1 -> routed_e22_resume1).
LINEAGE_STRIP = re.compile(r'_(?:v1|resume\d+)(?=_f\d+_[a-z0-9]+_s\d+$)')
TERMINAL_STATES = ('RESULT', 'INTERRUPTED', 'INCOMPLETE_STOP', 'ABANDONED_UNKNOWN')
RESULT_KINDS = ('result', 'confirmation', 'qualification', 'external-audit')
LIVE_STATUS = re.compile(r'LIVE|ACTIVE', re.I)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cal_record_paths(root=ROOT):
    analysis = Path(root)/'analysis'
    return sorted(f'analysis/{n}' for n in os.listdir(analysis)
                  if n.startswith('CAL_GPU0_V1_') and CAL_RECORD.match(n) and (analysis/n).is_file())


def _status_fields(data):
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in sorted(data.items())
            if isinstance(v, str) and (k == 'status' or k.endswith('_status'))}


def classify(kinds):
    """(terminal_state, kinds that justify it) for one attempt's record kinds."""
    for state, pick in (('RESULT', lambda k: k in RESULT_KINDS),
                        ('INCOMPLETE_STOP', lambda k: k.startswith('incomplete-')),
                        ('INTERRUPTED', lambda k: 'interrupt' in k)):
        hit = sorted(k for k in kinds if pick(k))
        if hit:
            return state, hit
    return 'ABANDONED_UNKNOWN', []


def build_cal_index(root=ROOT):
    root = Path(root)
    attempts = {}
    paths = cal_record_paths(root)
    for rel in paths:
        m = CAL_RECORD.match(rel.rsplit('/', 1)[-1])
        attempt, kind = m.group('attempt'), m.group('kind')
        try:
            data = json.loads((root/rel).read_text(encoding='utf-8'))
        except ValueError:
            data = None
        status = _status_fields(data)
        live = 'live' in kind.split('-') or any(LIVE_STATUS.search(v) for v in status.values())
        a = attempts.setdefault(attempt, {'records': []})
        a['records'].append({'path': rel, 'kind': kind, 'sha256': sha256_file(root/rel),
                             'status': status, 'live': live})
    lineages = {}
    for attempt in sorted(attempts):
        r = RESUME.search(attempt)
        lineage = LINEAGE_STRIP.sub('', attempt)
        a = attempts[attempt]
        a['lineage'] = lineage
        a['resume_index'] = int(r.group('n')) if r else 0
        lineages.setdefault(lineage, []).append(attempt)
        a['records'].sort(key=lambda x: (x['kind'], x['path']))
        state, kinds = classify([x['kind'] for x in a['records']])
        a['terminal_state'] = state
        if kinds:
            a['terminal_evidence'] = [x['path'] for x in a['records'] if x['kind'] in kinds]
        else:  # no terminal record: cite the last record as the evidence of abandonment
            a['terminal_evidence'] = [a['records'][-1]['path']]
            a['terminal_note'] = ('no result, stop or interruption record; abandoned or '
                                  'unrecorded outcome')
        a['live_records'] = [x['path'] for x in a['records'] if x['live']]
    lineage_out = {}
    for lineage, members in sorted(lineages.items()):
        chain = sorted(members, key=lambda x: attempts[x]['resume_index'])
        idx = [attempts[x]['resume_index'] for x in chain]
        for i, x in enumerate(chain):
            attempts[x]['resumed_by'] = chain[i + 1] if i + 1 < len(chain) else None
            attempts[x]['resumes'] = chain[i - 1] if i else None
        lineage_out[lineage] = {'chain': chain, 'resume_indices': idx,
                                'terminal_state': attempts[chain[-1]]['terminal_state']}
    stale = sorted(x for x, a in attempts.items()
                   if a['live_records'] and a['terminal_state'] != 'RESULT')
    return {
        'schema': 1,
        'generated_by': COMMAND,
        'purpose': ('Read-only map of the legacy flat analysis/CAL_GPU0_V1_* job records. '
                    'File-name suffixes were the only job state; each attempt now has one '
                    'terminal state. No file was moved.'),
        'pattern': CAL_RECORD.pattern,
        'terminal_states': list(TERMINAL_STATES),
        'terminal_rule': {'RESULT': list(RESULT_KINDS), 'INCOMPLETE_STOP': 'kind starts with incomplete-',
                          'INTERRUPTED': 'kind contains interrupt',
                          'ABANDONED_UNKNOWN': 'none of the above',
                          'priority': list(TERMINAL_STATES)},
        'record_count': len(paths),
        'attempt_count': len(attempts),
        'stale_live_attempts': stale,
        'attempts': {k: attempts[k] for k in sorted(attempts)},
        'lineages': lineage_out,
    }


def check_cal_index(index):
    errors = []
    for attempt, a in index['attempts'].items():
        if a.get('terminal_state') not in TERMINAL_STATES:
            errors.append(f'{attempt}: no terminal state')
        if a['live_records'] and not a.get('terminal_evidence'):
            errors.append(f'{attempt}: live/active record without terminal evidence')
    return errors


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def render(obj):
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + '\n'


def atomic_write(path, text):
    """Write text to path via a same-directory temp file and os.replace."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def generated(root=ROOT):
    root = Path(root)
    current = CURRENT_DOC
    try:
        current = json.loads((root/REGISTRY).read_text(encoding='utf-8')).get('current', CURRENT_DOC)
    except (FileNotFoundError, ValueError):
        pass
    return {REGISTRY: render(build_doc_registry(root, current)),
            CAL_INDEX: render(build_cal_index(root))}


def load_json(root, rel):
    return json.loads((Path(root)/rel).read_text(encoding='utf-8'))


def check_all(root=ROOT):
    root = Path(root)
    errors = []
    try:
        outputs = generated(root)
    except ValueError as exc:
        return [str(exc)]
    for rel, text in outputs.items():
        try:
            on_disk = (root/rel).read_text(encoding='utf-8')
        except FileNotFoundError:
            on_disk = None
        if on_disk != text:
            errors.append(f'{rel} is out of date; run {COMMAND}')
    errors += check_doc_registry(json.loads(outputs[REGISTRY]), root)
    errors += check_cal_index(json.loads(outputs[CAL_INDEX]))
    log = load_json(root, LIFECYCLE)
    errors += validate_lifecycle(log)
    if not errors:
        records = {n['path']: load_json(root, n['path']) for n in log.get('superseded_records', [])
                   if (root/n['path']).exists()}
        errors += check_default_consistency(log, load_json(root, 'configs/microglia_default.json'),
                                            load_json(root, log['selection']), records)
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if stale or invalid')
    ap.add_argument('--root', default=str(ROOT), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not args.check:
        for rel, text in generated(root).items():
            atomic_write(root/rel, text)
    errors = check_all(root)
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
