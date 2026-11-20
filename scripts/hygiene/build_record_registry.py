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

Tamper evidence of the lifecycle log: each entry chains the sha256 of the entry
before it, entry 0 chains a digest of the fixed header (states, transitions,
state mapping, config and selection paths) and `head` holds the digest of the
last entry. That catches accidental edits, but anyone can re-chain a rewritten
log; the real guarantee is git history, which tests/test_record_registry.py
checks for append-only growth (in a full, non-shallow clone).

    python3 scripts/hygiene/build_record_registry.py          # rewrite generated files
    python3 scripts/hygiene/build_record_registry.py --check  # exit 1 if stale or invalid

Only the Python standard library and git are needed.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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
# Also matched, ignoring case: any name containing one of these words with a
# .txt or .md extension or no extension (TAKEOVER.md, takeover.txt, HANDOFF).
HANDOFF_WORDS = ('continuation', 'takeover', 'handoff')
HANDOFF_EXTS = ('.txt', '.md', '')
CURRENT_DOC = 'STATUS.md'
# Bannered historical handoffs whose names do not match HANDOFF_PATTERNS.
EXTRA_SUPERSEDED = ('analysis/particlegan_upgrade_20261117/README.txt',)
BANNER = re.compile(r'^SUPERSEDED by (?P<by>\S+) \((?P<date>\d{4}-\d{2}-\d{2})\)\.')
# Banners dated before the registry existed would claim a supersession that
# never happened; a later supersession may use its own date.
BANNER_EARLIEST = '2026-11-20'
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


def is_handoff_name(name):
    if any(fnmatch.fnmatchcase(name, pat) for pat in HANDOFF_PATTERNS):
        return True
    stem, ext = os.path.splitext(name.lower())
    return ext in HANDOFF_EXTS and any(w in stem for w in HANDOFF_WORDS)


def handoff_files(root=ROOT):
    """Every handoff document under root, as sorted repo-relative paths.

    Gitignored directories (runs/, data/) are not repository records and are
    not searched."""
    files = _git_files(root)
    if files is None:
        files = _walk_files(root)
    return sorted(p for p in files if is_handoff_name(p.rsplit('/', 1)[-1]))


class BannerError(ValueError):
    pass


def read_banner(path):
    """The SUPERSEDED banner of path, or None. Raises BannerError if unreadable."""
    try:
        with open(path, encoding='utf-8') as fh:
            first = fh.readline().rstrip('\r\n').lstrip('\ufeff')
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise BannerError(f'{path}: not UTF-8 text ({exc.reason} at byte {exc.start})') from None
    except OSError as exc:
        raise BannerError(f'{path}: cannot read ({exc.strerror})') from None
    m = BANNER.match(first)
    if not m or BANNER_HISTORICAL not in first:
        return None
    try:
        datetime.date.fromisoformat(m.group('date'))
    except ValueError:
        return None
    if m.group('date') < BANNER_EARLIEST:
        return None
    return {'by': m.group('by'), 'date': m.group('date')}


def _banner_or_error(path, errors):
    try:
        return read_banner(path)
    except BannerError as exc:
        errors.append(str(exc))
        return False


def build_doc_registry(root=ROOT, current=CURRENT_DOC):
    """Derive the document registry from banners. Raises on an unclassifiable handoff."""
    root = Path(root)
    docs, errors = {}, []
    for rel in sorted(set(handoff_files(root)) | {p for p in EXTRA_SUPERSEDED if (root/p).exists()}):
        if rel == current:
            continue
        banner = _banner_or_error(root/rel, errors)
        if banner is False:
            continue
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
            if not _banner_or_error(Path(root)/p, errors):
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
    # The replacing subject was reverted in the entry just before; the replaced
    # subject gets back exactly the state it held when it was replaced.
    ('REPLACED', 'reinstate', 'PROMOTED'),
    ('REPLACED', 'reinstate', 'OWNER_SELECTED_EXPERIMENTAL'),
)
TERMINAL_LIFECYCLE = ('REJECTED', 'REVERTED')  # REPLACED can be reinstated
HOLDING_STATES = ('OWNER_SELECTED_EXPERIMENTAL', 'UNDER_CONFIRMATION', 'CONFIRMATION_FAILED',
                  'CONFIRMED_QUALIFIED', 'PROMOTED')
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
    # A promotion recorded in this log: confirmation passed and selection.json
    # says so.
    'PROMOTED': {'config_status': 'qualified-default', 'selection_status': 'qualified_default',
                 'population_qualified': True, 'prospective_promotion_rule_passed': True},
    # A default imported from before this log (the 0.9.7 count-shape default,
    # status "development-selected"); selection.json describes another model.
    'PROMOTED_LEGACY': {'config_status': 'development-selected', 'selection_status': None,
                        'population_qualified': None, 'prospective_promotion_rule_passed': None},
}
CONFIG_PATH = 'configs/microglia_default.json'
SELECTION_PATH = 'analysis/default_empf_20261117/selection.json'
# Historical records whose stale claims the log must annotate (path, field).
REQUIRED_SUPERSEDED = (('analysis/empf_comparison_20261117/completion/status.json', 'default_changed'),)
HEADER_KEYS = ('schema', 'states', 'transitions', 'config', 'selection', 'state_mapping')
ENTRY_KEYS = {'seq', 'subject', 'at', 'event', 'from', 'to', 'evidence', 'actor', 'prev'}
OPTIONAL_KEYS = {'note', 'flags', 'replaced_by'}
TIME = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
SHA = re.compile(r'^[0-9a-f]{64}$')
COMMIT = re.compile(r'^[0-9a-f]{7,40}$')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def entry_digest(entry):
    return hashlib.sha256(canonical(entry).encode('utf-8')).hexdigest()


def allowed(frm, event, to):
    return (frm, event, to) in LIFECYCLE_TRANSITIONS


def header_digest(log):
    """Digest of the fixed header; entry 0 chains it as its `prev`."""
    return entry_digest({k: log.get(k) for k in HEADER_KEYS})


def _valid_time(at):
    if not isinstance(at, str) or not TIME.match(at):
        return False
    try:
        datetime.datetime.strptime(at, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        return False
    return True


def _transition_rows(rows):
    out = set()
    for t in rows if isinstance(rows, list) else []:
        if not isinstance(t, list) or len(t) != 3:
            return None
        out.add((t[0], t[1], t[2]))
    return out


def validate_lifecycle(log):
    """Errors for an append-only lifecycle log (structure, chain and transitions)."""
    if not isinstance(log, dict):
        return ['lifecycle log is not an object']
    errors = []
    entries = log.get('entries')
    if not isinstance(entries, list) or not entries:
        return ['lifecycle log has no entries']
    if list(log.get('states', [])) != list(LIFECYCLE_STATES):
        errors.append('declared states differ from the builder enum')
    declared = _transition_rows(log.get('transitions'))
    if declared != set(LIFECYCLE_TRANSITIONS):
        errors.append('declared transitions differ from the builder table')
    if log.get('state_mapping') != DEFAULT_HOLDING:
        errors.append('state_mapping differs from the builder DEFAULT_HOLDING table')
    if log.get('config') != CONFIG_PATH or log.get('selection') != SELECTION_PATH:
        errors.append(f'config/selection paths must be {CONFIG_PATH} and {SELECTION_PATH}')
    state, before_replace, prev, last_at = {}, {}, header_digest(log), ''
    replaced_by = {}
    seen_non_import = False
    for i, e in enumerate(entries):
        where = f'entry {i}'
        if not isinstance(e, dict):
            errors.append(f'{where}: not an object')
            prev = None
            continue
        keys = set(e)
        if not ENTRY_KEYS <= keys or keys - ENTRY_KEYS - OPTIONAL_KEYS:
            errors.append(f'{where}: keys {sorted(keys)}')
            prev = None
            continue
        if type(e['seq']) is not int or e['seq'] != i:
            errors.append(f'{where}: seq {e["seq"]!r} is not the integer {i}')
        if e['prev'] != prev:
            errors.append(f'{where}: prev digest does not chain to '
                          + ('the header' if i == 0 else f'entry {i - 1}'))
        prev = entry_digest(e)
        subject, event = e['subject'], e['event']
        if not isinstance(subject, str) or not subject or not isinstance(event, str):
            errors.append(f'{where}: subject and event must be non-empty strings')
            continue
        if not _valid_time(e['at']):
            errors.append(f'{where}: bad time {e["at"]!r}')
        elif e['at'] < last_at:
            errors.append(f'{where}: time goes backwards')
        else:
            last_at = e['at']
        if e['to'] not in LIFECYCLE_STATES or (e['from'] is not None and e['from'] not in LIFECYCLE_STATES):
            errors.append(f'{where}: unknown state')
        cur = state.get(subject)
        if e['from'] != cur:
            errors.append(f'{where}: from {e["from"]} but {subject} is in {cur}')
        if not allowed(e['from'], event, e['to']):
            errors.append(f'{where}: undeclared transition {e["from"]} -{event}-> {e["to"]}')
        if event == 'import_legacy_default' and seen_non_import:
            errors.append(f'{where}: import_legacy_default is only allowed before the first '
                          'lifecycle event; a new default must be declared and confirmed')
        if event != 'import_legacy_default':
            seen_non_import = True
        if event == 'owner_keep' and 'confirmation_failed' not in (e.get('flags') or []):
            errors.append(f'{where}: owner_keep after a failed confirmation must be flagged')
        if event == 'replace':
            by = e.get('replaced_by')
            nxt = entries[i + 1] if i + 1 < len(entries) else None
            if not isinstance(by, str) or by == subject:
                errors.append(f'{where}: replace must name the replacing subject in replaced_by')
            elif not (isinstance(nxt, dict) and nxt.get('subject') == by
                      and nxt.get('to') in HOLDING_STATES):
                errors.append(f'{where}: the next entry must move {by} into a default-holding state')
            replaced_by[subject] = by
            before_replace[subject] = e['from']
        elif 'replaced_by' in e:
            errors.append(f'{where}: replaced_by is only valid on a replace entry')
        if event == 'reinstate':
            by = replaced_by.get(subject)
            last = entries[i - 1] if i else None
            if not (isinstance(last, dict) and last.get('subject') == by and last.get('to') == 'REVERTED'):
                errors.append(f'{where}: reinstate needs the replacing subject {by} reverted in the '
                              'entry just before')
            if e['to'] != before_replace.get(subject):
                errors.append(f'{where}: reinstate must restore {before_replace.get(subject)}, '
                              f'not {e["to"]}')
        if not isinstance(e['actor'], str) or not e['actor']:
            errors.append(f'{where}: no actor')
        ev = e['evidence']
        if not isinstance(ev, list) or not ev:
            errors.append(f'{where}: no evidence')
        else:
            for item in ev:
                if not isinstance(item, dict) or not isinstance(item.get('path'), str) \
                        or not SHA.match(str(item.get('sha256', ''))) \
                        or not COMMIT.match(str(item.get('commit', ''))):
                    errors.append(f'{where}: evidence needs path, sha256 and commit: {item}')
        state[subject] = e['to']
    if log.get('head') != prev:
        errors.append('head does not match the digest of the last entry')
    return errors


def latest_states(log):
    state = {}
    for e in log['entries']:
        state[e['subject']] = e['to']
    return state


def holding_key(log, subject, state):
    """The DEFAULT_HOLDING row for subject in state (legacy imports differ)."""
    if state == 'PROMOTED':
        first = next(e for e in log['entries'] if e['subject'] == subject)
        if first['event'] == 'import_legacy_default':
            return 'PROMOTED_LEGACY'
    return state


def check_default_consistency(log, config, selection, records):
    """Errors when the latest lifecycle state disagrees with config/selection.

    `records` maps a repo path to its parsed JSON, for the superseded-record
    notes (e.g. completion/status.json default_changed:false)."""
    errors = []
    state = latest_states(log)
    holders = sorted(s for s, st in state.items() if st in HOLDING_STATES)
    if holders != [config.get('name')]:
        errors.append(f'default holders {holders} != configured default {config.get("name")!r}')
        return errors
    expect = DEFAULT_HOLDING[holding_key(log, config['name'], state[config['name']])]
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
    notes = log.get('superseded_records', [])
    for path, field in REQUIRED_SUPERSEDED:
        if not any(n.get('path') == path and n.get('field') == field for n in notes):
            errors.append(f'{path}: required superseded_records note for {field} is missing')
    for note in notes:
        target = by_seq.get(note.get('superseded_by_seq'))
        # The note stays valid after a later revert: it is about the override itself.
        if target is None or target['event'] != 'owner_override':
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
# Whole words only: INACTIVE and DELIVERED are not live.
LIVE_STATUS = re.compile(r'(?<![A-Za-z0-9])(?:LIVE|ACTIVE)(?![A-Za-z0-9])', re.I)
CAL_PREFIX = 'CAL_GPU0_V1_'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cal_record_paths(root=ROOT):
    analysis = Path(root)/'analysis'
    return sorted(f'analysis/{n}' for n in os.listdir(analysis)
                  if n.startswith(CAL_PREFIX) and CAL_RECORD.match(n) and (analysis/n).is_file())


def cal_unmatched_paths(root=ROOT):
    """CAL_GPU0_V1_* files the index cannot classify (they would be dropped silently)."""
    analysis = Path(root)/'analysis'
    index_name = CAL_INDEX.rsplit('/', 1)[-1]
    return sorted(f'analysis/{n}' for n in os.listdir(analysis)
                  if n.startswith(CAL_PREFIX) and n != index_name and not CAL_RECORD.match(n))


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
        if a['live_records'] and a.get('terminal_state') == 'ABANDONED_UNKNOWN':
            errors.append(f'{attempt}: live/active record and no result, stop or interruption '
                          'record; add the terminal record before indexing it')
    return errors


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def render(obj):
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + '\n'


STALE_TMP_SECONDS = 15 * 60


def _umask():
    old = os.umask(0o022)
    os.umask(old)
    return old


def remove_stale_temps(path, max_age=STALE_TMP_SECONDS, now=None):
    """Delete .<name>.*.tmp files a killed writer left beside path.

    Only files older than max_age go, so a concurrent writer's temp file is kept."""
    path = Path(path)
    now = time.time() if now is None else now
    removed = []
    for tmp in path.parent.glob(f'.{path.name}.*.tmp'):
        try:
            if now - tmp.stat().st_mtime > max_age:
                tmp.unlink()
                removed.append(tmp.name)
        except FileNotFoundError:
            pass
    return sorted(removed)


def atomic_write(path, text):
    """Write text to path via a same-directory temp file and os.replace.

    The result gets the usual umask-derived mode (mkstemp alone gives 0600)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def generated(root=ROOT):
    """The generated files. The CURRENT document is the CURRENT_DOC constant, never
    read back from the registry, so a damaged registry can always be regenerated."""
    root = Path(root)
    return {REGISTRY: render(build_doc_registry(root, CURRENT_DOC)),
            CAL_INDEX: render(build_cal_index(root))}


def load_json(root, rel):
    return json.loads((Path(root)/rel).read_text(encoding='utf-8'))


def check_all(root=ROOT):
    root = Path(root)
    errors = []
    try:
        outputs = generated(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    for rel, text in outputs.items():
        try:
            on_disk = (root/rel).read_text(encoding='utf-8')
        except FileNotFoundError:
            on_disk = None
        if on_disk != text:
            errors.append(f'{rel} is out of date; run {COMMAND}')
    errors += check_doc_registry(json.loads(outputs[REGISTRY]), root)
    errors += [f'{p}: CAL_GPU0_V1 file name does not match {CAL_RECORD.pattern}; it is not indexed'
               for p in cal_unmatched_paths(root)]
    errors += check_cal_index(json.loads(outputs[CAL_INDEX]))
    try:
        log = load_json(root, LIFECYCLE)
    except FileNotFoundError:
        return errors + [f'{LIFECYCLE}: missing']
    except ValueError as exc:
        return errors + [f'{LIFECYCLE}: not valid JSON ({exc})']
    errors += validate_lifecycle(log)
    if not errors:
        try:
            records = {n['path']: load_json(root, n['path'])
                       for n in log.get('superseded_records', []) if (root/n['path']).exists()}
            errors += check_default_consistency(log, load_json(root, CONFIG_PATH),
                                                load_json(root, SELECTION_PATH), records)
        except (OSError, ValueError) as exc:
            errors.append(f'default consistency inputs unreadable: {exc}')
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if stale or invalid')
    ap.add_argument('--root', default=str(ROOT), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not args.check:
        try:
            outputs = generated(root)
        except (OSError, ValueError) as exc:
            print(f'not written: {exc}', file=sys.stderr)
            return 1
        for rel, text in outputs.items():
            remove_stale_temps(root/rel)
            atomic_write(root/rel, text)
    try:
        errors = check_all(root)
    except OSError as exc:
        errors = [str(exc)]
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
