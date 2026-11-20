"""The default-model lifecycle log, the CAL_GPU0_V1 index and the pin guard.

CPU only and offline. Checks that need git history skip outside a git checkout.
"""
import copy
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/hygiene/build_record_registry.py'
spec = importlib.util.spec_from_file_location('build_record_registry', SCRIPT)
REG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REG)

LOG = json.loads((ROOT/REG.LIFECYCLE).read_text(encoding='utf-8'))
CONFIG = json.loads((ROOT/'configs/microglia_default.json').read_text(encoding='utf-8'))
SELECTION = json.loads((ROOT/LOG['selection']).read_text(encoding='utf-8'))
STATUS_JSON = 'analysis/empf_comparison_20261117/completion/status.json'
RECORDS = {STATUS_JSON: json.loads((ROOT/STATUS_JSON).read_text(encoding='utf-8'))}
STALE_LIVE = {
    'pure_confirm_0da2a09092c823377ca5_f0_pg_s1': 'INTERRUPTED',
    'released_rate_g025d050p050_f0_pg_s0': 'INCOMPLETE_STOP',
    'released_rate_g050d050p050_f0_pg_s0': 'INCOMPLETE_STOP',
    'released_rate_g100d050p050_f0_pg_s0': 'INCOMPLETE_STOP',
}
# Handoffs that received their SUPERSEDED banner together with this registry.
BANNERED_HERE = tuple(f'analysis/{d}_20261114/TAKEOVER.txt' for d in (
    'amplitude_challenge', 'branch_geometry', 'count_dependence', 'covariance_challenge',
    'dispersion_challenge', 'empirical_challenge', 'hybrid_challenge', 'paired_challenge')) + \
    ('analysis/improve_20261109/TAKEOVER_20261113.txt',)


def git(*args, root=ROOT):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True).stdout


def in_git():
    try:
        git('rev-parse', 'HEAD')
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def chain(entries, log=None):
    """Renumber and re-chain entries from the header digest, as an honest appender would."""
    out, prev = [], REG.header_digest(LOG if log is None else log)
    for i, e in enumerate(entries):
        e = dict(e, seq=i, prev=prev)
        prev = REG.entry_digest(e)
        out.append(e)
    return out


def with_entries(entries, log=None):
    log = copy.deepcopy(LOG if log is None else log)
    log['entries'] = chain(entries, log)
    log['head'] = REG.entry_digest(log['entries'][-1])
    return log


def step(subject, frm, event, to, at='2026-11-21T00:00:00Z', **extra):
    e = {'subject': subject, 'from': frm, 'event': event, 'to': to, 'at': at, 'actor': 'test',
         'evidence': [{'path': 'x', 'sha256': '0' * 64, 'commit': '0' * 40}]}
    e.update(extra)
    return e


def is_shallow():
    return git('rev-parse', '--is-shallow-repository').decode().strip() == 'true'


# ---------------------------------------------------------------- (a) log validation
def test_committed_log_is_valid():
    assert REG.validate_lifecycle(LOG) == []
    assert REG.latest_states(LOG) == {'microglia-pbulk-countshape-v2': 'REPLACED',
                                      'microglia-empf-countshape-v1': 'UNDER_CONFIRMATION'}


def test_every_consecutive_pair_is_declared():
    declared = {(t[0], t[1], t[2]) for t in LOG['transitions']}
    last = {}
    for e in LOG['entries']:
        assert (last.get(e['subject']), e['event'], e['to']) in declared
        assert e['from'] == last.get(e['subject'])
        last[e['subject']] = e['to']


def test_initial_reconstruction_cites_committed_evidence():
    commits = {ev.get('commit', '')[:7] for e in LOG['entries'] for ev in e['evidence']}
    assert {'d578e83', '2b7ca0f', 'dc8e101', '77b2d7d'} <= commits
    paths = {ev['path'] for e in LOG['entries'] for ev in e['evidence']}
    assert {'analysis/default_empf_20261117/selection.json', STATUS_JSON} <= paths


def test_evidence_hashes_match_their_commits():
    if not in_git():
        pytest.skip('not a git checkout')
    shallow, checked, missing = is_shallow(), 0, []
    for e in LOG['entries']:
        for ev in e['evidence']:
            try:
                blob = git('show', f'{ev["commit"]}:{ev["path"]}')
            except subprocess.CalledProcessError:
                missing.append((e['seq'], ev['commit'], ev['path']))
                continue
            assert hashlib.sha256(blob).hexdigest() == ev['sha256'], (e['seq'], ev['path'])
            checked += 1
    # An unresolvable commit is fabricated evidence, unless the clone is shallow.
    assert shallow or not missing, missing
    if not checked:
        pytest.skip('shallow clone: evidence commits not available')


def test_log_is_append_only_across_history():
    if not in_git():
        pytest.skip('not a git checkout')
    if is_shallow():
        pytest.skip('shallow clone: history cannot show the log is append-only (use fetch-depth: 0)')
    revs = git('log', '--format=%H', '--', REG.LIFECYCLE).decode().split()
    for rev in revs:
        old = json.loads(git('show', f'{rev}:{REG.LIFECYCLE}'))
        if 'head' not in old:
            continue  # before the header digest was introduced (never pushed)
        assert LOG['entries'][:len(old['entries'])] == old['entries'], rev
        assert {k: LOG[k] for k in REG.HEADER_KEYS} == {k: old[k] for k in REG.HEADER_KEYS}, rev
        notes = old.get('superseded_records', [])
        assert LOG['superseded_records'][:len(notes)] == notes, rev


def test_header_and_head_are_anchored():
    """Header fields and the last entry are covered by the chain (adversary: edit in place)."""
    assert LOG['entries'][0]['prev'] == REG.header_digest(LOG)
    assert LOG['head'] == REG.entry_digest(LOG['entries'][-1])
    log = copy.deepcopy(LOG)
    log['entries'][-1]['actor'] = 'forged'
    assert any('head' in e for e in REG.validate_lifecycle(log))
    for key, value in (('state_mapping', {}), ('selection', 'elsewhere.json'),
                       ('config', 'configs/other.json'), ('transitions', LOG['transitions'][:-1])):
        log = copy.deepcopy(LOG)
        log[key] = value
        errs = REG.validate_lifecycle(log)
        assert any('header' in e for e in errs) and len(errs) >= 2, (key, errs)
    log = copy.deepcopy(LOG)
    log['transitions'] = LOG['transitions'] + [['X']]
    assert REG.validate_lifecycle(log)  # malformed row is an error, not a crash


def test_rechained_rewrite_still_needs_new_head():
    """Re-chaining a rewritten history needs head rewritten too; only git history can
    catch a full forgery, which test_log_is_append_only_across_history does."""
    entries = copy.deepcopy(LOG['entries'])
    entries[7]['actor'] = 'agent, not owner'
    log = copy.deepcopy(LOG)
    log['entries'] = chain(entries)
    assert any('head' in e for e in REG.validate_lifecycle(log))


def test_tampering_with_history_breaks_the_chain():
    log = copy.deepcopy(LOG)
    log['entries'][2]['actor'] = 'someone else'
    assert any('prev digest' in e for e in REG.validate_lifecycle(log))
    log = copy.deepcopy(LOG)
    log['entries'][3], log['entries'][4] = log['entries'][4], log['entries'][3]
    assert REG.validate_lifecycle(log)
    log = copy.deepcopy(LOG)
    del log['entries'][5]
    assert REG.validate_lifecycle(log)


@pytest.mark.parametrize('mutate', [
    lambda e: e.update(at='2026-11-01T00:00:00Z'),          # time goes backwards
    lambda e: e.update(evidence=[]),                         # no evidence
    lambda e: e.update(evidence=[{'path': 'x'}]),             # evidence without sha256
    lambda e: e.update(evidence=[{'path': 'x', 'sha256': '0' * 64}]),  # without commit
    lambda e: e.update(evidence=[{'path': 'x', 'sha256': '0' * 64, 'commit': 'HEAD'}]),
    lambda e: e.update(actor=''),
    lambda e: e.update(to='PROMOTED_ANYWAY'),
    lambda e: e.update(extra_field=1),
    lambda e: e.update(at='2026-13-99T99:99:99Z'),            # impossible date
    lambda e: e.update(subject=['x']),                        # crashed the validator
    lambda e: e.update(subject=''),
    lambda e: e.update(replaced_by='someone'),                # only valid on replace
])
def test_malformed_appended_entry_is_rejected(mutate):
    e = step('microglia-empf-countshape-v1', 'UNDER_CONFIRMATION', 'pass', 'CONFIRMED_QUALIFIED')
    mutate(e)
    assert REG.validate_lifecycle(with_entries(LOG['entries'] + [e]))


def test_float_seq_is_rejected():
    log = with_entries(LOG['entries'] + [step('microglia-empf-countshape-v1', 'UNDER_CONFIRMATION',
                                              'pass', 'CONFIRMED_QUALIFIED')])
    log['entries'][-1]['seq'] = float(log['entries'][-1]['seq'])
    log['head'] = REG.entry_digest(log['entries'][-1])
    assert any('integer' in e for e in REG.validate_lifecycle(log))


EMPF, OLD = 'microglia-empf-countshape-v1', 'microglia-pbulk-countshape-v2'
FAIL_REVERT = [step(EMPF, 'UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
               step(EMPF, 'CONFIRMATION_FAILED', 'revert', 'REVERTED', at='2026-11-21T00:00:01Z')]


def legacy_config():
    c = copy.deepcopy(CONFIG)
    c['name'], c['status'] = OLD, 'development-selected'
    return c


def test_failed_confirmation_can_reinstate_the_previous_default():
    """Adversary (high): REPLACED was a dead end, so restoring 0.9.7 could not be recorded."""
    log = with_entries(LOG['entries'] + FAIL_REVERT + [
        step(OLD, 'REPLACED', 'reinstate', 'PROMOTED', at='2026-11-21T00:00:02Z')])
    assert REG.validate_lifecycle(log) == []
    assert REG.latest_states(log)[OLD] == 'PROMOTED'
    assert REG.check_default_consistency(log, legacy_config(), SELECTION, RECORDS) == []
    # The config must follow the reinstatement.
    assert REG.check_default_consistency(log, CONFIG, SELECTION, RECORDS)
    # Without the reinstate entry nobody holds the default.
    log = with_entries(LOG['entries'] + FAIL_REVERT)
    assert REG.validate_lifecycle(log) == []
    assert REG.check_default_consistency(log, legacy_config(), SELECTION, RECORDS)


@pytest.mark.parametrize('tail', [
    [step(OLD, 'REPLACED', 'reinstate', 'PROMOTED')],                      # replacer still holds
    [step(EMPF, 'UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
     step(OLD, 'REPLACED', 'reinstate', 'PROMOTED', at='2026-11-21T00:00:01Z')],  # not reverted
    FAIL_REVERT + [step(OLD, 'REPLACED', 'reinstate', 'OWNER_SELECTED_EXPERIMENTAL',
                        at='2026-11-21T00:00:02Z')],                        # wrong restored state
])
def test_reinstate_is_guarded(tail):
    assert REG.validate_lifecycle(with_entries(LOG['entries'] + tail))


def test_late_legacy_import_cannot_promote_an_untested_model():
    """Adversary (medium): import_legacy_default after the log started skipped confirmation."""
    tail = FAIL_REVERT + [step('microglia-untested', None, 'import_legacy_default', 'PROMOTED',
                               at='2026-11-21T00:00:02Z')]
    errs = REG.validate_lifecycle(with_entries(LOG['entries'] + tail))
    assert any('import_legacy_default is only allowed' in e for e in errs)
    # Dropping the superseded_records note is reported too.
    log = copy.deepcopy(LOG)
    log['superseded_records'] = []
    assert any('required superseded_records' in e
               for e in REG.check_default_consistency(log, CONFIG, SELECTION, RECORDS))


def test_replace_must_name_and_hand_over_to_its_replacer():
    entries = copy.deepcopy(LOG['entries'])
    del entries[6]['replaced_by']
    assert any('replaced_by' in e for e in REG.validate_lifecycle(with_entries(entries)))
    entries = copy.deepcopy(LOG['entries'])
    entries[6]['replaced_by'] = 'microglia-someone-else'
    assert any('next entry' in e for e in REG.validate_lifecycle(with_entries(entries)))


def test_new_promotion_requires_a_qualified_selection():
    """Adversary (low): PROMOTED left selection.json unchecked."""
    log = with_entries(LOG['entries'] + [
        step(EMPF, 'UNDER_CONFIRMATION', 'pass', 'CONFIRMED_QUALIFIED'),
        step(EMPF, 'CONFIRMED_QUALIFIED', 'promote', 'PROMOTED', at='2026-11-21T00:00:01Z')])
    assert REG.validate_lifecycle(log) == []
    config = copy.deepcopy(CONFIG)
    config['status'] = 'qualified-default'
    config['selection']['population_qualified'] = True
    errs = REG.check_default_consistency(log, config, SELECTION, RECORDS)
    assert any('selection status' in e for e in errs)
    assert any('population_qualified' in e for e in errs)
    selection = dict(SELECTION, status='qualified_default', population_qualified=True,
                     prospective_promotion_rule_passed=True)
    assert REG.check_default_consistency(log, config, selection, RECORDS) == []
    config['status'] = 'development-selected'  # legacy vocabulary is not a new promotion
    assert REG.check_default_consistency(log, config, selection, RECORDS)


def test_declared_future_paths():
    s = 'microglia-empf-countshape-v1'
    ok = [
        [step(s, 'UNDER_CONFIRMATION', 'pass', 'CONFIRMED_QUALIFIED'),
         step(s, 'CONFIRMED_QUALIFIED', 'promote', 'PROMOTED')],
        [step(s, 'UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
         step(s, 'CONFIRMATION_FAILED', 'revert', 'REVERTED')],
        [step(s, 'UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
         step(s, 'CONFIRMATION_FAILED', 'owner_keep', 'OWNER_SELECTED_EXPERIMENTAL',
              flags=['confirmation_failed'])],
    ]
    for tail in ok:
        assert REG.validate_lifecycle(with_entries(LOG['entries'] + tail)) == []
    bad = [
        [step(s, 'UNDER_CONFIRMATION', 'promote', 'PROMOTED')],          # skips confirmation
        [step(s, 'UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
         step(s, 'CONFIRMATION_FAILED', 'owner_keep', 'OWNER_SELECTED_EXPERIMENTAL')],  # unflagged
        [step('microglia-pbulk-countshape-v2', 'REPLACED', 'promote', 'PROMOTED')],    # not reinstate
        [step(s, 'AUDITED', 'owner_override', 'OWNER_SELECTED_EXPERIMENTAL')],  # stale `from`
    ]
    for tail in bad:
        assert REG.validate_lifecycle(with_entries(LOG['entries'] + tail)), tail


def _random_sequence(rnd):
    states = [None] + list(REG.LIFECYCLE_STATES)
    events = sorted({t[1] for t in REG.LIFECYCLE_TRANSITIONS})
    seq, cur = [], None
    for _ in range(rnd.randint(1, 8)):
        if rnd.random() < 0.7:  # mostly follow the graph, sometimes jump anywhere
            options = [t for t in REG.LIFECYCLE_TRANSITIONS if t[0] == cur]
            t = rnd.choice(options) if options else (cur, rnd.choice(events), rnd.choice(states[1:]))
        else:
            t = (cur, rnd.choice(events), rnd.choice(states[1:]))
        seq.append(t)
        cur = t[2]
    return seq


def _check_sequence(seq):
    """Single-subject sequences; replace/reinstate need a second subject, so a
    sequence containing them must be rejected (their cross-subject rules fail)."""
    declared = {(t[0], t[1], t[2]) for t in LOG['transitions']}
    entries = [step('s', f, ev, to, at=f'2026-12-01T00:00:{i:02d}Z',
                    **({'flags': ['confirmation_failed']} if ev == 'owner_keep' else {}))
               for i, (f, ev, to) in enumerate(seq)]
    legal = all(t in declared for t in seq) \
        and not any(ev in ('replace', 'reinstate') for _, ev, _ in seq) \
        and not any(ev == 'import_legacy_default' for _, ev, _ in seq[1:])
    assert (REG.validate_lifecycle(with_entries(entries)) == []) == legal, seq
    return legal


try:
    from hypothesis import given, settings, strategies as st

    @settings(max_examples=300, deadline=None)
    @given(st.randoms(use_true_random=False))
    def test_random_sequences_match_the_declared_table(rnd):
        _check_sequence(_random_sequence(rnd))
except ImportError:  # seeded fallback
    def test_random_sequences_match_the_declared_table():
        rnd = random.Random(20261120)
        legal = sum(_check_sequence(_random_sequence(rnd)) for _ in range(600))
        assert 50 < legal < 550  # both legal and illegal sequences were exercised


# ---------------------------------------------------------------- (b) consistency
def test_latest_state_matches_config_and_selection():
    assert REG.check_default_consistency(LOG, CONFIG, SELECTION, RECORDS) == []
    assert LOG['state_mapping'] == REG.DEFAULT_HOLDING
    assert REG.check_all(ROOT) == []


@pytest.mark.parametrize('target,mutate', [
    ('config', lambda c, s, r: c.update(status='qualified')),
    ('config', lambda c, s, r: c.update(name='microglia-other')),
    ('config', lambda c, s, r: c['selection'].update(population_qualified=True)),
    ('selection', lambda c, s, r: s.update(status='promoted_default')),
    ('selection', lambda c, s, r: s.update(prospective_promotion_rule_passed=True)),
    ('record', lambda c, s, r: r[STATUS_JSON].update(default_changed=True)),
    ('record', lambda c, s, r: r.pop(STATUS_JSON)),
])
def test_mutated_fixture_fails(target, mutate):
    c, s, r = copy.deepcopy(CONFIG), copy.deepcopy(SELECTION), copy.deepcopy(RECORDS)
    mutate(c, s, r)
    assert REG.check_default_consistency(LOG, c, s, r), target


def test_lifecycle_change_without_config_change_fails():
    s = 'microglia-empf-countshape-v1'
    reverted = with_entries(LOG['entries'] + [step(s, 'UNDER_CONFIRMATION', 'fail', 'CONFIRMATION_FAILED'),
                                              step(s, 'CONFIRMATION_FAILED', 'revert', 'REVERTED')])
    assert REG.check_default_consistency(reverted, CONFIG, SELECTION, RECORDS)
    log = copy.deepcopy(LOG)
    log['superseded_records'][0]['superseded_by_seq'] = 0
    assert REG.check_default_consistency(log, CONFIG, SELECTION, RECORDS)


def test_superseded_status_note_is_exact():
    note, = LOG['superseded_records']
    assert note['path'] == STATUS_JSON and note['field'] == 'default_changed'
    assert note['recorded_value'] is False and RECORDS[STATUS_JSON]['default_changed'] is False
    assert note['sha256'] == REG.sha256_file(ROOT/STATUS_JSON)
    assert LOG['entries'][note['superseded_by_seq']]['event'] == 'owner_override'


# ---------------------------------------------------------------- (d) CAL index
def test_cal_index_is_fresh_and_deterministic():
    first, second = REG.render(REG.build_cal_index(ROOT)), REG.render(REG.build_cal_index(ROOT))
    assert first == second == (ROOT/REG.CAL_INDEX).read_text(encoding='utf-8')


def test_cal_index_ignores_listing_order(monkeypatch):
    expected = REG.render(REG.build_cal_index(ROOT))
    real = os.listdir
    monkeypatch.setattr(REG.os, 'listdir', lambda p: list(reversed(real(p))))
    assert REG.render(REG.build_cal_index(ROOT)) == expected


def test_cal_index_covers_every_record_once():
    index = json.loads((ROOT/REG.CAL_INDEX).read_text(encoding='utf-8'))
    on_disk = sorted(f'analysis/{n}' for n in os.listdir(ROOT/'analysis')
                     if n.startswith('CAL_GPU0_V1_') and n != 'CAL_GPU0_V1_INDEX.json')
    indexed = sorted(r['path'] for a in index['attempts'].values() for r in a['records'])
    assert indexed == on_disk and index['record_count'] == len(on_disk) >= 90
    for a in index['attempts'].values():
        for r in a['records']:
            assert r['sha256'] == REG.sha256_file(ROOT/r['path'])


def test_no_live_attempt_lacks_a_terminal_state():
    index = json.loads((ROOT/REG.CAL_INDEX).read_text(encoding='utf-8'))
    assert REG.check_cal_index(index) == []
    live = {k for k, a in index['attempts'].items()
            if any(REG.LIVE_STATUS.search(v) for r in a['records'] for v in r['status'].values())}
    assert live
    for k in live:
        assert index['attempts'][k]['terminal_state'] in REG.TERMINAL_STATES
    stale = {k: index['attempts'][k]['terminal_state'] for k in index['stale_live_attempts']}
    assert stale == STALE_LIVE
    for k in stale:
        a = index['attempts'][k]
        assert a['terminal_evidence'] and all((ROOT/p).exists() for p in a['terminal_evidence'])


def test_resumption_chains():
    index = json.loads((ROOT/REG.CAL_INDEX).read_text(encoding='utf-8'))
    chains = {k: v['chain'] for k, v in index['lineages'].items() if len(v['chain']) > 1}
    assert chains['released_rate_g100d050p050_f0_pg_s0'] == [
        'released_rate_g100d050p050_f0_pg_s0', 'released_rate_g100d050p050_resume1_f0_pg_s0',
        'released_rate_g100d050p050_resume2_f0_pg_s0']
    assert chains['routed_e22_f0_pg_s0'] == ['routed_e22_v1_f0_pg_s0', 'routed_e22_resume1_f0_pg_s0']
    for lineage, v in chains.items():
        assert index['lineages'][lineage]['terminal_state'] == 'RESULT'
        assert index['attempts'][v[-1]]['terminal_state'] == 'RESULT'
        assert [index['attempts'][x]['resumed_by'] for x in v] == v[1:] + [None]


def test_classify_priority():
    assert REG.classify(['native-training-live', 'incomplete-host-RAM-stop']) == \
        ('INCOMPLETE_STOP', ['incomplete-host-RAM-stop'])
    assert REG.classify(['interrupted', 'result'])[0] == 'RESULT'
    assert REG.classify(['turnover-interruption'])[0] == 'INTERRUPTED'
    assert REG.classify(['resources-live']) == ('ABANDONED_UNKNOWN', [])


def test_live_status_matches_whole_words_only():
    """Adversary (low): INACTIVE and DELIVERED were flagged live."""
    for status in ('INACTIVE_DELIVERED_RESULT', 'DELIVERED', 'inactive'):
        assert not REG.LIVE_STATUS.search(status), status
    for status in ('NATIVE_CUDA_TRAINING_LIVE', 'ACTIVE', 'live-run', 'GPU_ACTIVE_STAGE'):
        assert REG.LIVE_STATUS.search(status), status


def test_live_only_attempt_fails_check(tmp_path):
    """Adversary (low): a new attempt with only a -live record passed as ABANDONED_UNKNOWN."""
    root = _fixture_tree(tmp_path)
    (root/'analysis/CAL_GPU0_V1_newjob_v1_f0_pg_s0-native-training-live.json').write_text(
        '{"status": "NATIVE_CUDA_TRAINING_LIVE"}')
    assert REG.main(['--root', str(root)]) == 1
    errs = REG.check_all(root)
    assert any('newjob_v1_f0_pg_s0: live/active record' in e for e in errs)
    (root/'analysis/CAL_GPU0_V1_quiet_v1_f0_pg_s0-result.json').write_text(
        '{"status": "INACTIVE_DELIVERED_RESULT"}')
    REG.main(['--root', str(root)])
    index = json.loads((root/REG.CAL_INDEX).read_text())
    assert index['attempts']['quiet_v1_f0_pg_s0']['live_records'] == []


def test_unmatched_cal_name_is_reported(tmp_path):
    root = _fixture_tree(tmp_path)
    assert REG.main(['--root', str(root)]) == 0
    (root/'analysis/CAL_GPU0_V1_weird-name-result.json').write_text('{}')
    assert REG.main(['--root', str(root), '--check']) == 1
    assert any('CAL_GPU0_V1_weird-name-result.json' in e for e in REG.check_all(root))
    assert REG.cal_unmatched_paths(ROOT) == []


# ---------------------------------------------------------------- bad inputs
@pytest.mark.parametrize('mode', [[], ['--check']])
@pytest.mark.parametrize('damage', ['truncated', 'missing'])
def test_damaged_lifecycle_is_reported_not_raised(tmp_path, capsys, mode, damage):
    root = _fixture_tree(tmp_path)
    path = root/REG.LIFECYCLE
    if damage == 'missing':
        path.unlink()
    else:
        path.write_text(path.read_text()[:500])
    assert REG.main(['--root', str(root)] + mode) == 1
    assert REG.LIFECYCLE in capsys.readouterr().err


def test_registry_with_wrong_current_can_be_regenerated(tmp_path):
    root = _fixture_tree(tmp_path)
    (root/REG.REGISTRY).parent.mkdir(parents=True, exist_ok=True)
    (root/REG.REGISTRY).write_text(json.dumps({'current': 'CONTINUATION.txt'}))
    assert REG.main(['--root', str(root)]) == 0
    assert json.loads((root/REG.REGISTRY).read_text())['current'] == 'STATUS.md'


def test_unbannered_handoff_in_write_mode_reports_the_file(tmp_path, capsys):
    root = _fixture_tree(tmp_path)
    (root/'analysis/x').mkdir(parents=True)
    (root/'analysis/x/TAKEOVER.txt').write_text('CURRENT AUTHORITATIVE: do X\n')
    assert REG.main(['--root', str(root)]) == 1
    assert 'analysis/x/TAKEOVER.txt' in capsys.readouterr().err


def test_binary_handoff_names_the_file(tmp_path, capsys):
    root = _fixture_tree(tmp_path)
    (root/'analysis/HANDOFF').write_bytes(b'\x89PNG\r\n\x1a\n\xff\xfe')
    assert REG.main(['--root', str(root), '--check']) == 1
    err = capsys.readouterr().err
    assert 'analysis/HANDOFF' in err and 'not UTF-8' in err and 'Traceback' not in err


# ---------------------------------------------------------------- atomic writes
def _fixture_tree(tmp_path):
    """A small offline copy: lifecycle inputs plus a few CAL records and handoffs."""
    root = tmp_path/'repo'
    needed = [REG.LIFECYCLE, 'configs/microglia_default.json', LOG['selection'], STATUS_JSON,
              'STATUS.md', 'CONTINUATION.txt',
              'analysis/CAL_GPU0_V1_released_rate_g100d050p050_f0_pg_s0-native-training-live.json',
              'analysis/CAL_GPU0_V1_released_rate_g100d050p050_f0_pg_s0-incomplete-host-RAM-stop.json',
              'analysis/CAL_GPU0_V1_released_rate_g100d050p050_resume2_f0_pg_s0-result.json']
    for rel in needed:
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, root/rel)
    return root


def test_build_on_fixture_tree(tmp_path):
    root = _fixture_tree(tmp_path)
    assert REG.main(['--root', str(root)]) == 0
    assert REG.main(['--root', str(root), '--check']) == 0
    index = json.loads((root/REG.CAL_INDEX).read_text())
    assert index['record_count'] == 3
    assert index['lineages']['released_rate_g100d050p050_f0_pg_s0']['terminal_state'] == 'RESULT'
    (root/'analysis/CAL_GPU0_V1_new_v1_f0_pg_s0-result.json').write_text('{}')
    assert REG.main(['--root', str(root), '--check']) == 1  # stale index is detected


def test_atomic_write_failure_keeps_old_file(tmp_path, monkeypatch):
    target = tmp_path/'out.json'
    target.write_text('old\n')

    def boom(src, dst):
        raise OSError('injected crash before rename')
    monkeypatch.setattr(REG.os, 'replace', boom)
    with pytest.raises(OSError):
        REG.atomic_write(target, 'new\n')
    assert target.read_text() == 'old\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out.json']


def test_killed_writer_leaves_target_intact(tmp_path):
    """A process killed between writing and renaming leaves the old file whole."""
    root = _fixture_tree(tmp_path)
    assert REG.main(['--root', str(root)]) == 0
    before = (root/REG.CAL_INDEX).read_bytes()
    (root/'analysis/CAL_GPU0_V1_extra_v1_f0_pg_s0-result.json').write_text('{}')
    code = ('import importlib.util, os, sys\n'
            f'spec = importlib.util.spec_from_file_location("r", {str(SCRIPT)!r})\n'
            'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
            'm.os.replace = lambda a, b: os._exit(9)\n'
            f'm.main(["--root", {str(root)!r}])\n')
    proc = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True)
    assert proc.returncode == 9
    assert (root/REG.CAL_INDEX).read_bytes() == before
    orphans = [p.name for p in (root/'analysis').iterdir() if p.name.endswith('.tmp')]
    assert orphans and all(p.startswith('.') for p in orphans)
    # The orphan neither enters the index nor blocks the next build.
    assert REG.main(['--root', str(root)]) == 0
    index = json.loads((root/REG.CAL_INDEX).read_text())
    assert index['record_count'] == 4
    assert not any(r['path'].endswith('.tmp') for a in index['attempts'].values() for r in a['records'])


def test_outputs_follow_the_umask(tmp_path):
    """Adversary (low): mkstemp made every generated file 0600."""
    target = tmp_path/'out.json'
    old = os.umask(0o022)
    try:
        REG.atomic_write(target, 'x\n')
    finally:
        os.umask(old)
    assert target.stat().st_mode & 0o777 == 0o644


def test_stale_temp_files_are_removed(tmp_path):
    target = tmp_path/'out.json'
    stale, fresh = tmp_path/'.out.json.abc.tmp', tmp_path/'.out.json.def.tmp'
    stale.write_text('partial')
    fresh.write_text('partial')
    old = os.stat(stale).st_mtime - REG.STALE_TMP_SECONDS - 60
    os.utime(stale, (old, old))
    assert REG.remove_stale_temps(target) == [stale.name]
    assert fresh.exists() and not stale.exists()  # a concurrent writer's temp survives


def test_concurrent_builders_agree(tmp_path):
    root = _fixture_tree(tmp_path)
    code = ('import importlib.util, sys\n'
            f'spec = importlib.util.spec_from_file_location("r", {str(SCRIPT)!r})\n'
            'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
            f'sys.exit(m.main(["--root", {str(root)!r}]))\n')
    procs = [subprocess.Popen([sys.executable, '-B', '-c', code]) for _ in range(6)]
    assert [p.wait(timeout=120) for p in procs] == [0] * 6
    expected = REG.generated(root)
    for rel, text in expected.items():
        assert (root/rel).read_text(encoding='utf-8') == text
    assert not [p for p in (root/'analysis').iterdir() if p.name.endswith('.tmp')]


# ---------------------------------------------------------------- (e) pins
# Directories the running confirmation queue reads. Its source_manifest.json is
# untracked until the queue finishes, so these are pinned by prefix.
PINNED_PREFIXES = ('analysis/coverage_empf_20261117/', 'analysis/coverage_empf_confirm_20261120/',
                   'analysis/improve_20261109/r5/', 'runs/coverage_empf_20261117/',
                   'runs/microglia_empf/')
# Every key of the queue's analysis/coverage_empf_confirm_20261120/source_manifest.json
# (2026-11-20). The adversary found 16 of them outside the old guard.
CONFIRM_QUEUE_INPUTS = (
    'analysis/coverage_empf_20261117/calibration.py', 'analysis/coverage_empf_20261117/run.py',
    'analysis/coverage_empf_confirm_20261120/PROTOCOL.md',
    'analysis/coverage_empf_confirm_20261120/score.py',
    'analysis/coverage_empf_confirm_20261120/analyze.py',
    'analysis/coverage_empf_confirm_20261120/test_confirm.py',
    'analysis/coverage_empf_confirm_20261120/source_manifest.json',
    'analysis/improve_20261109/r5/architecture_score.py', 'analysis/improve_20261109/r5/evaluate.py',
    'analysis/improve_20261109/r5/generate.py', 'analysis/improve_20261109/r5/interval_metrics.py',
    'analysis/improve_20261109/r5/metrics.py', 'analysis/improve_20261109/r5/mode_cellshape.py',
    'configs/microglia_default.json', 'scripts/microglia.py', 'scripts/microglia_runtime.py',
    'analysis/coverage_empf_20261117/completion/audit.json',
    'analysis/coverage_empf_20261117/completion/baseline_raw_score.json',
    'analysis/coverage_empf_20261117/completion/calibration/baseline_half0.json',
    'analysis/coverage_empf_20261117/completion/calibration/baseline_half1.json',
    'analysis/coverage_empf_20261117/completion/calibration/quarter_half0.json',
    'analysis/coverage_empf_20261117/completion/calibration/quarter_half1.json',
    'analysis/coverage_empf_20261117/completion/comparison.json',
    'analysis/coverage_empf_20261117/completion/diagnosis.json',
    'analysis/coverage_empf_20261117/completion/quarter_generated.receipt.json',
    'analysis/coverage_empf_20261117/completion/quarter_raw_score.json',
    'analysis/default_empf_20261117/selected_model.json',
    'runs/coverage_empf_20261117/quarter_generated.npz',
    'runs/coverage_empf_20261117/quarter_generated.receipt.json',
    'runs/microglia_empf/verification/final_generated.npz',
    'runs/microglia_empf/verification/final_generated.receipt.json',
)


def is_pinned(path, pins):
    return path in pins or path.startswith(PINNED_PREFIXES)


def pinned_paths(root=ROOT):
    """Paths pinned by the live runtime and confirmation manifests, plus explicit pins."""
    pins = {'configs/microglia_default.json', 'analysis/default_empf_20261117/selection.json',
            'analysis/default_empf_20261117/selected_model.json',
            'analysis/default_empf_20261117/runtime_source_manifest.json'}
    runtime = json.loads((root/'analysis/default_empf_20261117/runtime_source_manifest.json').read_text())
    pins |= set(runtime['sources'])
    pins.add(runtime['archived_previous_entrypoint']['archive'])
    for rel in ('analysis/empf_comparison_20261117/source_manifest.json',):
        pins |= {k.replace('/ml2/alz/', '', 1) for k in json.loads((root/rel).read_text())}
    for rel in ('analysis/coverage_empf_20261117/source_manifest.json',
                'analysis/coverage_empf_confirm_20261120/source_manifest.json'):
        if (root/rel).exists():
            m = json.loads((root/rel).read_text())
            for key in ('execution_sources', 'inputs', 'analysis_sources'):
                pins |= set(m.get(key, {}))
            pins.add(rel)
    pins |= {p.relative_to(root).as_posix() for p in (root/'analysis/empf_comparison_20261117').rglob('*')
             if p.is_file()}
    pins |= set(CONFIRM_QUEUE_INPUTS)
    return pins


def test_pin_guard_covers_the_confirmation_queue_inputs():
    """Adversary (low): 16 queue inputs were outside the guard because the queue's
    manifest is untracked at the base."""
    pins = pinned_paths()
    assert all(is_pinned(p, pins) for p in CONFIRM_QUEUE_INPUTS)
    manifest = ROOT/'analysis/coverage_empf_confirm_20261120/source_manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        keys = {k for key in ('execution_sources', 'inputs', 'analysis_sources') for k in m.get(key, {})}
        assert {k for k in keys if not is_pinned(k, pins)} == set()
    assert is_pinned('analysis/coverage_empf_confirm_20261120/new_helper.py', pins)


def test_live_runtime_manifest_pins_hold():
    runtime = json.loads((ROOT/'analysis/default_empf_20261117/runtime_source_manifest.json').read_text())
    present = {p: h for p, h in runtime['sources'].items() if (ROOT/p).exists()}
    assert 'configs/microglia_default.json' in present
    assert {p for p, h in present.items() if REG.sha256_file(ROOT/p) != h} == set()


def test_registry_files_are_not_pinned():
    ours = {REG.REGISTRY, REG.CAL_INDEX, REG.LIFECYCLE, 'scripts/hygiene/build_record_registry.py',
            'tests/test_repo_docs.py', 'tests/test_record_registry.py'} | set(BANNERED_HERE)
    pins = pinned_paths()
    assert not {p for p in ours if is_pinned(p, pins)}


def test_branch_touches_no_pinned_path_while_under_confirmation():
    """While the default is UNDER_CONFIRMATION the confirmation queue hash-checks
    pinned sources, so a branch must not change them. Set RECORD_PIN_BASE to
    compare against another base."""
    if REG.latest_states(LOG).get(CONFIG['name']) != 'UNDER_CONFIRMATION':
        pytest.skip('pins are frozen only while the default is under confirmation')
    if not in_git():
        pytest.skip('not a git checkout')
    base = os.environ.get('RECORD_PIN_BASE', 'origin/research/calibration-optimization')
    try:
        merge_base = git('merge-base', base, 'HEAD').decode().strip()
    except subprocess.CalledProcessError:
        pytest.skip(f'base {base} not available')
    changed = set(git('diff', '--name-only', merge_base, 'HEAD').decode().split())
    changed |= set(git('diff', '--name-only', 'HEAD').decode().split())
    pins = pinned_paths()
    touched = sorted(p for p in changed if is_pinned(p, pins))
    assert not touched, touched


def test_orphan_temp_files_are_gitignored(tmp_path):
    """Adversary (low): an orphan temp file showed up in git status."""
    if not in_git():
        pytest.skip('not a git checkout')
    for rel in (REG.REGISTRY, REG.CAL_INDEX):
        d, name = rel.rsplit('/', 1)
        probe = f'{d}/.{name}.k3j2_x.tmp'
        assert subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', '--no-index', probe]).returncode == 0
