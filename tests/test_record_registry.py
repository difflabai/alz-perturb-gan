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


def chain(entries):
    """Renumber and re-chain entries, as an honest appender would."""
    out, prev = [], None
    for i, e in enumerate(entries):
        e = dict(e, seq=i, prev=prev)
        prev = REG.entry_digest(e)
        out.append(e)
    return out


def with_entries(entries):
    log = copy.deepcopy(LOG)
    log['entries'] = chain(entries)
    return log


def step(subject, frm, event, to, at='2026-11-21T00:00:00Z', **extra):
    e = {'subject': subject, 'from': frm, 'event': event, 'to': to, 'at': at, 'actor': 'test',
         'evidence': [{'path': 'x', 'sha256': '0' * 64}]}
    e.update(extra)
    return e


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
    checked = 0
    for e in LOG['entries']:
        for ev in e['evidence']:
            try:
                blob = git('show', f'{ev["commit"]}:{ev["path"]}')
            except subprocess.CalledProcessError:
                continue  # shallow clone without that commit
            assert hashlib.sha256(blob).hexdigest() == ev['sha256'], (e['seq'], ev['path'])
            checked += 1
    if not checked:
        pytest.skip('evidence commits not in this clone')


def test_log_is_append_only_across_history():
    if not in_git():
        pytest.skip('not a git checkout')
    revs = git('log', '--format=%H', '--', REG.LIFECYCLE).decode().split()
    for rev in revs:
        old = json.loads(git('show', f'{rev}:{REG.LIFECYCLE}'))
        assert LOG['entries'][:len(old['entries'])] == old['entries'], rev


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
    lambda e: e.update(actor=''),
    lambda e: e.update(to='PROMOTED_ANYWAY'),
    lambda e: e.update(extra_field=1),
])
def test_malformed_appended_entry_is_rejected(mutate):
    e = step('microglia-empf-countshape-v1', 'UNDER_CONFIRMATION', 'pass', 'CONFIRMED_QUALIFIED')
    mutate(e)
    assert REG.validate_lifecycle(with_entries(LOG['entries'] + [e]))


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
        [step('microglia-pbulk-countshape-v2', 'REPLACED', 'promote', 'PROMOTED')],    # terminal
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
    declared = {(t[0], t[1], t[2]) for t in LOG['transitions']}
    entries = [step('s', f, ev, to, at=f'2026-12-01T00:00:{i:02d}Z',
                    **({'flags': ['confirmation_failed']} if ev == 'owner_keep' else {}))
               for i, (f, ev, to) in enumerate(seq)]
    legal = all(t in declared for t in seq)
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
    return pins


def test_live_runtime_manifest_pins_hold():
    runtime = json.loads((ROOT/'analysis/default_empf_20261117/runtime_source_manifest.json').read_text())
    present = {p: h for p, h in runtime['sources'].items() if (ROOT/p).exists()}
    assert 'configs/microglia_default.json' in present
    assert {p for p, h in present.items() if REG.sha256_file(ROOT/p) != h} == set()


def test_registry_files_are_not_pinned():
    ours = {REG.REGISTRY, REG.CAL_INDEX, REG.LIFECYCLE, 'scripts/hygiene/build_record_registry.py',
            'tests/test_repo_docs.py', 'tests/test_record_registry.py'} | set(BANNERED_HERE)
    assert not ours & pinned_paths()


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
    assert not changed & pinned_paths(), sorted(changed & pinned_paths())
