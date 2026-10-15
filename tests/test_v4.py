import dataclasses, hashlib, json, os
from pathlib import Path
import pytest, torch

from alz_perturb_gan import paths
from alz_perturb_gan.paths import scoped, scoped_glob, create_new, ScopeError
from alz_perturb_gan.gene_protocol import StudyProtocol, StudyData
from alz_perturb_gan.gene_models import ModelConfig, build_models
from alz_perturb_gan.gene_evaluation import evaluate
from alz_perturb_gan.gene_calibration import crossfit_report, cap_hits, near_uniform, T_LO, T_HI
from alz_perturb_gan import gene_analysis


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / 'scope'; r.mkdir(); monkeypatch.setenv('ALZ_SCOPE_ROOT', str(r)); return r


# ---- scoped() containment -------------------------------------------------------------------
def test_scoped_refuses_traversal(root):
    (root / 'a').mkdir()
    with pytest.raises(ScopeError): scoped(str(root / 'a' / '..' / '..'))
    assert scoped(str(root / 'a' / '..' / 'a')) == (root / 'a').resolve()

def test_scoped_refuses_prefix_sibling(root):
    sib = Path(str(root) + '-evil'); sib.mkdir()
    with pytest.raises(ScopeError): scoped(str(sib))

def test_scoped_refuses_symlink_escape(root, tmp_path):
    out = tmp_path / 'outside'; out.mkdir(); (root / 'link').symlink_to(out)
    with pytest.raises(ScopeError): scoped(str(root / 'link'))
    with pytest.raises(ScopeError): scoped(str(root / 'link' / 'new'), must_exist=False)
    (root / 'flink').symlink_to(root / 'x')
    with pytest.raises(ScopeError): scoped(str(root / 'flink'), must_exist=False)

@pytest.mark.parametrize('bad', ['', '   ', None, '~', '~/x', '~mikkel', 'relative/path', '.'])
def test_scoped_refuses_tilde_empty_relative(root, bad):
    with pytest.raises(ScopeError): scoped(bad)

def test_scoped_glob_refuses_empty_and_unscoped(root):
    with pytest.raises(ScopeError): scoped_glob(str(root / 'nothing*'))
    with pytest.raises(ScopeError): scoped_glob('')
    with pytest.raises(ScopeError): scoped_glob('*')
    (root / 'f1').write_text('x'); assert scoped_glob(str(root / 'f*')) == [(root / 'f1').resolve()]

def test_allowlisted_exact_paths_only(root):
    assert str(scoped('/proc/locks')) == '/proc/locks'
    with pytest.raises(ScopeError): scoped('/proc/self/environ')
    with pytest.raises(ScopeError): scoped('/tmp/ember-pg-20261012-slots/other.lock')

def test_create_new_is_exclusive_and_nofollow(root, tmp_path):
    p = create_new(str(root), 'r.json', b'{}'); assert p.read_bytes() == b'{}' and (p.stat().st_mode & 0o777) == 0o444
    with pytest.raises(FileExistsError): create_new(str(root), 'r.json', b'x')
    (root / 'dangling').symlink_to(tmp_path / 'target')
    with pytest.raises(FileExistsError): create_new(str(root), 'dangling', b'x')
    assert not (tmp_path / 'target').exists()
    for bad in ('', '..', 'a/b'):
        with pytest.raises(ScopeError): create_new(str(root), bad, b'x')
    with pytest.raises(ScopeError): create_new(str(tmp_path), 'escape.json', b'x')


# ---- calibration group --------------------------------------------------------------------
NAMES = tuple('ABCDEFGHIJKL')

def cal_study():
    ids = tuple(f'ENSG_T_{n}' for n in NAMES)
    p = StudyProtocol(NAMES, ids, ('A', 'B', 'C'), ('H', 'I'), ('J', 'K'), final=('L',), calibration=('D', 'E', 'F', 'G'),
                      draws=3, data_kind='synthetic')
    g = torch.Generator().manual_seed(3)
    cells = {n: torch.randn(8, 4, generator=g) + i * .2 for i, n in enumerate(NAMES)}
    return StudyData.partition(p, torch.ones(4), cells, {n: tuple(f'{n}-{i}' for i in range(8)) for n in NAMES},
                               source_provenance={'data_sha256': 'synthetic'})

def test_calibration_ids_excluded_from_training():
    s = cal_study(); t = s.training_tensors()
    used = {NAMES[i] for i in t['ids'].unique().tolist()}
    assert used == {'A', 'B', 'C'} and not used & set(s.protocol.calibration)
    import copy
    leaky = copy.copy(s.protocol); object.__setattr__(leaky, 'fit', s.protocol.fit + ('D',))  # bypass construction checks
    with pytest.raises(ValueError): dataclasses.replace(s, protocol=leaky).training_tensors()

def test_calibration_must_partition():
    with pytest.raises(ValueError):
        StudyProtocol(NAMES, tuple(NAMES), ('A', 'B', 'C', 'D'), ('H', 'I'), ('J', 'K'), final=('L',), calibration=('D', 'E', 'F', 'G'))

def test_empty_calibration_keeps_historical_fingerprint():
    a = StudyProtocol(tuple('ABCDEF'), tuple('abcdef'), ('A', 'B', 'C'), ('D',), ('E', 'F'))
    d = dataclasses.asdict(a); d.pop('calibration')
    from alz_perturb_gan.gene_features import json_hash
    assert a.fingerprint == json_hash(d)


# ---- crossfit flags -----------------------------------------------------------------------
def test_crossfit_flags_uniform_and_cap():
    logits = torch.zeros(16, 50, dtype=torch.float64); pos = list(range(16))
    rep, final = crossfit_report(logits, pos, [f'n{i}' for i in range(16)])
    assert rep['status'] != 'informative' and rep['final_all16']['near_uniform']
    assert len(rep['per_fold']) == 4 and sorted(sum((p['held_out'] for p in rep['per_fold']), [])) == sorted(f'n{i}' for i in range(16))
    assert cap_hits(T_HI)['upper'] and cap_hits(T_LO)['lower'] and not any(cap_hits(1.0).values())
    assert rep['thresholds']['cap_hit_rule'] == 'T <= 1.05*lo or T >= 0.5*hi' and rep['thresholds']['near_uniform_spread'] == 0.05

def test_crossfit_informative_when_signal():
    g = torch.Generator().manual_seed(0); logits = torch.randn(16, 50, generator=g, dtype=torch.float64)
    pos = list(range(16)); logits[torch.arange(16), pos] += 2.5
    rep, _ = crossfit_report(logits, pos, [f'n{i}' for i in range(16)])
    assert rep['status'] == 'informative', rep['checks']


# ---- sealing, raw-logit scoring and the explore refusal -----------------------------------
def test_evaluate_seals_exploratory_and_scores_raw_logits(tmp_path):
    s = cal_study(); m = build_models(ModelConfig(4, 12, hidden=8, z_dim=2, particles=8, seed=1, steps=1, batch_size=4))
    out = tmp_path / 'ev'; r = evaluate(m, s, output=out)
    summ = json.loads((out / 'metrics/summary.json').read_text())
    assert summ['splits']['exploratory'].get('sealed') is True and 'head' not in summ['splits']['exploratory']
    assert summ['scoring']['mode'] == 'raw_logit_ranks'
    assert set(summ['calibration']['calibrator']['calibration_identities']) == {'D', 'E', 'F', 'G'}
    assert 'status' in summ['calibration']['report'] and 'per_fold' in summ['calibration']['report']
    for f, h in summ['sealed']['exploratory'].items():
        assert hashlib.sha256((out / 'sealed' / f).read_bytes()).hexdigest() == h
    assert not any('"split": "exploratory"' in l for l in (out / 'metrics').glob('*.jsonl') for l in l.read_text().splitlines())
    with pytest.raises(FileExistsError): (out / 'sealed').joinpath('exploratory.json').open('x')

def test_explore_refuses_without_matching_hash_or_ledger(root):
    sel = root / 'sel.json'; sel.write_text('{"bindings": {}}')
    with pytest.raises(PermissionError): gene_analysis.explore(str(sel), '', str(root))
    with pytest.raises(PermissionError): gene_analysis.explore(str(sel), '0' * 64, str(root))
    digest = hashlib.sha256(sel.read_bytes()).hexdigest()
    (root / 'ledger.jsonl').write_text(json.dumps({'event': 'select', 'sha256': 'f' * 64}) + '\n')
    with pytest.raises(PermissionError): gene_analysis.explore(str(sel), digest, str(root))

def test_explore_refuses_outside_scope(tmp_path, root):
    sel = tmp_path / 'sel.json'; sel.write_text('{}')
    with pytest.raises(PermissionError): gene_analysis.explore(str(sel), hashlib.sha256(b'{}').hexdigest(), str(root))

def test_rule_has_stable_hash():
    assert gene_analysis.RULE_SHA256 == hashlib.sha256(gene_analysis.RULE.encode()).hexdigest()
    assert 'G_frozen' not in gene_analysis.CANDIDATES


# ---- in-memory withholding and output scoping ------------------------------------------------
def test_evaluate_returns_no_exploratory_metrics_in_memory(tmp_path):
    s = cal_study(); m = build_models(ModelConfig(4, 12, hidden=8, z_dim=2, particles=8, seed=1, steps=1, batch_size=4))
    for out in (None, tmp_path / 'ev'):
        r = evaluate(m, s, output=out)
        ex = r['summary']['splits']['exploratory']
        assert ex['withheld'] and set(ex) == {'sealed', 'withheld', 'target_tensor_sha256', 'sealed_sha256'}
        assert 'exploratory' not in r['pairs']
        assert ex['sealed'] == (out is not None) and bool(ex['sealed_sha256']) == (out is not None)

def test_evaluate_output_must_be_in_scope(tmp_path, monkeypatch):
    s = cal_study(); m = build_models(ModelConfig(4, 12, hidden=8, z_dim=2, particles=8, seed=1, steps=1, batch_size=4))
    inner = tmp_path / 'scope'; inner.mkdir(); monkeypatch.setenv('ALZ_SCOPE_ROOT', str(inner))
    with pytest.raises(ScopeError): evaluate(m, s, output=tmp_path / 'outside')
    (inner / 'esc').symlink_to(tmp_path)
    with pytest.raises(ScopeError): evaluate(m, s, output=inner / 'esc' / 'new')
    assert not (tmp_path / 'outside').exists() and not (tmp_path / 'new').exists()
    r = evaluate(m, s, output=inner / 'a' / 'b')
    assert (inner / 'a' / 'b' / 'metrics' / 'summary.json').stat().st_mode & 0o777 == 0o444


# ---- end-to-end select / explore on synthetic runs ------------------------------------------
BUNDLE = 'synthetic-bundle'

def _synthetic_cohort(base):
    """3 seeds x 6 arms of REAL (tiny) trained checkpoints, each bound to its training.json and evaluation."""
    from alz_perturb_gan.gene_training import train_arm
    from alz_perturb_gan.gene_checkpoint import save_checkpoint
    s = cal_study(); base.mkdir(parents=True); runs, rruns = {}, {}
    steps = 2000
    for seed in (42, 43, 44):
        run = base / f'seed{seed}'; run.mkdir()
        (run / 'grant.json').write_text(json.dumps({'bundle_sha256': BUNDLE, 'protocol_sha256': s.protocol.fingerprint,
                                                    'seeds': [seed], 'harness_commit': 'c' * 40, 'max_steps_per_arm': steps}))
        m = build_models(ModelConfig(4, 12, hidden=8, z_dim=2, particles=8, seed=seed, steps=2, batch_size=4))
        receipt = train_arm(m, s, decision_cells=2, max_wall_seconds=60)
        for arm in gene_analysis.ARMS:
            m.config = dataclasses.replace(m.config, arm=arm, steps=steps)   # synthetic stand-in for a 2000-step arm
            hist = [{'step': i, 'j_loss': 0.5, **({'realness_weight_mean': 0.5} if i in (1000, 2000) else {})} for i in range(1, steps + 1)]
            r = json.loads(json.dumps(dict(receipt, arm=arm, steps=steps, history=hist)))
            (run / arm).mkdir()
            save_checkpoint(run / arm / 'checkpoint.pt', m, s.protocol, training_receipt=r)
            (run / arm / 'training.json').write_text(json.dumps(r))
            evaluate(m, s, output=run / arm / 'evaluation')
        runs[seed] = str(run); rruns[str(seed)] = {'path': str(run), 'grant_sha256': hashlib.sha256((run / 'grant.json').read_bytes()).hexdigest()}
    rec = base / 'supervisor_receipt.json'
    rec.write_text(json.dumps({'schema': 'supervisor_receipt_v4_2', 'rc': 0, 'stop_reason': None, 'seeds': [42, 43, 44],
                               'static_checks': {'harness_commit': 'c' * 40}, 'runs': rruns,
                               'arms': [{'seed': sd, 'arm': a, 'status': 'ok'} for sd in runs for a in gene_analysis.ARMS]}))
    return runs, str(rec), s.protocol.fingerprint

def _sel(runs, rec, out, proto, **kw):
    return gene_analysis.select(runs, rec, str(out), expected_seeds=[42, 43, 44], bundle_sha256=kw.get('bundle', BUNDLE),
                                protocol_sha256=kw.get('protocol', proto))

@pytest.fixture
def cohort(tmp_path):
    runs, rec, proto = _synthetic_cohort(tmp_path / 'cohort'); out = tmp_path / 'analysis'   # select creates it (v4.4)
    return runs, rec, proto, out

def test_select_then_explore_end_to_end(cohort):
    runs, rec, proto, out = cohort
    path, digest, doc = _sel(runs, rec, out, proto)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert doc['split'] == 'inner' and doc['preconditions'] == {'all_seeds_complete': True, 'gate_ok': True}
    assert doc['selected'] == {'G': False, 'H': False, 'GH': False}
    assert doc['wording'] == 'selection criterion unmet; evidence inconclusive'
    b = doc['bindings']
    assert b['bundle_sha256'] == BUNDLE and b['protocol_sha256'] == proto and b['analysis_code_sha256'] and b['harness_commit'] == 'c' * 40
    arm = b['runs']['42']['arms']['G']
    assert {'training_json_sha256', 'provenance_file_hashes_sha256', 'checkpoint_sha256', 'calibrator_sha256'} <= set(arm)
    rp, rd, rep = gene_analysis.explore(str(path), digest, str(out))
    assert rep['header'].startswith('POST-SELECTION, NON-CONFIRMATORY') and set(rep['exploratory_head']) == {'42', '43', '44'}
    assert [json.loads(l)['event'] for l in (out / 'ledger.jsonl').read_text().splitlines()] == ['select', 'explore']

@pytest.mark.parametrize('fault', ['bundle', 'protocol', 'receipt_rc', 'receipt_arm', 'receipt_path', 'grant', 'summary', 'unlisted', 'sealed', 'seed_missing'])
def test_select_refuses_binding_gaps(cohort, fault):
    runs, rec, proto, out = cohort; kw = {}
    R = json.loads(Path(rec).read_text())
    if fault == 'bundle': kw['bundle'] = 'other'
    if fault == 'protocol': kw['protocol'] = 'f' * 64
    if fault == 'receipt_rc': R['rc'] = 3
    if fault == 'receipt_arm': R['arms'][4]['status'] = 'error_rc7'
    if fault == 'receipt_path': R['runs']['43']['path'] = runs[44]
    if fault == 'grant':
        g = Path(runs[42]) / 'grant.json'; g.write_text(g.read_text().replace(BUNDLE, 'other'))
    if fault == 'summary':
        f = Path(runs[43]) / 'H' / 'evaluation' / 'metrics' / 'summary.json'; f.chmod(0o644); f.write_text(f.read_text() + ' ')
    if fault == 'unlisted': (Path(runs[44]) / 'GH' / 'evaluation' / 'metrics' / 'extra.json').write_text('{}')
    if fault == 'sealed':
        f = Path(runs[42]) / 'B0' / 'evaluation' / 'sealed' / 'exploratory.jsonl'; f.chmod(0o644); f.write_text('')
    Path(rec).write_text(json.dumps(R))
    if fault == 'seed_missing':
        with pytest.raises(PermissionError): gene_analysis.select({42: runs[42]}, rec, str(out), expected_seeds=[42, 43, 44], bundle_sha256=BUNDLE, protocol_sha256=proto)
        return
    with pytest.raises(PermissionError): _sel(runs, rec, out, proto, **kw)
    assert not list(out.glob('selection_frozen_*'))

@pytest.mark.parametrize('fault', ['training', 'grant', 'receipt', 'sealed'])
def test_explore_refuses_when_bound_inputs_change(cohort, fault):
    runs, rec, proto, out = cohort
    path, digest, _ = _sel(runs, rec, out, proto)
    if fault == 'training': (Path(runs[44]) / 'G' / 'training.json').write_text('{"history": []}')
    if fault == 'grant': g = Path(runs[43]) / 'grant.json'; g.write_text(g.read_text() + ' ')
    if fault == 'receipt': Path(rec).write_text(Path(rec).read_text() + ' ')
    if fault == 'sealed':
        f = Path(runs[43]) / 'H' / 'evaluation' / 'sealed' / 'exploratory.json'; f.chmod(0o644); f.write_text(f.read_text() + ' ')
    with pytest.raises(PermissionError): gene_analysis.explore(str(path), digest, str(out))

@pytest.mark.parametrize('fault', ['symlink', 'replaced', 'hardlink', 'missing', 'other_dir'])
def test_explore_validates_ledger_reopen(cohort, tmp_path, fault):
    runs, rec, proto, out = cohort
    path, digest, _ = _sel(runs, rec, out, proto)
    led = out / 'ledger.jsonl'; text = led.read_text()
    if fault == 'symlink':
        real = tmp_path / 'elsewhere.jsonl'; real.write_text(text); led.unlink(); led.symlink_to(real)
    if fault == 'replaced': led.rename(tmp_path / 'old-ledger.jsonl'); led.write_text(text)   # same content, new inode (old kept alive)
    if fault == 'hardlink': os.link(led, tmp_path / 'second-link')
    if fault == 'missing': led.unlink()
    target = str(out)
    if fault == 'other_dir': target = str(tmp_path / 'cohort')
    with pytest.raises(PermissionError): gene_analysis.explore(str(path), digest, target)

def test_select_gate_failure_blocks_selection(cohort):
    runs, rec, proto, out = cohort
    monkey = gene_analysis._gate
    gene_analysis._gate = lambda run: (False, {}) if run == runs[44] else monkey(run)
    try:
        _, _, doc = _sel(runs, rec, out, proto)
    finally:
        gene_analysis._gate = monkey
    assert doc['preconditions']['gate_ok'] is False and not any(doc['selected'].values())


# ---- harness: containment before training ---------------------------------------------------
def test_harness_checks_output_containment_before_any_work(tmp_path, monkeypatch):
    from alz_perturb_gan import gene_harness
    inner = tmp_path / 'scope'; inner.mkdir(); monkeypatch.setenv('ALZ_SCOPE_ROOT', str(inner))
    called = []
    monkeypatch.setattr(gene_harness, 'train_arm', lambda *a, **k: called.append(1))
    monkeypatch.setattr(gene_harness, 'load_study_bundle', lambda *a, **k: called.append(2))
    g = inner / 'grant.json'; g.write_text('{}')
    for bad in (tmp_path / 'outside', inner / '..' / 'escape', Path('relative/out')):
        with pytest.raises(ScopeError):
            gene_harness.main(['train', '--bundle', str(g), '--bundle-sha256', '0', '--output', str(bad), '--max-wall-seconds', '5',
                               '--grant', str(g), '--grant-sha256', '0'])
    (inner / 'esc').symlink_to(tmp_path)
    with pytest.raises(ScopeError):
        gene_harness.main(['train', '--bundle', str(g), '--bundle-sha256', '0', '--output', str(inner / 'esc' / 'x'), '--max-wall-seconds', '5',
                           '--grant', str(g), '--grant-sha256', '0'])
    assert not called and not (tmp_path / 'outside').exists() and not (tmp_path / 'x').exists()
