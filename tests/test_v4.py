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
def _synthetic_cohort(base):
    s = cal_study(); runs = {}
    for seed in (42, 43, 44):
        run = base / f'seed{seed}'; run.mkdir(parents=True)
        (run / 'grant.json').write_text(json.dumps({'bundle_sha256': 'synthetic-bundle', 'seeds': [seed]}))
        m = build_models(ModelConfig(4, 12, hidden=8, z_dim=2, particles=8, seed=seed, steps=1, batch_size=4))
        for arm in gene_analysis.ARMS:
            (run / arm).mkdir()
            evaluate(m, s, output=run / arm / 'evaluation')
            (run / arm / 'checkpoint.pt').write_bytes(f'{seed}-{arm}'.encode())
            hist = [{'step': st, 'realness_weight_mean': 0.5} for st in (1000, 2000)]
            (run / arm / 'training.json').write_text(json.dumps({'history': hist}))
        runs[seed] = str(run)
    rec = base / 'supervisor_receipt.json'
    rec.write_text(json.dumps({'arms': [{'seed': sd, 'arm': a, 'status': 'ok'} for sd in runs for a in gene_analysis.ARMS]}))
    return runs, str(rec)

def test_select_then_explore_end_to_end(tmp_path):
    runs, rec = _synthetic_cohort(tmp_path / 'cohort'); out = tmp_path / 'analysis'; out.mkdir()
    path, digest, doc = gene_analysis.select(runs, rec, str(out), expected_seeds=[42, 43, 44])
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert doc['split'] == 'inner' and doc['preconditions'] == {'all_seeds_complete': True, 'gate_ok': True}
    assert doc['selected'] == {'G': False, 'H': False, 'GH': False}       # identical models: zero deltas
    assert doc['wording'] == 'selection criterion unmet; evidence inconclusive'
    assert 'exploratory' not in json.dumps(doc['inner_deltas']) and doc['bindings']['rule_sha256'] == gene_analysis.RULE_SHA256
    assert set(doc['bindings']['runs']['42']['arms']['G']['sealed_sha256']) == {'exploratory.json', 'exploratory.pt', 'exploratory.jsonl'}
    rp, rd, rep = gene_analysis.explore(str(path), digest, str(out))
    assert rep['header'].startswith('POST-SELECTION, NON-CONFIRMATORY') and set(rep['exploratory_head']) == {'42', '43', '44'}
    events = [json.loads(l)['event'] for l in (out / 'ledger.jsonl').read_text().splitlines()]
    assert events == ['select', 'explore']

def test_select_requires_all_seeds_and_explore_detects_tampering(tmp_path):
    runs, rec = _synthetic_cohort(tmp_path / 'cohort'); out = tmp_path / 'analysis'; out.mkdir()
    with pytest.raises(ValueError): gene_analysis.select({42: runs[42]}, rec, str(out), expected_seeds=[42, 43, 44])
    path, digest, _ = gene_analysis.select(runs, rec, str(out), expected_seeds=[42, 43, 44])
    sealed = Path(runs[43]) / 'H' / 'evaluation' / 'sealed' / 'exploratory.json'
    sealed.chmod(0o644); sealed.write_text(sealed.read_text() + ' ')
    with pytest.raises(PermissionError): gene_analysis.explore(str(path), digest, str(out))

def test_select_gate_failure_blocks_selection(tmp_path):
    runs, rec = _synthetic_cohort(tmp_path / 'cohort'); out = tmp_path / 'analysis'; out.mkdir()
    (Path(runs[44]) / 'G' / 'training.json').write_text(json.dumps({'history': [{'step': 1000, 'realness_weight_mean': 0.9}]}))
    _, _, doc = gene_analysis.select(runs, rec, str(out), expected_seeds=[42, 43, 44])
    assert doc['preconditions']['gate_ok'] is False and not any(doc['selected'].values())
