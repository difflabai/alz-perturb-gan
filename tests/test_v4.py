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
