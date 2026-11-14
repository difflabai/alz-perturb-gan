"""Prevent partial or extraneous artifact sets from becoming complete proofs."""
from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_hybrid as A


def test_confirmation_is_required_only_for_passing_fixed_screen():
    screen=A.required_set(False);full=A.required_set(True)
    assert len(set(screen))==len(screen)==24
    assert len(set(full))==len(full)==72
    assert set(screen)<set(full)
    assert {f for _,f,_ in full}==set(range(9))


def roots(tmp_path,monkeypatch):
    results=tmp_path/'results';modes=tmp_path/'modes'
    monkeypatch.setattr(A.H,'ROOT',results);monkeypatch.setattr(A.H,'MODES',modes)
    results.mkdir();modes.mkdir()
    return results,modes


@pytest.mark.parametrize('orphan',['seed0.model.npz','seed0.empirical.npz','seed1.receipt.json'])
def test_orphan_components_or_receipts_reject_complete_proof(tmp_path,monkeypatch,orphan):
    results,modes=roots(tmp_path,monkeypatch)
    score=results/'hybrid32/fold0/seed0.json';score.parent.mkdir(parents=True);score.write_text('{}')
    receipt=modes/'hybrid32/fold0/seed0.receipt.json';receipt.parent.mkdir(parents=True);receipt.write_text('{}')
    required=[('hybrid32',0,0)]
    A.exact_artifacts(required)
    (receipt.parent/orphan).write_text('{}')
    with pytest.raises(ValueError):A.exact_artifacts(required)


def test_live_scorer_or_unpruned_archive_is_not_ready(tmp_path,monkeypatch):
    results,modes=roots(tmp_path,monkeypatch)
    score=results/'hybrid32/fold0/seed0.json';score.parent.mkdir(parents=True)
    score.write_text('{}')
    assert not A.ready('hybrid32',0,0)
    score.write_text(json.dumps({'challenge':{'recipe':'hybrid32'},'pruned_source_sha256':'hash'}))
    assert A.ready('hybrid32',0,0)
    archive=modes/'hybrid32/fold0/seed0.npz';archive.parent.mkdir(parents=True);archive.write_bytes(b'archive')
    assert not A.ready('hybrid32',0,0)


def test_last_screen_row_finishing_during_audit_requires_next_eligibility_pass(tmp_path,monkeypatch):
    results,modes=roots(tmp_path,monkeypatch)
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(A.H,'HERE',source)
    manifest=source/'source_manifest_prelaunch.json';manifest.write_text('{}')
    (results/'source_manifest.json').write_text('{}')
    monkeypatch.setattr(A,'PRELAUNCH_SHA256',A.H.sha(manifest))
    monkeypatch.setattr(A.H,'source_manifest',lambda:{})
    missing=('hybrid32',2,1)
    def create(key):
        name,f,r=key;path=results/name/f'fold{f}/seed{r}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{"challenge":{"recipe":"fixed"},"pruned_source_sha256":"hash"}')
        receipt=modes/name/f'fold{f}/seed{r}.receipt.json';receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps({role+'_component':{'generated_SHA256':'a'*64}
                                      for role in ('model','empirical')}))
    for key in A.required_set(False):
        if key!=missing:create(key)
    def cache(fold):create(missing);return {}
    monkeypatch.setattr(A.H.P,'verify_cache',cache)
    monkeypatch.setattr(A.H.P,'verify_fit',lambda *a:{})
    monkeypatch.setattr(A.H,'finalized',lambda *a:True)
    def no_report(*a,**kw):raise AssertionError('initial screen was incomplete')
    monkeypatch.setattr(A.C,'report',no_report)
    out=A.audit()
    assert not out['complete'] and out['verified_artifacts']==23 and out['required_artifacts']==24
    assert out['screen_passes'] is None and out['confirmation_disposition']=='eligibility_incomplete'
    assert out['missing']==[{'recipe':'hybrid32','fold':2,'replicate':1}]
