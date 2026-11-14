"""Synthetic strict feasibility and artifact-set audit checks."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import amp_common as B
import generate_amplitude as D
import audit_amplitude as A


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'results';modes=tmp_path/'modes';root.mkdir()
    monkeypatch.setattr(B,'ROOT',root);monkeypatch.setattr(B,'MODES',modes)
    donors=['a','b','c','d'];records=[]
    for donor in donors:
        _,rec=D.F.fit_scalar(lambda t:.7-.2*t,lambda t:-.2,1.)
        rec.update(donor=donor,generated_bags=32,generated_cells=32*256,generated_gene_values=32*256*2000)
        records.append(rec)
    _,law=D.F.normalize([r['raw_t'] for r in records])
    law.update(fit_donors=donors,n_train_targets=4,donor_fits=records,
               fitting='concave peak/segment roots;12 derivative and12 response bisections; exact endpoint roots',
               fit_precision='float64 evaluation of fixed float32 fully calibrated NB parameters')
    cache={'manifest_sha256':'c'*64,'train_donors':donors}
    monkeypatch.setattr(B.P,'verify_cache',lambda *a,**kw:cache)
    monkeypatch.setattr(B.A,'cache_metadata',lambda f,c:c)
    monkeypatch.setattr(B.P,'verify_fit',lambda *a,**kw:{'fit':'f'*64})
    receipt=modes/'current32/fold0/seed0.receipt.json';receipt.parent.mkdir(parents=True);receipt.write_text('{}')
    record={'fold':0,'replicate':0,'sources':{},'law':law,'fit_hashes':{'fit':'f'*64},
            'cache_manifest_sha256':'c'*64,'current_receipt_sha256':B.sha(receipt)}
    path=root/'raw_laws/fold0/seed0.json';B.write(path,record)
    status={'recipe':'amplitude32','fold':0,'replicate':0,'raw_law_sha256':B.sha(path)}
    return status,path,record


def test_explicit_infeasibility_requires_full_raw_training_lineage(tmp_path,monkeypatch):
    status,path,record=fixture(tmp_path,monkeypatch)
    assert A.verify_infeasible(status,{})==('amplitude32',0,0)


@pytest.mark.parametrize('mutation',['held','source','calibration','positive','dimension','archive','score','digest'])
def test_infeasibility_proof_cannot_false_pass(tmp_path,monkeypatch,mutation):
    status,path,record=fixture(tmp_path,monkeypatch)
    if mutation=='held':record['law']['fit_donors'][-1]='held'
    elif mutation=='source':record['sources']={'changed':'code'}
    elif mutation=='calibration':record['current_receipt_sha256']='b'*64
    elif mutation=='positive':record['law']['donor_fits'][0]['raw_t']=.1
    elif mutation=='dimension':record['law']['donor_fits'][0]['generated_bags']=16
    elif mutation in ('archive','score'):
        out=(B.MODES if mutation=='archive' else B.ROOT)/'amplitude32/fold0'/('seed0.npz' if mutation=='archive' else 'seed0.json')
        out.parent.mkdir(parents=True);out.write_text('unplanned')
    B.write(path,record)
    status['raw_law_sha256']='a'*64 if mutation=='digest' else B.sha(path)
    with pytest.raises(ValueError):A.verify_infeasible(status,{})


def test_exact_prefix_counts_include_raw_laws_without_allowing_orphans(tmp_path,monkeypatch):
    monkeypatch.setattr(B,'ROOT',tmp_path/'results');monkeypatch.setattr(B,'MODES',tmp_path/'modes')
    jobs=A.ordered_jobs(range(3));required=jobs[:3];failed=jobs[3]
    for name,f,r in required:
        B.write(B.ROOT/name/f'fold{f}/seed{r}.json',{})
        B.write(B.MODES/name/f'fold{f}/seed{r}.receipt.json',{})
    B.write(B.ROOT/'raw_laws/fold0/seed0.json',{})
    A.exact_artifacts(required,[failed])
    orphan=B.MODES/'amplitude32/fold0/seed0.npz';orphan.parent.mkdir(parents=True);orphan.write_text('orphan')
    with pytest.raises(ValueError):A.exact_artifacts(required,[failed])


def test_ordered_queue_prefix_is_fold_then_repeat_then_recipe():
    jobs=A.ordered_jobs(range(3))
    assert len(jobs)==24 and len(set(jobs))==24
    assert jobs[:4]==[(n,0,0) for n in B.RECIPES]
    assert jobs[4:8]==[(n,0,1) for n in B.RECIPES]


def test_comparison_serialization_normalizes_fold_keys_without_allowing_numeric_changes():
    actual={'per_fold':{0:.125,1:.25},'passes':False}
    saved=json.loads(json.dumps(actual))
    assert A.serialized_equal(saved,actual)
    saved['per_fold']['0']+=1e-10
    assert not A.serialized_equal(saved,actual)
