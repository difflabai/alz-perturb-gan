"""Algebra, variance convention and TRAIN-read boundary checks."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose as D
np=D.np


def test_total_trace_decomposes_within_and_between_targets():
    rng=np.random.default_rng(6);y=rng.normal(size=(4,16,100))+np.arange(4)[:,None,None]
    r=D.trace_parts(y)
    assert abs(r['total_population_trace']-r['within_target_trace']-r['between_target_trace'])<1e-10


def test_mixture_center_separation_is_measured_in_mean_and_sd_blocks():
    rng=np.random.default_rng(7);left=rng.normal(size=(4,16,100));right=left.copy()
    right[...,:50]+=2;right[...,50:]+=4
    result=D.geometry(left,right)
    assert abs(result['means']['component_center_trace']-50)<1e-10
    assert abs(result['sds']['component_center_trace']-200)<1e-10
    assert abs(result['combined']['component_center_trace']-250)<1e-10
    for block in result.values():
        within=.5*(block['model']['total_population_trace']+block['empirical']['total_population_trace'])
        assert abs(block['mixture']['total_population_trace']-within-block['component_center_trace'])<1e-10


def test_gene_variance_is_within_bag_and_does_not_add_mixture_center_separation():
    means=np.concatenate((np.ones((2,3,4)),np.full((2,3,4),3.)),axis=1)
    arrays={'mean_log':means,'nonzero':np.full_like(means,.5),'sq_log':means**2+1}
    real=SimpleNamespace(mean_log=np.full((2,4),2.),nonzero=np.full((2,4),.5),log_var=np.full((2,4),256/255))
    vectors,summary=D.gene_errors(arrays,real)
    assert np.max(np.abs(vectors['logvariance']))<1e-12
    assert summary['logvariance']['rmse_gene']<1e-12


def test_heldout_and_other_count_payloads_are_forbidden(tmp_path):
    cache=tmp_path/'caches/hvg2000/fold0';opened=set();guard=D.read_guard(cache,opened)
    guard('open',(str(cache/'train/counts.npz'),'rb',0))
    assert str(cache/'train/counts.npz') in opened
    for path in (cache/'heldout/counts.npz',cache/'heldout/metadata.npz',tmp_path/'external/counts.npz',
                 cache.parent/'fold1/heldout/metadata.npz',cache.parent/'fold1/train/metadata.npz'):
        with pytest.raises(AssertionError):guard('open',(str(path),'rb',0))


def test_missing_predecessor_runtime_links_are_rejected(monkeypatch):
    monkeypatch.setattr(D.B.P,'read',lambda p:{'runtime_sha256':{}})
    with pytest.raises(ValueError,match='exact predecessor runtime links'):D.prerequisite()


def test_train_target_hook_is_restored_after_failure(monkeypatch):
    original=D.B.G.targets
    monkeypatch.setattr(D.B.D.ZC,'train_targets',lambda ctx:['train'])
    with pytest.raises(RuntimeError):
        with D.train_targets_only():
            assert D.B.G.targets(None)==['train'];raise RuntimeError('injected')
    assert D.B.G.targets is original
