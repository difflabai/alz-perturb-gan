"""Scientific contracts for the new readout and prospective gates."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
import compare as C
import generate_recipe as W
import provenance as PV


def row(profile=1., loss=1., coverage=.6):
    return {'historical':{**{m:profile if m=='pb_rmse' else 1. for m in C.HISTORICAL},
                          'pc_spread_rep':1.},
            'intervals':{split:{'percentile':{g:{'coverage':coverage, 'width':1.,
                         'interval_score':loss, 'below':(1-coverage)/2, 'above':(1-coverage)/2}
                         for g in ('mean','sd','combined')}} for split in ('train','heldout')}}


def rows(folds=(0,1,2), **kw):
    return {(f,r):row(**kw) for f in folds for r in (0,1)}


def verdict(candidate=None, current=None, empirical=None, original=None, folds=(0,1,2), confirmation=False):
    return C.verdict(candidate if candidate is not None else rows(folds,loss=.7,coverage=.7),
                     current if current is not None else rows(folds),
                     empirical if empirical is not None else rows(folds),
                     original if original is not None else rows(folds), folds, confirmation)


def test_radial_midrank_ties_and_uniform_untied_table():
    center=torch.zeros(2)
    z=torch.tensor([[1.,0.],[-1.,0.],[2.,0.],[3.,0.]])
    radial=(z-center).square().sum(-1).sort().values
    assert torch.equal(W.radial_alpha(z,center,radial),torch.tensor([.25,.25,.625,.875]))
    z=torch.arange(1.,9.)[:,None]
    assert torch.equal(W.radial_alpha(z,torch.zeros(1),z[:,0].square()),
                       (torch.arange(8)+.5)/8)


def test_radial_contraction_preserves_bag_genes_and_cell_libraries():
    g=torch.Generator().manual_seed(22)
    mu=torch.rand((8,17,11),generator=g)*10
    alpha=torch.arange(8.)/7
    out=W.contract(mu,alpha)
    torch.testing.assert_close(out.sum(1),mu.sum(1),rtol=2e-6,atol=2e-5)
    torch.testing.assert_close(out.sum(-1),mu.sum(-1),rtol=2e-6,atol=2e-5)
    assert torch.equal(out[0],mu[0])
    assert bool((out>=0).all())


def test_radial_patch_no_rng_consumption_or_parameter_mutation():
    z=torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[3.,0.]])
    prior=SimpleNamespace(z=z.clone(),code_law='table')
    mu=torch.arange(1.,25.).reshape(4,3,2)
    theta=torch.tensor([2.,3.])
    model=SimpleNamespace(parameters_nb=lambda *args:(mu,theta))
    before=torch.get_rng_state().clone()
    info=W.patch_radial(model,prior)
    out,th=model.parameters_nb(z,None,3,None,None)
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(prior.z,z) and th is theta and info['particles']==4
    torch.testing.assert_close(out.sum(1),mu.sum(1))
    with pytest.raises(ValueError):W.patch_radial(model,prior)


def test_radial_law_rotation_invariant():
    z=torch.tensor([[-2.,1.],[0.,1.],[1.,-3.],[4.,2.]])
    center=z.mean(0)
    rotation=torch.tensor([[0.,-1.],[1.,0.]])
    a=W.radial_alpha(z,center,(z-center).square().sum(-1).sort().values)
    zr=z@rotation;cr=zr.mean(0)
    assert torch.equal(a,W.radial_alpha(zr,cr,(zr-cr).square().sum(-1).sort().values))


def test_streams_fixed_and_replicates_distinct():
    assert W.generation_streams(0,3,0)==W.generation_streams(0,3,0)
    assert W.generation_streams(0,3,0)!=W.generation_streams(0,3,1)
    assert W.generation_streams(0,3,0)!=W.generation_streams(1,3,0)
    real=SimpleNamespace(panel='hvg2000',fold=0,donors=['example'])
    a=W.empirical_rng(real,0,0).integers(0,2**32,size=10)
    b=W.empirical_rng(real,0,1).integers(0,2**32,size=10)
    assert not np.array_equal(a,b)
    assert np.array_equal(a,W.empirical_rng(real,0,0).integers(0,2**32,size=10))


def test_paired_noise_includes_empirical_draw_variation():
    candidate=rows(loss=.9)
    empirical=rows()
    for f in range(3):
        empirical[f,0]=row(loss=.1)
        empirical[f,1]=row(loss=1.9)
    stats=C.metrics(candidate,empirical,(0,1,2))['sd/interval_score']
    assert stats['mean_diff']==pytest.approx(-.1)
    assert stats['se_computational_replicate']>.5
    assert not C.improves(stats)


def test_coverage_distance_computed_before_averaging():
    candidate=rows(coverage=.6)
    candidate[0,0]=row(coverage=.7)
    candidate[0,1]=row(coverage=.9)
    reference=rows(coverage=.8)
    metric=C.metrics(candidate,reference,(0,1,2))['sd/coverage_distance']
    assert metric['per_fold'][0]==pytest.approx(.1)


def test_perfect_fixed_improvements_pass_and_self_comparison_fails():
    assert verdict()['passes']
    assert not verdict(candidate=rows())['passes']


def test_missing_pair_is_incomplete():
    candidate=rows(loss=.7,coverage=.7);candidate.pop((2,1))
    assert verdict(candidate=candidate)['status']=='incomplete'


@pytest.mark.parametrize('guard',C.GUARDS)
def test_gene_and_spread_regressions_rejected(guard):
    candidate=rows(loss=.7,coverage=.7)
    for row_ in candidate.values():
        row_['historical']['pc_spread_rep' if guard=='abs_log_spread' else guard]=2.
    assert not verdict(candidate=candidate)['passes']


def test_original_profile_budget_does_not_compound_against_current():
    assert not verdict(candidate=rows(profile=1.008,loss=.7,coverage=.7),
                       current=rows(profile=1.004))['passes']


def test_per_fold_profile_budget_cannot_hide_in_overall_mean():
    candidate=rows(loss=.7,coverage=.7)
    for r in (0,1):candidate[0,r]['historical']['pb_rmse']=1.006
    result=verdict(candidate=candidate)
    assert result['gates']['profile_total_half_percent_budget']
    assert not result['gates']['profile_every_fold_half_percent_budget']


def test_improvement_must_beat_both_comparators():
    assert not verdict(empirical=rows(loss=.5))['passes']


def test_confirmation_repeat_failure_not_hidden_by_other_folds():
    folds=tuple(range(3,9));candidate=rows(folds,loss=.5,coverage=.7)
    for f in (6,7,8):
        for r in (0,1):candidate[f,r]=row(loss=1.1,coverage=.7)
    result=verdict(candidate=candidate,folds=folds,confirmation=True)
    assert not result['passes']
    assert not result['gates']['empirical32/sd/interval_score/repeat2']


def test_nonfinite_metrics_fail_closed():
    broken=row();broken['historical']['pc_spread_rep']=math.inf
    with pytest.raises(ValueError):C.value(broken,'abs_log_spread')


@pytest.mark.parametrize('field,bad', [('fit','/wrong/fit'),('generation_device','cuda'),
                                      ('M',128),('mode','biascal@stagehalf')])
def test_replay_rejects_incompatible_calibration_before_loading_payloads(tmp_path,field,bad):
    fit=tmp_path/'fit'
    receipt={'fit':str(fit),'generation_device':'cpu','M':32,
             'mode':W.MODES_BY_NAME['current32']}
    receipt[field]=bad
    p=tmp_path/'calibration.json';W.write(p,receipt)
    with pytest.raises(ValueError):W.replay(fit,tmp_path/'cache',p,tmp_path/'out.npz',128)


def test_resume_rejects_unpruned_actual_archive(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(W,'MODES',tmp_path/'modes')
    cache=W.RUNS/'caches/hvg2000/fold0/manifest.json';W.write(cache,{'fold':0})
    receipt=W.MODES/'empirical32/fold0/seed0.receipt.json'
    sources={'source':'fixed'}
    W.write(receipt,{'generated_SHA256':'archive-hash','sources':sources,
                    'challenge_recipe':'empirical32','computational_replicate':0,
                    'M':32,'K':256,'generation_device':'cpu','cache_manifest_sha256':W.sha(cache)})
    score=tmp_path/'score.json'
    W.write(score,{'fingerprint':{'generated_sha256':'archive-hash','cache_sha256':W.sha(cache)},
                   'pruned_source_sha256':'archive-hash','challenge':{'recipe':'empirical32','fold':0,
                    'replicate':0,'sources':sources,'receipt_sha256':W.sha(receipt),
                    'K':256,'PC_CELLS':64,'generation_device':'cpu'},'fold':0,'M':32})
    assert W.check_finished(score,'empirical32',0,0,sources)
    (receipt.parent/'seed0.npz').write_bytes(b'unpruned archive')
    with pytest.raises(ValueError):W.check_finished(score,'empirical32',0,0,sources)


def test_comparator_rejects_source_mutation_before_reading_scores(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'ROOT',tmp_path)
    source=tmp_path/'algorithm.py';source.write_text('original')
    W.write(tmp_path/'source_manifest.json',{str(source):W.sha(source)})
    source.write_text('changed')
    with pytest.raises(ValueError):C.load('blind32',(0,1,2))


def test_cache_payload_mutation_under_same_manifest_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(PV,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(PV,'ANCHORS',tmp_path/'anchors.json')
    cache=PV.RUNS/'caches/hvg2000/fold0'
    payloads={}
    for name in PV.CACHE_FILES:
        p=cache/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original payload')
        payloads[name]=PV.sha(p)
    W.write(cache/'manifest.json',{'train_donors':['a'],'heldout_donors':['b'],
                                 'files_SHA256':payloads})
    expected={'manifest_sha256':PV.sha(cache/'manifest.json'),'payload_sha256':payloads,
              'train_donors':['a'],'heldout_donors':['b']}
    W.write(PV.ANCHORS,{'caches':{'0':expected}})
    assert PV.verify_cache(0)==expected
    (cache/'train/counts.npz').write_bytes(b'changed payload, unchanged manifest')
    with pytest.raises(ValueError):PV.verify_cache(0)
