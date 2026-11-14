"""Synthetic numerical, routing and provenance checks; no study payloads."""
import copy
import json
import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import amp_common as B
import amplitude_model as F
import generate_amplitude as D
import test_factor_model as OLD
np=B.np;torch=B.torch


@pytest.mark.parametrize('function,derivative,target,expected',[
    (lambda t:.7-.2*t,lambda t:-.2,.7,0.),
    (lambda t:.3+.2*t,lambda t:.2,.5,1.),
    (lambda t:.7-.2*t,lambda t:-.2,.2,1.),
    (lambda t:.5,lambda t:0.,.9,.5)])
def test_exact_endpoints_clipping_and_degeneracy(function,derivative,target,expected):
    value,rec=F.fit_scalar(function,derivative,target)
    assert value==expected
    D.validate_fit(rec)


def test_two_roots_choose_closest_to_half_and_retain_peak_resolution():
    f=lambda t:.8-.4*(t-.4)**2
    t,rec=F.fit_scalar(f,lambda t:-.8*(t-.4),.78)
    assert len(rec['roots'])==2 and abs(t-(.4+math.sqrt(.05)))<1/4096
    assert rec['peak_resolution']==2**-12 and rec['peak_bisection_steps']==12
    assert all(root['bisection_steps']==12 for root in rec['roots'])
    D.validate_fit(rec)


def test_exact_zero_endpoint_does_not_become_numerical_positive():
    f=lambda t:.8-.2*(t-.4)**2
    t,rec=F.fit_scalar(f,lambda t:-.4*(t-.4),f(0.))
    assert rec['roots'][0]['t']==0 and rec['roots'][0]['bracket']==[0.,0.]
    assert rec['roots'][0]['bisection_steps']==0
    D.validate_fit(rec)


@pytest.mark.parametrize('values',[[0,0,0,0],[0,0,0,.3,0],[0,0,0,.3],[.1,.1,.1,.1],
                                  [0,.02,.2,.8,1],[.001,.001,.001,.001,.7,.9]])
def test_active_set_normalization_or_explicit_infeasibility(values):
    law,info=F.normalize(values)
    if np.count_nonzero(values)/len(values)<.25:
        assert law is None and info['feasible'] is False
    else:
        assert info['feasible'] and abs(np.square(law).mean()-.25)<1e-12
        assert np.array_equal(law,np.minimum(1,info['c']*np.asarray(values)))
        assert np.all(law[np.asarray(values)==0]==0)
        assert abs(info['second_moment_float32']-.25)<1e-6


@pytest.mark.parametrize('values',[[],[-.1],[1.1],[float('nan')]])
def test_invalid_raw_law_rejected(values):
    with pytest.raises(ValueError):F.normalize(values)


def test_active_set_index_and_actual_saturated_count_are_distinct_at_equality():
    law,info=F.normalize([1,.5,.5,.5,0,0,0])
    assert info['c']==1 and info['active_set_k']==0 and info['saturated_count']==1


def test_normalization_conserves_composition_covariance_but_not_variable_library_count_covariance():
    raw=np.array([[2.,8.],[12.,8.],[2.,3.]])
    lib=raw.sum(-1,keepdims=True);base=lib*raw.sum(0,keepdims=True)/lib.sum()
    law,info=F.normalize([0.,.8]);assert abs(np.square(law).mean()-.25)<1e-12
    fixed=.5*raw+.5*base
    varied=np.stack([(1-t)*base+t*raw for t in law])
    assert np.allclose(np.mean(np.var(varied/lib,axis=1),axis=0),np.var(fixed/lib,axis=0))
    assert not np.allclose(np.mean(np.var(varied,axis=1),axis=0),np.var(fixed,axis=0))


def test_nb_response_derivative_and_concavity_with_variable_libraries():
    raw=torch.tensor([[[2.,8.],[12.,8.],[2.,3.]]],dtype=torch.float64)
    library=raw.sum(-1,keepdim=True);base=library*raw.sum(1,keepdim=True)/library.sum(1,keepdim=True)
    theta=torch.tensor([[[.7,4.]]],dtype=torch.float64)
    f,d=F.cached_response([(raw,theta,base)])
    grid=np.linspace(0,1,21);derivatives=np.array([d(t) for t in grid])
    assert np.all(np.diff(derivatives)<=1e-14)
    for t in (.1,.5,.9):assert abs((f(t+1e-5)-f(t-1e-5))/2e-5-d(t))<1e-10


def test_actual_nb_variable_library_response_has_two_roots():
    raw=torch.tensor([[[9.,1.],[10.,90.]]],dtype=torch.float64)
    lib=raw.sum(-1,keepdim=True);base=lib*raw.sum(1,keepdim=True)/lib.sum(1,keepdim=True)
    f,d=F.cached_response([(raw,torch.full((1,1,2),.7,dtype=torch.float64),base)])
    assert d(0)>0 and d(1)<0
    _,peak=F.fit_scalar(f,d,1.)
    target=(max(f(0),f(1))+peak['F_peak'])/2
    value,rec=F.fit_scalar(f,d,target)
    assert len(rec['roots'])==2
    assert all(abs(f(root['t'])-target)<1e-4 for root in rec['roots'])
    D.validate_fit(rec)


def test_constant_half_full_synthetic_count_loop_matches_frozen_cs(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch)
    reference=copy.deepcopy(models)
    for model,_ in reference.values():B.CS.patch_generator(model,.5)
    with torch.no_grad():expected,stats=F.F.ORIGINAL_GENERATE(ctx,reference,plans,T=targets)
    route=F.AmplitudeRoute([.5],lambda d:B.amplitude_seed(0,d,0))
    for model,_ in models.values():route.patch(model)
    with route.installed():actual,actual_stats=B.G.generate_arrays(ctx,models,plans,T=targets)
    assert B.array_fingerprints(expected)==B.array_fingerprints(actual) and stats==actual_stats
    assert route.current is None and all(m.training and p.training for m,p in models.values())


def test_dynamic_amplitude_preserves_expected_totals_theta_and_private_assignments(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch)
    route=F.AmplitudeRoute([0.,1.],lambda d:B.amplitude_seed(0,d,0))
    before=torch.random.get_rng_state().clone();np_before=np.random.get_state()
    assignments=route.indices('a').copy()
    assert torch.equal(before,torch.random.get_rng_state()) and np.array_equal(np_before[1],np.random.get_state()[1])
    assert assignments!=F.AmplitudeRoute([0.,1.],lambda d:B.amplitude_seed(0,d,1)).indices('a')
    originals={name:m.parameters_nb for name,(m,p) in models.items()}
    for model,_ in models.values():route.patch(model);B.ZC.patch_generator(model,'cpu')
    names={id(m):name for name,(m,p) in models.items()}
    with route.installed():
        for ti,start,n,model,z,c,libs,cell,count in B.ZC.iterate_chunks(ctx,models,plans,targets):
            state=cell.get_state().clone();raw,theta=originals[names[id(model)]](z,c,B.G.K,libs,cell)
            cell.set_state(state);mu,newtheta=B.ZC.base_nb(model,z,c,B.G.K,libs,cell)
            assert torch.equal(theta,newtheta)
            assert torch.allclose(mu.sum(1),raw.sum(1),rtol=1e-6,atol=1e-4)
            assert torch.allclose(mu.sum(-1),libs,rtol=1e-6,atol=1e-4)
        saved=copy.deepcopy(route.assignments)
        B.G.generate_arrays(ctx,models,plans,T=targets)
        assert route.assignments==saved


def test_route_exception_restores_hooks_and_model_modes(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch);route=F.AmplitudeRoute([.5],lambda d:12)
    for model,_ in models.values():route.patch(model)
    def fail(*a,**kw):raise RuntimeError('injected')
    monkeypatch.setattr(models['snap'][0],'rates',fail)
    with pytest.raises(RuntimeError,match='injected'):
        with route.installed():B.G.generate_arrays(ctx,models,plans,T=targets)
    assert B.G.generate_arrays is F.F.ORIGINAL_GENERATE and B.ZC.iterate_chunks is F.F.ORIGINAL_ITERATE
    assert route.current is None and all(m.training and p.training for m,p in models.values())


def test_fit_only_train_uses_full_current_theta_and_excludes_heldout(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch);ctx.heldout_ids=['held_sentinel']
    def forbidden(*a,**kw):raise AssertionError('heldout target or outer-theta bypass')
    monkeypatch.setattr(B.G,'targets',forbidden)
    for model,_ in models.values():
        B.ZC.patch_generator(model,'cpu')['s'].fill_(.4)
        model._zcal_orig_nb=forbidden
    original=F.cached_response;seen=[]
    def response(cached):
        seen.extend([theta.clone() for mu,theta,base in cached]);return original(cached)
    monkeypatch.setattr(F,'cached_response',response)
    law,info=F.fit_law(ctx,models,plans)
    assert info['fit_donors']==['a','b'] and info['n_train_targets']==2
    expected=float((torch.tensor(3.)*torch.exp(torch.tensor(.4))).double())
    assert seen and all(bool((theta==expected).all()) for theta in seen)
    for rec in info['donor_fits']:D.validate_fit(rec)


@pytest.mark.parametrize('mutation',['root','endpoint','steps','resolution','peak','degenerate','chosen'])
def test_fitting_metadata_cannot_false_pass(mutation):
    f=lambda t:.7-.2*t
    t,rec=F.fit_scalar(f,lambda t:-.2,.7)
    if mutation=='root':rec['roots']=[]
    elif mutation=='endpoint':rec['roots'][0]['t']=1e-9
    elif mutation=='steps':rec['roots'][0]['bisection_steps']=12
    elif mutation=='resolution':rec['peak_resolution']=1e-4
    elif mutation=='peak':rec['peak']=.2
    elif mutation=='degenerate':rec['degenerate']=True
    else:rec['raw_t']=1e-9
    with pytest.raises(ValueError):D.validate_fit(rec)


def control_fixture(tmp_path,monkeypatch):
    root=tmp_path/'results';modes=tmp_path/'modes';root.mkdir()
    monkeypatch.setattr(B,'ROOT',root);monkeypatch.setattr(B,'MODES',modes)
    receipt=modes/'current32/fold0/seed0.receipt.json';receipt.parent.mkdir(parents=True)
    fingerprints={name:'a'*64 for name in (*B.H.ARRAYS,*B.H.IDENTITY)}
    receipt.write_text(json.dumps({'array_fingerprints':fingerprints,
        'mode_extra':{'biascal':{'delta':[.123]},'zcal':{'s':[.456]}}}))
    proof={'passed':True,'fold':0,'replicate':0,'sources':{},'current_receipt_sha256':B.sha(receipt),
           'control_array_fingerprints':fingerprints,'control_bias_delta':[.123],'control_gene_s':[.456],
           'pruned_control_sha256':'b'*64}
    path=root/'constant_amplitude_fidelity.json';path.write_text(json.dumps(proof));return path,proof


def test_full_calibration_fidelity_metadata_accepted(tmp_path,monkeypatch):
    control_fixture(tmp_path,monkeypatch);D.verify_fidelity({})


@pytest.mark.parametrize('mutation',['source','fingerprint','missing_fingerprint','delta','s','digest','receipt','unpruned'])
def test_fidelity_requires_full_arrays_coefficients_provenance_and_pruning(tmp_path,monkeypatch,mutation):
    path,proof=control_fixture(tmp_path,monkeypatch)
    if mutation=='source':proof['sources']={'changed':'code'}
    elif mutation=='fingerprint':proof['control_array_fingerprints']['pcs']='c'*64
    elif mutation=='missing_fingerprint':proof['control_array_fingerprints'].pop('pcs')
    elif mutation=='delta':proof['control_bias_delta']=[.124]
    elif mutation=='s':proof['control_gene_s']=[.457]
    elif mutation=='digest':proof['pruned_control_sha256']='short'
    elif mutation=='receipt':proof['current_receipt_sha256']='c'*64
    else:
        archive=B.MODES/'constant_fidelity/fold0/seed0.npz';archive.parent.mkdir(parents=True);archive.write_bytes(b'unpruned')
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):D.verify_fidelity({})


def test_cannot_rebless_old_195_execution_sources(monkeypatch):
    monkeypatch.setattr(B.Q,'sources',lambda:{})
    with pytest.raises(ValueError,match='prior195'):B.source_manifest()
