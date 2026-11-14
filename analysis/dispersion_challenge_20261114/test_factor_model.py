"""Mechanism/fidelity checks with synthetic models; no dataset/weight loads."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as B
import factor_model as F

np=B.np;torch=B.torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('log_theta',torch.full((5,),math.log(3)))
    def parameters_nb(self,z,c,k,library,gen=None):
        noise=torch.randn((len(z),k,5),generator=gen)*.2
        probs=torch.softmax(z[:,:5,None].transpose(1,2)+noise,dim=-1)
        return library[:,:,None]*probs,torch.full((1,1,5),3.)
    def rates(self,z,c,k,library,gen=None):
        mu,theta=self.parameters_nb(z,c,k,library,gen)
        return torch.distributions.Gamma(theta,theta/mu).rsample()


class Prior(torch.nn.Module):
    def sample(self,n,generator=None):return torch.randn((n,8),generator=generator),None


def setup(monkeypatch):
    monkeypatch.setattr(B.G,'M',8);monkeypatch.setattr(B.G,'K',64);monkeypatch.setattr(B.G,'PC_CELLS',8)
    monkeypatch.setattr(B.G,'generation_streams',lambda f,j:B.streams(f,j,0))
    monkeypatch.setattr(B.T,'development_order',lambda:{'a':0,'b':1})
    raw=np.zeros((2,7));raw[:,3]=np.array([1,3])/3
    ctx=SimpleNamespace(device='cpu',fold=0,n_genes=5,genes=np.array(list('abcde')),
        mean=torch.zeros(5),sd=torch.ones(5),W=torch.arange(250,dtype=torch.float32).reshape(5,50)/500,
        ids=['a','b'],raw_c=raw,by=[np.arange(3),np.arange(3,6)],
        X=torch.tensor([[1.,0,2,1,0],[1,1,0,0,1],[0,1,1,0,1],
                        [1,1,0,1,1],[1,0,1,0,0],[0,1,0,1,1]]))
    ctx.fitlibs=ctx.X.sum(-1)
    targets=[('a',False,raw[0],np.array([20.,30.,40.])),('b',True,raw[1],np.array([25.,35.]))]
    plans=[(np.repeat(r[None],8,axis=0).astype(np.float32),['snap']*4+['final']*4) for r in raw]
    models={name:(Model(),Prior()) for name in ('snap','final')}
    return ctx,targets,plans,models


def test_bounded_bisection_has_fixed_accuracy_and_step_count():
    function=lambda u:1/(1+math.exp(-u))
    value,info=F.fit_scalar(function,function(.237))
    assert abs(value-.237)<=1/4096
    assert info['bisection_steps']==12 and info['clipped'] is None


@pytest.mark.parametrize('target,expected,side',[(0.,-1.,'low'),(1.,1.,'high')])
def test_unreachable_nonzero_targets_are_clipped(target,expected,side):
    value,info=F.fit_scalar(lambda u:.5+.1*u,target)
    assert value==expected and info['clipped']==side and info['bisection_steps']==0


@pytest.mark.parametrize('bad',[float('nan'),-1.,2.])
def test_invalid_observed_target_is_rejected(bad):
    with pytest.raises(ValueError):F.fit_scalar(lambda x:.5,bad)


def test_normalization_preserves_inverse_precision_not_mean_log():
    raw=np.linspace(-1,1,25);law=F.normalized_factors(raw)
    assert np.max(np.abs(law))<=2
    assert abs(np.exp(-law).mean()-1)<1e-14
    assert np.exp(-(raw-raw.mean())).mean()>1.1
    # The NB excess variance mu^2/theta is conserved over the factor law.
    assert abs(np.mean(7.**2/(3.*np.exp(law)))-7.**2/3.)<1e-12


def test_identical_fitted_offsets_give_literal_zero_law():
    assert np.array_equal(F.normalized_factors([.5]*25),np.zeros(25))


def test_private_factor_stream_is_stable_distinct_and_leaves_other_rng_unchanged(monkeypatch):
    setup(monkeypatch)
    route=F.FactorRoute(np.linspace(-1,1,25),lambda d:B.factor_seed(0,d,0))
    before=torch.random.get_rng_state().clone();np_before=np.random.get_state()
    a=route.indices('a').copy()
    assert a==route.indices('a') and a!=route.indices('b')
    assert a!=F.FactorRoute(np.linspace(-1,1,25),lambda d:B.factor_seed(0,d,1)).indices('a')
    assert torch.equal(before,torch.random.get_rng_state())
    assert np.array_equal(np_before[1],np.random.get_state()[1])


def test_zero_factor_loop_exactly_reproduces_frozen_generation(monkeypatch):
    ctx,targets,plans,models=setup(monkeypatch)
    with torch.no_grad():expected,stats=F.ORIGINAL_GENERATE(ctx,models,plans,T=targets)
    route=F.FactorRoute([0.],lambda d:B.factor_seed(0,d,0))
    copied=copy.deepcopy(models)
    for model,_ in copied.values():route.patch(model)
    with route.installed():actual,actual_stats=B.G.generate_arrays(ctx,copied,plans,T=targets)
    assert B.array_fingerprints(expected)==B.array_fingerprints(actual)
    assert stats==actual_stats
    assert all(model.training and prior.training for model,prior in copied.values())
    assert route.current is None


def test_expected_and_sampled_generation_share_factor_assignments(monkeypatch):
    ctx,targets,plans,models=setup(monkeypatch)
    route=F.FactorRoute([-.7,.4],lambda d:B.factor_seed(0,d,0))
    for model,_ in models.values():
        route.patch(model)
        B.ZC.patch_generator(model,'cpu')
    with route.installed():
        before={}
        for ti,start,n,model,z,c,lib,cell,count in B.ZC.iterate_chunks(ctx,models,plans,targets):
            _,theta=B.ZC.base_nb(model,z,c,B.G.K,lib,cell)
            expected=3*torch.exp(torch.tensor(route.values[np.array(route.indices(targets[ti][0]))[start:start+n]]))
            assert torch.equal(theta[:,0,0],expected)
            before[targets[ti][0]]=route.indices(targets[ti][0]).copy()
        B.G.generate_arrays(ctx,models,plans,T=targets)
        assert before==route.assignments


def test_exception_restores_generation_hooks_and_model_modes(monkeypatch):
    ctx,targets,plans,models=setup(monkeypatch)
    route=F.FactorRoute([0.],lambda d:12)
    for model,_ in models.values():route.patch(model)
    def fail(*a,**kw):raise RuntimeError('injected failure')
    monkeypatch.setattr(models['snap'][0],'rates',fail)
    with pytest.raises(RuntimeError,match='injected'):
        with route.installed():B.G.generate_arrays(ctx,models,plans,T=targets)
    assert B.G.generate_arrays is F.ORIGINAL_GENERATE and B.ZC.iterate_chunks is F.ORIGINAL_ITERATE
    assert all(model.training and prior.training for model,prior in models.values())
    assert route.current is None


def test_factor_fit_uses_only_train_targets_and_fully_calibrated_theta(monkeypatch):
    ctx,targets,plans,models=setup(monkeypatch)
    ctx.heldout_ids=['held_sentinel']
    def heldout_forbidden(*a,**kw):raise AssertionError('requested held-out generation targets during factor fit')
    monkeypatch.setattr(B.G,'targets',heldout_forbidden)
    def forbidden(*a,**kw):raise AssertionError('bypassed current calibration')
    for model,_ in models.values():model._zcal_orig_nb=forbidden
    law,info=F.fit_law(ctx,models,plans)
    assert info['fit_donors']==['a','b'] and info['n_train_targets']==2
    assert 'held_sentinel' not in info['fit_donors']
    assert len(law)==2 and abs(np.exp(-law).mean()-1)<1e-12
