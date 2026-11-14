"""Synthetic sampler/derivative/fidelity checks without study payloads."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import joint_common as J
import gamma_head as F
import test_factor_model as OLD
B=J.B;np=J.np;torch=J.torch


def test_group_ties_signs_empty_groups_and_zero_loading_rows():
    w=np.zeros((4,50));w[0,3]=2;w[1,2]=-3;w[2,1]=-2;w[2,4]=2
    assert J.groups(w).tolist()==[6,5,3,0]
    assert len(np.bincount(J.groups(w),minlength=100))==100


def test_literal_zero_head_reproduces_full_synthetic_count_arrays(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch)
    for model,_ in models.values():B.CS.patch_generator(model,.5)
    with torch.no_grad():expected,stats=OLD.F.ORIGINAL_GENERATE(ctx,models,plans,T=targets)
    copied=copy.deepcopy(models)
    for model,_ in copied.values():F.GammaHead(np.arange(5),0.).patch(model)
    with torch.no_grad():actual,actual_stats=OLD.F.ORIGINAL_GENERATE(ctx,copied,plans,T=targets)
    assert B.array_fingerprints(actual)==B.array_fingerprints(expected) and stats==actual_stats


def test_moderate_shape_completed_marginals_count_moments_zeros_and_covariance():
    torch.manual_seed(1337);n=80000
    mu=torch.tensor([3.,5.,9.]).reshape(1,1,3).expand(n,1,3)
    theta=torch.tensor([1.,2.,5.]);head=F.GammaHead([0,0,1],.5)
    rate=head.rates(mu,theta).numpy()[:,0]
    counts=torch.poisson(torch.from_numpy(rate),generator=torch.Generator().manual_seed(19)).numpy()
    m=mu[0,0].numpy().astype(np.float64);t=theta.numpy().astype(np.float64)
    expected_var=m+m*m/t
    assert np.all(np.abs(counts.mean(0)-m)<6*np.sqrt(expected_var/n))
    # Use actual fourth moments for a conservative uncertainty of sample variance.
    centered=counts-counts.mean(0);fourth=(centered**4).mean(0)
    assert np.all(np.abs(counts.var(0)-expected_var)<6*np.sqrt(fourth/n))
    expected_zero=(1+m/t)**(-t);zero=(counts==0).mean(0)
    assert np.all(np.abs(zero-expected_zero)<6*np.sqrt(expected_zero*(1-expected_zero)/n))
    covariance=np.cov(counts.T,ddof=0);expected=.5*m[0]*m[1]*min(t[0],t[1])/(t[0]*t[1])
    assert abs(covariance[0,1]-expected)<6*np.sqrt((expected_var[0]*expected_var[1]+expected**2)/n)
    assert abs(covariance[0,2])<6*np.sqrt(expected_var[0]*expected_var[2]/n)
    assert head.stats['rate_error_bound_sum']>0 and head.stats['rate_error_bound_max']>0
    assert head.stats['rate_error_bound_sum']<1e-8 and head.stats['rate_error_bound_max']<1e-12


def test_tied_and_tiny_increments_are_retained_and_underflow_frequency_is_accounted():
    torch.manual_seed(20261114);n=20000
    theta=torch.tensor([2e-10,2e-10,2e-6,1.,1.],dtype=torch.float64)
    head=F.GammaHead([0,0,0,0,0],.5)
    result=head.rates(torch.ones((n,1,5),dtype=torch.float64),theta)
    assert torch.isfinite(result).all() and (result>=0).all()
    assert head.stats['exact_tied_time_increments']==2
    assert head.stats['minimum_positive_increment']==float(theta[0]/2)
    assert head.stats['shared_increment_draws']==n*3
    assert head.stats['independent_draws']==n*5
    assert head.stats['minimum_gamma_sample']==F.TINY
    for role in ('shared','independent'):
        expected=head.stats['expected_'+role+'_tiny_hits'];variance=head.stats[role+'_tiny_hit_variance']
        assert abs(head.stats[role+'_tiny_hits']-expected)<6*np.sqrt(variance)+3


@pytest.mark.parametrize('kind',['mu_nan','mu_negative','theta_zero','theta_nan','theta_broadcast','changed_theta','changed_theta_float64'])
def test_invalid_parameters_cannot_silently_enter_sampler(kind):
    mu=torch.ones((2,3,2));theta=torch.ones(2);head=F.GammaHead([0,1],.5)
    if kind=='mu_nan':mu[0,0,0]=float('nan')
    elif kind=='mu_negative':mu[0,0,0]=-1
    elif kind=='theta_zero':theta[0]=0
    elif kind=='theta_nan':theta[0]=float('nan')
    elif kind=='theta_broadcast':theta=theta.repeat(2,1)
    else:
        if kind=='changed_theta_float64':theta=theta.double()
        head.rates(mu,theta);theta[0]=2
    with pytest.raises(ValueError):head.rates(mu,theta)


def test_efficient_projection_equals_explicit_zero_diagonal_raw_kernel():
    mu=np.array([2.,3.,4.,5.,6.]);theta=np.array([.5,.5,2.,4.,1.]);groups=np.array([0,0,0,1,1])
    p=np.arange(15).reshape(5,3)/10
    full=.5*(mu/theta)[:,None]*(mu/theta)[None,:]*np.minimum(theta[:,None],theta[None,:])
    full*=groups[:,None]==groups[None,:];np.fill_diagonal(full,0)
    assert np.array_equal(np.diag(full),np.zeros(5)) and (full>=0).all()
    assert np.allclose(F.projected_kernel(mu,theta,groups,p),p.T@full@p,rtol=1e-13,atol=1e-13)


def test_full_logcp_jacobian_matches_count_finite_differences_including_denominator():
    mu=np.array([2.,7.,11.]);sd=np.array([.7,1.1,2.]);p=np.array([[.3,.8],[.5,-.1],[-.9,.4]])
    jac=F.logcp_jacobian(mu,p,sd)
    def project(x):return (np.log1p(1e4*x/x.sum())/sd)@p
    numeric=np.stack([(project(mu+np.eye(3)[g]*1e-5)-project(mu-np.eye(3)[g]*1e-5))/2e-5 for g in range(3)])
    assert np.allclose(jac,numeric,rtol=1e-7,atol=1e-9)
    missing_denominator=p/sd[:,None]*(1e4/(mu.sum()+1e4*mu))[:,None]
    assert not np.allclose(jac,missing_denominator)


def test_partition_does_not_alias_and_never_truncates_fractional_groups():
    groups=np.array([1,2]);head=F.GammaHead(groups,.5);groups[0]=99
    assert head.groups.tolist()==[1,2]
    for bad in ([.5,2],[-1,2],[100,0],[float('nan'),1],['0','1']):
        with pytest.raises(ValueError):F.GammaHead(bad,.5)


def test_singleton_kernel_exact_zero_and_heterogeneous_precision_avoids_cancellation():
    assert np.array_equal(F.projected_kernel([1,1],[1e-24,1e24],[0,1],np.ones((2,1))),np.zeros((1,1)))
    answer=F.projected_kernel([1,1],[1e-24,1e24],[0,0],np.ones((2,1)))
    assert np.allclose(answer,[[1e-24]],rtol=1e-12,atol=0)


def test_logcp_floor_branch_and_nondifferentiable_total_one():
    mu=np.array([.2,.3]);sd=np.array([.7,1.2]);p=np.array([[1.,2.],[-.2,.4]])
    def project(x):return (np.log1p(1e4*x/max(x.sum(),1))/sd)@p
    numeric=np.stack([(project(mu+np.eye(2)[g]*1e-6)-project(mu-np.eye(2)[g]*1e-6))/2e-6 for g in range(2)])
    assert np.allclose(F.logcp_jacobian(mu,p,sd),numeric,rtol=1e-7,atol=1e-8)
    with pytest.raises(ValueError,match='kink'):F.logcp_jacobian([.2,.8],p,sd)
