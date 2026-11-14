"""Synthetic observation geometry, gradient, scale, and neutral-route checks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cov_common as C
import covmom as M
import train_covariance as TRAIN
np=C.np;torch=C.torch
torch.set_num_threads(3)


def sample_context():
    rng=np.random.default_rng(1672);g=64;n=256
    x=torch.tensor(rng.poisson(3,size=(n,g)),dtype=torch.float32)
    w=np.linalg.qr(rng.normal(size=(g,50)))[0].astype(np.float32)
    ctx=SimpleNamespace(X=x,mean=torch.zeros(g),sd=torch.ones(g),
        space={'mean':np.zeros(g,np.float32),'sd':np.ones(g,np.float32),
               'W':w,'pc_sd':np.ones(50,np.float32)},
        ids=np.array(['a','b']),by=[np.arange(128),np.arange(128,256)],
        n_genes=g,device=torch.device('cpu'),flags=tuple(C.ARM.split(',')),fold=0,bag=128,batch_size=32)
    ctx.Z=C.T.standardize_counts(x,ctx.mean,ctx.sd);ctx.W=torch.tensor(w);ctx.trainpcs=ctx.Z@ctx.W
    return ctx


def test_covariance_matches_unbiased_numpy_and_translation():
    x=torch.randn(3,17,50,generator=torch.Generator().manual_seed(41),dtype=torch.float64)
    actual=M.covariance(x).numpy()
    assert np.allclose(actual,np.stack([np.cov(v.numpy(),rowvar=False,ddof=1) for v in x]),rtol=1e-14,atol=1e-14)
    assert torch.allclose(M.covariance(x+3),M.covariance(x),rtol=1e-13,atol=1e-13)


def test_coordinate_layout_signed_offdiagonal_geometry_and_no_crossblock():
    x=torch.randn(2,19,50,generator=torch.Generator().manual_seed(42),dtype=torch.float64)
    y=x.flip(-1)*2
    out=M.coordinates(x,y);a=M.covariance(x);b=M.covariance(y)
    assert out.shape==(2,2500)
    assert torch.allclose(out[:,:1225].square().sum(1),a.square().sum((1,2))-a.diagonal(dim1=1,dim2=2).square().sum(1),rtol=1e-13,atol=1e-13)
    assert torch.allclose(out[:,1225:].square().sum(1),b.square().sum((1,2)),rtol=1e-13,atol=1e-13)
    assert (out<0).any()
    assert torch.equal(M.coordinates(x,-y),out)


def test_cell_permutation_invariance_and_bag_independence():
    x=torch.randn(2,21,50,generator=torch.Generator().manual_seed(43),dtype=torch.float64)
    y=x.cos();out=M.coordinates(x,y)
    order=torch.randperm(21,generator=torch.Generator().manual_seed(44))
    assert torch.allclose(out,M.coordinates(x[:,order],y[:,order]),rtol=1e-13,atol=1e-13)
    assert torch.equal(out[:1],M.coordinates(x[:1],y[:1]))


@pytest.mark.parametrize('shape',[(2,1,50),(2,17,49),(2,50)])
def test_invalid_geometry_refused(shape):
    with pytest.raises(ValueError):M.covariance(torch.zeros(shape))


def test_one_center_and_global_scale_match_norm_without_whitening():
    rng=np.random.default_rng(45);values=rng.normal(size=(32,2500)).astype(np.float32)
    values[:,:1225]*=7;released=rng.normal(size=(32,150))
    center,scale,energy=M.fit_center_scale(values,released)
    assert np.array_equal(center,values.astype(np.float64).mean(0).astype(np.float32))
    assert abs(energy['covmom_scaled_reference_energy']/energy['released150_reference_energy']-1)<2e-6
    assert energy['PCA_scaled_reference_energy']>30*energy['random_scaled_reference_energy']
    assert scale==float(format(scale,'.6g'))
    assert np.array_equal(M.fit_center_scale(values,released)[0],center)
    with pytest.raises(ValueError):M.fit_center_scale(np.ones_like(values),released)


def test_setup_uses_private_reference_rng_and_state_repeats_exactly():
    ctx=sample_context();a=M.CovarianceArm(None,'d.covmom')
    before=torch.get_rng_state().clone();numpy_before=np.random.get_state()
    a.setup(ctx)
    assert torch.equal(before,torch.get_rng_state())
    assert all(np.array_equal(x,y) for x,y in zip(numpy_before,np.random.get_state()))
    b=M.CovarianceArm(None,'d.covmom');b.setup(ctx)
    assert C.T._same(a.state(),b.state()) and a.info==b.info
    assert a.info['reference_donors']==2 and a.info['reference_weighting']=='equal donors'
    assert len(a.feature_blocks(ctx))==1 and a.feature_blocks(ctx)[0].width==2500


def test_fake_forward_is_real_and_rate_gradient_matches_count_finite_difference():
    ctx=sample_context();a=M.CovarianceArm(None,'d.covmom');a.setup(ctx);block=M.CovarianceBlock(a)
    idx=torch.tensor([[0,1,2,3,4],[128,129,130,131,132]])
    counts=ctx.X[idx].double();anchor_rate=counts*.8+.3;rate=anchor_rate.clone().requires_grad_(True)
    # Use float64 projections to compare derivatives tightly; reference
    # constants themselves stay unchanged FP32 values converted to float64.
    ctx.mean=ctx.mean.double();ctx.sd=ctx.sd.double();ctx.W=ctx.W.double()
    a.projection=a.projection.double();a.center=a.center.double()
    discrete=C.T.standardize_counts(counts,ctx.mean,ctx.sd)
    smooth_anchor=C.T.standardize_counts(anchor_rate,ctx.mean,ctx.sd)
    def f(x):
        # Anchored smooth surrogate for the actual production ST gradient,
        # with discrete counts held fixed and rate different from counts.
        cf=discrete+(C.T.standardize_counts(x,ctx.mean,ctx.sd)-smooth_anchor)
        return block.fake(ctx,cf,x,x)
    cf=C.T.critic_features(counts,rate,ctx.mean,ctx.sd)
    actual=block.fake(ctx,cf,rate,counts)
    expected=block.value(C.T.standardize_counts(counts,ctx.mean,ctx.sd)@ctx.W,
                         C.T.standardize_counts(counts,ctx.mean,ctx.sd)@a.projection)
    assert torch.equal(actual,expected)
    weight=torch.linspace(-1,1,2500,dtype=torch.float64)
    grad=torch.autograd.grad((actual*weight).sum(),rate)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    for pos in ((0,0,0),(1,2,17),(0,4,55)):
        hi=anchor_rate.clone();lo=anchor_rate.clone();eps=1e-5;hi[pos]+=eps;lo[pos]-=eps
        numeric=((f(hi)-f(lo))*weight).sum()/(2*eps)
        assert torch.allclose(grad[pos],numeric,rtol=2e-6,atol=1e-7)


def test_real_and_fake_float32_forward_same_cells():
    ctx=sample_context();a=M.CovarianceArm(None,'d.covmom');a.setup(ctx);block=M.CovarianceBlock(a)
    idx=torch.tensor([[0,1,2,3],[128,129,130,131]])
    out=block.real(ctx,idx,np.array([0,1]));counts=ctx.X[idx]
    cf=C.T.standardize_counts(counts,ctx.mean,ctx.sd)
    assert torch.equal(out,block.fake(ctx,cf,counts,counts))


def test_recipe_cap_and_forbidden_variants():
    assert C.KAPPA==.679975 and abs(C.KAPPA*(4650**.5)/(2150**.5)-1)<1e-6
    assert C.T.arm_name(C.T.parse_arm(C.ARM))==C.ARM
    assert C.T.arms.train_options(C.T.parse_arm(C.ARM)).kappa==C.KAPPA
    ctx=sample_context();ctx.flags=('d.covmom','pbulk')
    with pytest.raises(ValueError,match='fixed'):M.CovarianceArm(None,'d.covmom').setup(ctx)


def test_deferred_generation_restores_original_after_exception():
    original=C.G.generate_fit
    with pytest.raises(RuntimeError,match='injected'):
        with TRAIN.deferred_generation():
            assert C.G.generate_fit('unopened','unopened')[1]['status'].startswith('deferred')
            raise RuntimeError('injected')
    assert C.G.generate_fit is original


def test_checkpoint_peak_counts_temporary_file_and_restores_on_cap_failure(tmp_path,monkeypatch):
    p=tmp_path/'recovery.partial.pt';original=torch.save;seen=[]
    def peak():
        assert p.exists() and p.stat().st_size>0
        seen.append(True);return p.stat().st_size+100
    monkeypatch.setattr(C,'disk_bytes',peak);stats={}
    # First measurement is before the write; subsequent measurements include
    # the temporary file. Other already-retained bytes remain counted.
    monkeypatch.setattr(C,'disk_bytes',lambda:peak() if p.exists() else 100)
    with TRAIN.watch_checkpoint_peak(stats):torch.save({'synthetic':torch.ones(1)},p)
    assert seen==[True] and stats['peak_logical_task_bytes']==p.stat().st_size+100
    assert stats['max_serialized_checkpoint_bytes']==p.stat().st_size
    assert torch.equal(torch.load(p,weights_only=False)['synthetic'],torch.ones(1))
    assert torch.save is original


def test_checkpoint_preflight_prevents_disk_cap_breach_before_writing(tmp_path,monkeypatch):
    p=tmp_path/'recovery.partial.pt';original=torch.save;stats={}
    monkeypatch.setattr(C,'disk_bytes',lambda:2999999999)
    with pytest.raises(ValueError,match='preflight would exceed'):
        with TRAIN.watch_checkpoint_peak(stats):torch.save({'synthetic':torch.ones(1)},p)
    assert not p.exists() and torch.save is original
