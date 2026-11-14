"""Frozen identity and analytic count-kernel checks, with synthetic counts only."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pair_common as B
import pair_cells as K
import generate_pair as D
from scipy.sparse import csr_matrix
np=B.np


def fake_real():
    class Forbidden:
        def __getattribute__(self,key):raise AssertionError('held-out source counts accessed')
    x=np.array([[3,1,0,1],[0,5,1,2],[2,0,4,0],[1,1,1,2]])
    return SimpleNamespace(donors=np.array(['a','b','held']),is_heldout=np.array([False,False,True]),
        stage=np.array([0,1,2]),genes=np.array(['g0','g1','g2','g3']),panel='toy',fold=0,
        cells=[csr_matrix(x),csr_matrix(x[::-1]+np.array([0,0,1,0])),Forbidden()],
        libs=[x.sum(1),(x[::-1]+np.array([0,0,1,0])).sum(1),np.array([2,12])],
        mean=np.zeros(4),sd=np.ones(4),W=np.arange(8).reshape(4,2)/8,pc_sd=np.array([1.,2.]))


def test_single_cell_control_is_bitwise_frozen_resampler(monkeypatch):
    real=fake_real()
    monkeypatch.setattr(B.BL,'_target_rng',lambda r,t:B.empirical_rng(r,t,1,'reference'))
    expected,_=B.BL.generate('resample_blind',real,M=32,K=256,kpc=64)
    actual,info=K.generate(real,1,32,'reference',parts=1)
    assert B.array_fingerprints(expected)==B.array_fingerprints(actual)
    assert info['parts']==1


def test_paired_generation_never_reads_heldout_counts_or_uses_self_source():
    real=fake_real();arrays,info=K.generate(real,0,16,'component',parts=2)
    assert info['target_bag_source_donors']['a']==['b']*16
    assert info['target_bag_source_donors']['b']==['a']*16
    assert set(info['target_bag_source_donors']['held'])=={'a','b'}
    assert arrays['pb_counts'].shape==(3,16,4) and arrays['is_heldout'].tolist()==[False,False,True]
    assert np.isfinite(arrays['mean_log']).all() and (arrays['nonzero']>=0).all()


def test_cardinality_only_source_reconstruction_matches_actual_kernel(monkeypatch):
    real=fake_real();real.panel='hvg2000'
    _,info=K.generate(real,1,16,'component',parts=2)
    for i,target in enumerate(real.donors):
        expected=B.component_sources(real.donors[:2].tolist(),str(target),len(real.libs[i]),0,1)
        assert info['target_bag_source_donors'][target]==expected


@pytest.mark.parametrize('target_depth',[2.,10.])
def test_actual_kernel_matches_mean_and_covariance_identity_including_fallback(target_depth):
    source=np.array([[3,0],[0,3],[1,2]])
    cells=csr_matrix(source);libraries=source.sum(1);n=100000;target=np.full(n,target_depth)
    p=min(1.,target_depth/3)
    selected=source if target_depth<=3 else source[-1:]
    mu=(p*selected).mean(0);conditional=p*selected
    centered=conditional-conditional.mean(0)
    sigma=centered.T@centered/len(selected)+np.diag((selected*p*(1-p)).mean(0))
    single,missing1=K.draw(cells,libraries,target,np.random.default_rng(105),1)
    paired,missing2=K.draw(cells,libraries,target,np.random.default_rng(106),2)
    assert missing1==missing2==(n if target_depth>3 else 0)
    assert np.issubdtype(paired.dtype,np.integer) and (paired>=0).all()
    assert np.allclose(single.mean(0),mu,atol=.012,rtol=0)
    assert np.allclose(paired.mean(0),mu,atol=.012,rtol=0)
    assert np.allclose(np.cov(single,rowvar=False,bias=True),sigma,atol=.025,rtol=0)
    assert np.allclose(np.cov(paired,rowvar=False,bias=True),.5*sigma+.5*np.diag(mu),atol=.025,rtol=0)
    if target_depth>3:assert paired.mean(0).sum()<4 # fallback is not upsampled to10


def test_two_independent_picks_and_half_thinning_use_same_source_donor():
    class RNG:
        def __init__(self):self.picks=0;self.probabilities=[]
        def random(self,n):
            self.picks+=1;return np.full(n,0 if self.picks==1 else .999)
        def binomial(self,n,p):
            self.probabilities.append(p.copy());return n
    cells=csr_matrix([[2,0],[0,2]]);rng=RNG()
    counts,_=K.draw(cells,np.array([2,2]),np.array([2]),rng,2)
    assert counts.tolist()==[[2,2]] and rng.picks==2
    assert all(np.array_equal(p,[.5]) for p in rng.probabilities)


def test_fixed_half_bins_preserve_every_component_bag_summary_exactly():
    real=fake_real();left,_=K.generate(real,0,16,'component');right,_=K.generate(real,1,16,'component')
    mixed=B.H.combine(left,right)
    for key in B.H.ARRAYS:
        assert np.array_equal(mixed[key][:,:16],left[key])
        assert np.array_equal(mixed[key][:,16:],right[key])
    changed=dict(right,is_heldout=~right['is_heldout'])
    with pytest.raises(ValueError):B.H.combine(left,changed)


@pytest.mark.parametrize('parts',[0,3,4])
def test_no_pair_size_search_is_available(parts):
    with pytest.raises(ValueError):K.draw(csr_matrix([[1]]),np.array([1]),np.array([1]),np.random.default_rng(1),parts)


def test_fresh_streams_and_roles_are_distinct():
    real=fake_real()
    assert B.streams(0,0,0)!=B.streams(0,0,1)!=B.D.streams(0,0,1)
    assert not np.array_equal(B.empirical_rng(real,0,0,'reference').integers(100,size=32),
                             B.empirical_rng(real,0,0,'component').integers(100,size=32))
