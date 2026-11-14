"""Synthetic exact-loop, disjoint-readout and prospective decision checks."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import joint_common as J
import gamma_head as H
import diagnose_joint as D
import test_factor_model as OLD
B=J.B;np=J.np;torch=J.torch


def test_new_zero_count_summary_loop_preserves_all_frozen_canonical_arrays(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch)
    for model,_ in models.values():B.CS.patch_generator(model,.5)
    with torch.no_grad():reference,stats=OLD.F.ORIGINAL_GENERATE(ctx,models,plans,T=targets)
    copied=copy.deepcopy(models)
    for model,_ in copied.values():H.GammaHead(np.arange(5),0.).patch(model)
    route=D.SummaryRoute(ctx.W,np.concatenate((ctx.W.numpy(),ctx.W.numpy()),axis=-1))
    with route.installed():actual,actual_stats=B.G.generate_arrays(ctx,copied,plans,T=targets)
    assert B.array_fingerprints(reference)==B.array_fingerprints(actual) and stats==actual_stats
    assert actual['raw_cov'].shape==(2,8,100,100) and np.isfinite(actual['raw_cov']).all()
    assert np.array_equal(actual['random_pcs'],actual['diagnostic_pcs'])
    assert all(m.training and p.training for m,p in copied.values())


def test_real_cell_halves_disjoint_private_stable_and_exact64_bags():
    halves,bags=D.real_indices(301,'train_a')
    assert not set(halves[0])&set(halves[1]) and set(np.concatenate(halves))==set(range(301))
    assert bags.shape==(2,16,64)
    for half in (0,1):
        for bag in bags[half]:assert len(set(bag))==64 and set(bag)<=set(halves[half])
    assert np.array_equal(bags,D.real_indices(301,'train_a')[1])
    assert not np.array_equal(bags,D.real_indices(301,'train_b')[1])
    with pytest.raises(ValueError,match='no fallback'):D.real_indices(127,'too_small')


def fixture():
    baseline={'covariance':np.repeat((2*np.eye(100))[None],2,axis=0),'sd_center':np.full((2,100),2.),
              'zero_total_cells':0,'total_generated_cells':8192,'pc_zero_total_cells':0,'total_pc_cells':2048}
    response={'covariance':np.repeat((1.2*np.eye(100))[None],2,axis=0),'sd_center':np.full((2,100),1.2),
              'zero_total_cells':0,'total_generated_cells':8192,'pc_zero_total_cells':0,'total_pc_cells':2048}
    truth={'covariance':np.tile(np.eye(100),(2,2,1,1)),'sd_center':np.ones((2,2,100))}
    return baseline,response,truth


def test_positive_training_response_requires_all_eight_joint_gates_plus_zero_rate():
    result=D.decide(*fixture())
    assert result['passes_TRAIN_mechanism_only'] is True and len(result['gates'])==9


@pytest.mark.parametrize('failure',['second_half_random_cov','second_half_random_sd','zero_totals'])
def test_sampling_half_projection_or_total_failure_cannot_be_averaged_away(failure):
    baseline,response,truth=fixture()
    if failure=='second_half_random_cov':truth['covariance'][1,:,50:,50:]=2*np.eye(50)
    elif failure=='second_half_random_sd':truth['sd_center'][1,:,50:]=2
    else:response['zero_total_cells']=1
    assert D.decide(baseline,response,truth)['passes_TRAIN_mechanism_only'] is False


def test_separate_nb_point_pass_preserves_actual_generation_mu_draws(monkeypatch):
    ctx,targets,plans,models=OLD.setup(monkeypatch);actual=[]
    for model,_ in models.values():
        original=model.parameters_nb
        def capture(*args,_original=original,**kwargs):
            mu,theta=_original(*args,**kwargs);actual.append(mu[:,0].clone());return mu,theta
        model.parameters_nb=capture
    with torch.no_grad():OLD.F.ORIGINAL_GENERATE(ctx,models,plans,T=targets)
    expected=[x.clone() for x in actual];actual.clear()
    iterator=OLD.F.ORIGINAL_ITERATE(ctx,models,plans,targets)
    try:
        for ti,start,n,model,z,c,libs,cell,count in iterator:model.parameters_nb(z,c,B.G.K,libs,cell)
    finally:iterator.close()
    assert len(expected)==len(actual) and all(torch.equal(a,b) for a,b in zip(expected,actual))


def test_replay_hooks_restore_targets_and_model_loader_on_exception():
    original_targets=B.G.targets;original_load=B.G.load_models
    with pytest.raises(RuntimeError,match='injected'):
        with D.replay_hooks(np.arange(5),.5,{}):raise RuntimeError('injected')
    assert B.G.targets is original_targets and B.G.load_models is original_load


def test_read_guard_rejects_other_counts_and_cache_halves(tmp_path):
    cache=tmp_path/'caches/hvg2000/fold0';opened=set();guard=B.Q.read_guard(cache,opened)
    guard('open',(str(cache/'train/counts.npz'),'r',0))
    assert opened=={str((cache/'train/counts.npz').resolve())}
    for path in (cache/'heldout/counts.npz',cache/'heldout/metadata.npz',cache.parent/'fold1/manifest.json',tmp_path/'external/counts.npz'):
        with pytest.raises(AssertionError):guard('open',(str(path),'r',0))


def test_analytic_alignment_is_reported_separately_from_actual_count_gates():
    baseline,response,truth=fixture()
    baseline['raw_covariance']=baseline['covariance'];response['raw_covariance']=response['covariance']
    truth['raw_covariance']=truth['covariance']
    analytic={'analytic_raw_projected_delta':np.tile((response['covariance']-baseline['covariance'])[:,None],(1,16,1,1)),
              'analytic_delta_method_logcp_projected_delta':np.tile((baseline['covariance']-response['covariance'])[:,None],(1,16,1,1))}
    tables=D.alignment_tables(baseline,response,truth,analytic)
    assert tables['half0/PCA']['raw_gap_actual_response_inner_product']>0
    assert tables['half0/PCA']['normalized_actual_delta_approximation_cosine']<0
    assert D.decide(baseline,response,truth)['passes_TRAIN_mechanism_only'] is True


def test_frozen_gene_rmse_averages_donors_before_squaring():
    error=np.array([[1.,-2.],[-1.,2.]])
    summary=D.gene_error_summary(error)
    assert summary['rmse_gene']==0 and summary['rmse_donor_gene']==np.sqrt(2.5)
