"""False-pass protections for the fixed law, prior lineage and zero control."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as B
import factor_model as F
import generate_factor as D
import run_factor as R


def law_fixture():
    train=['a','b','c'];held=['d'];raw=[-.5,0.,.5]
    h=F.normalized_factors(raw)
    records=[{'donor':d,'u':u,'generated_bags':32,'generated_cells':32*256,
              'generated_gene_values':32*256*2000,'observed_nonzero':.5,'F_low':.2,'F_high':.8,
              'expected_nonzero_residual':0.,'clipped':None,'bisection_steps':12} for d,u in zip(train,raw)]
    law={'fit_donors':train,'n_train_targets':3,'donor_fits':records,'h':h.tolist(),
         'generation_h_float32':h.astype(B.np.float32).tolist(),'mean_inverse_precision_factor':1.,
         'mean_inverse_precision_factor_float32':float(B.np.exp(-h.astype(B.np.float32).astype(B.np.float64)).mean()),
         'normalization':'h=u+log(mean(exp(-u)))','fitting':'12-step bounded bisection; u in[-1,1]',
         'target_bag_indices':{d:B.np.random.default_rng(B.factor_seed(0,d,0)).integers(3,size=32).tolist()
                               for d in train+held}}
    return {'factor_law':law},{'train_donors':train,'heldout_donors':held}


def test_valid_fixed_law_and_assignment_metadata():
    info,cache=law_fixture();D.validate_law(info,cache,0,0)


@pytest.mark.parametrize('mutation',['heldout_fit','bound','wrong_centering','clipping','iterations','replicate','draw_count'])
def test_law_or_donor_identity_drift_is_rejected(mutation):
    info,cache=law_fixture();law=info['factor_law']
    if mutation=='heldout_fit':law['fit_donors']=['a','b','d']
    elif mutation=='bound':law['donor_fits'][0]['u']=-1.1
    elif mutation=='wrong_centering':law['h']=[-.5,0.,.5]
    elif mutation=='clipping':law['donor_fits'][0]['clipped']='low'
    elif mutation=='iterations':law['donor_fits'][0]['bisection_steps']=10
    elif mutation=='replicate':law['target_bag_indices']['a']=B.np.random.default_rng(B.factor_seed(0,'a',1)).integers(3,size=32).tolist()
    else:law['target_bag_indices']['a'].pop()
    with pytest.raises(ValueError):D.validate_law(info,cache,0,0)


def prior_fixture():
    status={'status':'completed','screen_passes':False}
    proof={'status':'passed','complete':True,'scope':'full_conditional_study',
           'verified_artifacts':24,'required_artifacts':24,'missing':[],'screen_passes':False,
           'confirmation_disposition':'not_triggered','prelaunch_manifest_sha256':B.PRIOR_MANIFEST_SHA,
           'artifacts':[{'recipe':name,'fold':f,'replicate':r} for name in B.H.RECIPES for f in range(3) for r in (0,1)]}
    comparison={'phase':'screen','folds':[0,1,2],'candidate':'hybrid32','verdict':{'status':'complete','passes':False}}
    pinned={'runtime_sha256':{name:'hash' for name in R.PRIOR_FILES}}
    return status,proof,comparison,pinned


def test_prior_complete_failed_study_is_accepted():R.validate_previous(*prior_fixture())


@pytest.mark.parametrize('mutation',['missing_link','incomplete','empty_artifacts','duplicate_artifact','changed_pins','passing'])
def test_prior_partial_or_wrong_lineage_cannot_authorize_new_study(mutation):
    status,proof,comparison,pinned=prior_fixture()
    if mutation=='missing_link':pinned['runtime_sha256'].pop('screen_comparison.json')
    elif mutation=='incomplete':proof['complete']=False
    elif mutation=='empty_artifacts':proof['artifacts']=[]
    elif mutation=='duplicate_artifact':proof['artifacts'][-1]=proof['artifacts'][0]
    elif mutation=='changed_pins':proof['prelaunch_manifest_sha256']='new-pins'
    else:comparison['verdict']['passes']=True
    with pytest.raises(ValueError):R.validate_previous(status,proof,comparison,pinned)


def test_prior178_dependencies_cannot_be_reblessed(monkeypatch):
    monkeypatch.setattr(B.H,'source_manifest',lambda:{})
    with pytest.raises(ValueError,match='prior178'):B.source_manifest()


def zero_fixture(tmp_path,monkeypatch):
    root=tmp_path/'results';modes=tmp_path/'modes'
    monkeypatch.setattr(B,'ROOT',root);monkeypatch.setattr(B,'MODES',modes)
    root.mkdir();current=modes/'current32/fold0/seed0.receipt.json';current.parent.mkdir(parents=True)
    fingerprints={name:'a'*64 for name in (*B.H.ARRAYS,*B.H.IDENTITY)}
    current.write_text(json.dumps({'array_fingerprints':fingerprints,'mode_extra':{'biascal':{'delta':[.1]},'zcal':{'s':[.2]}}}))
    proof={'passed':True,'sources':{},'fold':0,'replicate':0,'current_receipt_sha256':B.sha(current),
           'zero_array_fingerprints':fingerprints,'zero_bias_delta':[.1],'zero_gene_s':[.2],
           'pruned_control_sha256':'b'*64}
    path=root/'zero_factor_fidelity.json';path.write_text(json.dumps(proof))
    return path,proof


def test_exact_full_zero_fidelity_metadata(tmp_path,monkeypatch):
    zero_fixture(tmp_path,monkeypatch);D.verify_fidelity({})


@pytest.mark.parametrize('mutation',['source','arrays','tiny_coefficient_error','unpruned',
                                    'missing_digest','malformed_digest','malformed_fingerprint','missing_fingerprint',
                                    'malformed_reference'])
def test_zero_control_requires_exact_arrays_coefficients_and_pruning(tmp_path,monkeypatch,mutation):
    path,proof=zero_fixture(tmp_path,monkeypatch)
    if mutation=='source':proof['sources']={'different':'code'}
    elif mutation=='arrays':proof['zero_array_fingerprints']={'data':'different'}
    elif mutation=='tiny_coefficient_error':proof['zero_gene_s'][0]+=1e-10
    elif mutation=='missing_digest':proof.pop('pruned_control_sha256')
    elif mutation=='malformed_digest':proof['pruned_control_sha256']='g'*64
    elif mutation=='malformed_fingerprint':proof['zero_array_fingerprints']['pcs']='short'
    elif mutation=='missing_fingerprint':proof['zero_array_fingerprints'].pop('pcs')
    elif mutation=='malformed_reference':
        current=B.MODES/'current32/fold0/seed0.receipt.json'
        info=B.P.read(current);info['array_fingerprints']['pcs']='invalid'
        current.write_text(json.dumps(info));proof['current_receipt_sha256']=B.sha(current)
    else:
        archive=B.MODES/'zero_fidelity/fold0/seed0.npz';archive.parent.mkdir(parents=True);archive.write_bytes(b'not-pruned')
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):D.verify_fidelity({})


def test_common_fingerprints_require_literal_dtype_and_value_identity():
    arrays={name:B.np.arange(4,dtype=B.np.float32) for name in (*B.H.ARRAYS,*B.H.IDENTITY)}
    initial=B.array_fingerprints(arrays)
    arrays['pcs']=arrays['pcs'].astype(B.np.float64)
    assert initial!=B.array_fingerprints(arrays)
    arrays['pcs']=B.np.arange(4,dtype=B.np.float32);arrays['mean_log'][1]+=1e-6
    assert initial!=B.array_fingerprints(arrays)
