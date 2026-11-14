"""Scientific false-pass protections without generation or count/weight loads."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_hybrid as H
import compare_hybrid as C
import run_hybrid as R

np=H.np


def component(offset=0):
    shapes={'pb_counts':(2,16,3),'mean_log':(2,16,3),'nonzero':(2,16,3),
            'pcs':(2,16,4,2),'lib':(2,16),'sq_log':(2,16,3)}
    out={k:(np.arange(np.prod(shape)).reshape(shape)+offset).astype(
             np.float16 if k in ('pcs','sq_log') else np.float32) for k,shape in shapes.items()}
    out.update(targets=np.array(['train','held']),is_heldout=np.array([False,True]),
               stage=np.array([1,3]),genes=np.array(['a','b','c']))
    return out


def test_mixture_is_lossless_complete_bag_concatenation():
    model,empirical=component(),component(1000)
    before={k:v.copy() for k,v in model.items()}
    mixed=H.combine(model,empirical)
    for key in H.ARRAYS:
        assert np.array_equal(mixed[key][:,:16],model[key])
        assert np.array_equal(mixed[key][:,16:],empirical[key])
    for key in H.IDENTITY:assert np.array_equal(mixed[key],model[key])
    for key in model:assert np.array_equal(model[key],before[key])
    assert mixed['bag_component'].tolist()==['model']*16+['empirical']*16


@pytest.mark.parametrize('key',H.IDENTITY)
def test_component_identity_mismatch_rejected(key):
    model,empirical=component(),component(1000)
    empirical[key]=empirical[key][::-1]
    with pytest.raises(ValueError):H.combine(model,empirical)


@pytest.mark.parametrize('key',H.ARRAYS)
def test_wrong_component_bag_count_rejected(key):
    model,empirical=component(),component(1000)
    empirical[key]=empirical[key][:,:15]
    with pytest.raises(ValueError):H.combine(model,empirical)


def test_silent_dtype_conversion_rejected():
    model,empirical=component(),component(1000)
    empirical['pcs']=empirical['pcs'].astype(np.float32)
    with pytest.raises(ValueError):H.combine(model,empirical)


def test_fresh_reproducible_model_namespace():
    actual=H.streams(0,0,0)
    assert actual==H.streams(0,0,0)
    assert actual!=H.W.generation_streams(0,0,0)
    assert actual!=H.streams(0,0,1)
    assert actual!=H.streams(0,1,0)
    assert actual!=H.streams(1,0,0)
    assert len(set(actual.values()))==5


def test_empirical_component_reference_and_replicates_are_distinct():
    real=SimpleNamespace(panel='hvg2000',fold=0,donors=['a','b'])
    def draw(target,rep,role):return H.empirical_rng(real,target,rep,role).integers(2**31,size=8).tolist()
    ref=draw(0,0,'reference')
    assert ref==draw(0,0,'reference')
    assert ref!=draw(0,0,'component') and ref!=draw(0,1,'reference') and ref!=draw(1,0,'reference')
    with pytest.raises(ValueError):draw(0,0,'unknown')


def test_m16_replay_has_actual_valid_stagehalf_snapshot_allocation():
    matrix=H.A.expected_allocation(16,3,[i%4 for i in range(25)])
    assert matrix==[[1,0,0,2],[1,0,0,2],[0,1,0,2],[0,1,0,2],[0,0,1,1],[0,0,1,1]]
    assert [sum(row) for row in matrix]==[3,3,3,3,2,2]


def test_numeric_gates_delegate_unchanged_and_require_complete_pairs():
    verdict=C.C.verdict({}, {}, {}, {},(0,1,2))
    assert verdict['status']=='incomplete' and not verdict['passes']
    assert C.C.__file__==str(H.PREVIOUS/'compare.py')


def test_scoring_mixture_differs_from_averaging_component_quantiles():
    # Disjoint synthetic bag summaries demonstrate the mixture-tail mechanism.
    values=np.concatenate((np.zeros(16),np.ones(16)*10))
    assert np.quantile(values,.1)==0 and np.quantile(values,.9)==10
    assert (np.quantile(values[:16],.1)+np.quantile(values[16:],.1))/2==5


def prior():
    return ({'status':'completed','phase':'full','screen_eligible':[]},
            {'status':'passed','scope':'full_conditional_study','verified_artifacts':48,
             'required_artifacts':48,'complete':True,'missing':[],'screen_eligible':[],
             'confirmation_disposition':'not_triggered','prelaunch_manifest_sha256':H.A.PRELAUNCH_SHA256,
             'artifacts':[{'recipe':name,'fold':f,'replicate':r} for name in (*H.A.PRIMARY,*H.A.CONTROLS)
                          for f in range(3) for r in (0,1)]},
            {'phase':'screen','folds':[0,1,2],'passing_recipes_ranked':[],
             'arms':{name:{'status':'complete','passes':False} for name in ('blind32','radial32')}},
            {'runtime_sha256':{name:'hash' for name in R.PRIOR_FILES}})


def test_complete_failed_prior_lineage_is_accepted():
    R.validate_previous(*prior())


@pytest.mark.parametrize('remove',list(R.PRIOR_FILES))
def test_missing_prior_evidence_link_cannot_be_blessed(remove):
    status,proof,comparison,pinned=prior();pinned['runtime_sha256'].pop(remove)
    with pytest.raises(ValueError):R.validate_previous(status,proof,comparison,pinned)


@pytest.mark.parametrize('field,bad',[('complete',False),('required_artifacts',49),
    ('missing',[{}]),('prelaunch_manifest_sha256','new-unreviewed-pins'),('screen_eligible',['radial32'])])
def test_partial_or_wrong_prior_audit_is_rejected(field,bad):
    status,proof,comparison,pinned=prior();proof[field]=bad
    with pytest.raises(ValueError):R.validate_previous(status,proof,comparison,pinned)


def test_prior_dependency_changes_cannot_be_rehashed_as_new_pins(monkeypatch):
    def changed():raise ValueError('prior dependency changed')
    monkeypatch.setattr(H.A,'sources',changed)
    with pytest.raises(ValueError,match='prior dependency changed'):H.source_manifest()


@pytest.mark.parametrize('key,bad',[('baseline','stage_mean'),('empirical_role','component'),
    ('empirical_stream',1),('kpc',32),('source_selection','all donors')])
def test_empirical_reference_label_cannot_hide_wrong_stream_or_algorithm(key,bad):
    info={'baseline':'resample_blind','empirical_role':'reference','empirical_stream':0,'kpc':64,
          'frac_cells_depth_unmatched':0.,'source_selection':'training donors only; target excluded when in training'}
    H.validate_reference('empirical32',info,{},0)
    info[key]=bad
    with pytest.raises(ValueError):H.validate_reference('empirical32',info,{},0)


@pytest.mark.parametrize('mutation',['mode','bias_targets','dispersion_targets','coefficient','allocation'])
def test_model_reference_recipe_and_calibration_identity(mutation):
    import test_audit
    _,info,cache=test_audit.fixture()
    H.validate_reference('current32',info,cache,0)
    if mutation=='mode':info['mode']='biascal@stagehalf'
    elif mutation=='bias_targets':info['mode_extra']['biascal']['n_train_targets']=37
    elif mutation=='dispersion_targets':info['mode_extra']['zcal']['n_train_targets']=37
    elif mutation=='coefficient':info['mode_extra']['zcal']['s'][0]=float('nan')
    else:info['state_stage_allocation']['target_state_stage_counts'][0]=[[32,0,0,0]]+[[0]*4 for _ in range(5)]
    with pytest.raises(ValueError):H.validate_reference('current32',info,cache,0)
