"""False-pass protections for the separate artifact auditor."""
import sys
from pathlib import Path

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit as A
import generate_recipe as W


def fixture(recipe='current32'):
    train=[f'train{i}' for i in range(25)];held=[f'held{i}' for i in range(12)]
    cache={'train_donors':train,'heldout_donors':held,'manifest_sha256':'cache-hash',
           'donor_stages':{d:i%4 for group in (train,held) for i,d in enumerate(group)}}
    stages=[cache['donor_stages'][d] for d in train]
    alloc={'states':A.STATES.copy(),'targets':train+held,'is_heldout':[False]*25+[True]*12,
           'target_state_stage_counts':[A.expected_allocation(32,cache['donor_stages'][d],stages,
                                         blind=recipe=='blind32') for d in train+held]}
    block={'coverage':.6,'below':.3,'above':.1,'width':.4,'interval_score':.7}
    row={'fold':0,'M':32,'challenge':{'recipe':recipe,'fold':0,'replicate':0,
         'K':256,'PC_CELLS':64,'generation_device':'cpu'},'pruned_source_sha256':'generated-hash',
         'fingerprint':{'cache_sha256':'cache-hash','generated_sha256':'generated-hash'},
         'intervals':{split:{method:{group:dict(block) for group in ('mean','sd','combined')}
                         for method in ('percentile','rank')} for split in ('train','heldout')}}
    bias={'n_train_targets':25,'states':A.STATES.copy(),'delta':[0.]*2000}
    zcal={'n_train_targets':25,'states':A.STATES.copy(),'s':[0.]*2000,'joint_biascal':True}
    extra={'biascal':bias,'zcal':zcal,'cellshape':{'alpha':.5}}
    if recipe=='radial32':
        extra.pop('cellshape')
        extra['radialshape']={state:{'particles':512,'unique_radii':512,'alpha_min':.5/512,
                'alpha_max':511.5/512,'alpha_mean':.5,
                'law':'training-prior centered radial midrank; dependent on donor latent',
                'sorted_radii_sha256':'a'*64} for state in A.STATES}
    receipt={'challenge_recipe':recipe,'computational_replicate':0,'M':32,'K':256,'targets':37,
             'generation_device':'cpu','generated_SHA256':'generated-hash',
             'generation_streams':'challenge20261114; see committed PROTOCOL.txt',
             'cache_manifest_sha256':'cache-hash','heldout_targets':12,'states_used':A.STATES.copy(),
             'runtime':{'particlegan':'0.9.7'},'fit_artifacts_sha256':{name:'hash' for name in W.FIT_FILES},
             'mode':W.MODES_BY_NAME.get(recipe,W.MODES_BY_NAME['current32']),
             'mode_extra':extra,'state_stage_allocation':alloc}
    return row,receipt,cache


@pytest.mark.parametrize('recipe',['original32','current32','blind32','radial32'])
def test_valid_fixed_model_identities(recipe):
    row,receipt,cache=fixture(recipe)
    A.validate(recipe,0,0,row,receipt,cache)


@pytest.mark.parametrize('which',['biascal','zcal'])
def test_calibration_target_leakage_rejected(which):
    row,receipt,cache=fixture()
    receipt['mode_extra'][which]['n_train_targets']=37
    with pytest.raises(ValueError):A.validate('current32',0,0,row,receipt,cache)


@pytest.mark.parametrize('field,bad',[('K',128),('M',128),('targets',38),
                                    ('generation_device','cuda'),('computational_replicate',1)])
def test_wrong_receipt_dimensions_or_replicate_rejected(field,bad):
    row,receipt,cache=fixture();receipt[field]=bad
    with pytest.raises(ValueError):A.validate('current32',0,0,row,receipt,cache)


def test_missing_fit_artifact_rejected():
    row,receipt,cache=fixture();receipt['fit_artifacts_sha256'].pop('final.pt')
    with pytest.raises(ValueError):A.validate('current32',0,0,row,receipt,cache)


@pytest.mark.parametrize('mutation',['duplicate_donor','wrong_heldout','negative','wrong_total',
                                    'wrong_states','one_state','wrong_stage_plan'])
def test_allocation_corruption_rejected(mutation):
    _,receipt,cache=fixture();allocation=receipt['state_stage_allocation']
    if mutation=='duplicate_donor':allocation['targets'][-1]=allocation['targets'][0]
    elif mutation=='wrong_heldout':allocation['is_heldout'][0]=True
    elif mutation=='negative':allocation['target_state_stage_counts'][0][0][0]=-6
    elif mutation=='wrong_total':allocation['target_state_stage_counts'][0][0][0]=5
    elif mutation=='wrong_states':allocation['states'][0]='wrong_snapshot'
    elif mutation=='one_state':allocation['target_state_stage_counts'][0]=[[32,0,0,0]]+[[0]*4 for _ in range(5)]
    else:
        allocation['target_state_stage_counts'][0]=A.expected_allocation(32,0,
            [cache['donor_stages'][d] for d in cache['train_donors']],blind=True)
    with pytest.raises(ValueError):A.allocation(allocation,cache,32,'current32')


@pytest.mark.parametrize('field,bad',[('alpha_mean',.6),('alpha_min',.7),
    ('alpha_max',.4),('alpha_min',.1),('law','independent uniform noise'),
    ('sorted_radii_sha256','not-a-digest'),('unique_radii',512.0)])
def test_radial_strength_or_particle_law_cannot_drift(field,bad):
    row,receipt,cache=fixture('radial32')
    receipt['mode_extra']['radialshape'][A.STATES[0]][field]=bad
    with pytest.raises(ValueError):A.validate('radial32',0,0,row,receipt,cache)


def test_false_pruned_generated_hash_rejected():
    row,receipt,cache=fixture();receipt['generated_SHA256']='another-output'
    with pytest.raises(ValueError):A.validate('current32',0,0,row,receipt,cache)


def test_empirical_stream_identity_is_not_duplicated():
    row,receipt,cache=fixture('empirical32')
    receipt.update(baseline='resample_blind',kpc=64,empirical_stream=0,
                   frac_cells_depth_unmatched=0.,
                   source_selection='training donors only; target excluded when in training')
    A.validate('empirical32',0,0,row,receipt,cache)
    receipt['empirical_stream']=1
    with pytest.raises(ValueError):A.validate('empirical32',0,0,row,receipt,cache)


def test_interval_miss_partition_must_sum_to_one():
    row,receipt,cache=fixture()
    row['intervals']['heldout']['percentile']['sd']['above']=.5
    with pytest.raises(ValueError):A.validate('current32',0,0,row,receipt,cache)


@pytest.mark.parametrize('changes',[{'below':-.1,'above':.5},
    {'below':1.1,'above':-.7},{'interval_score':.3}])
def test_interval_bounds_and_score_penalty(changes):
    row,receipt,cache=fixture()
    row['intervals']['heldout']['percentile']['sd'].update(changes)
    with pytest.raises(ValueError):A.validate('current32',0,0,row,receipt,cache)


def test_eligibility_reader_never_reads_inflight_control_rows(monkeypatch):
    visited=[]
    def load(recipe,folds):
        if recipe in A.CONTROLS:raise ValueError('control scorer row not finalized')
        visited.append(recipe)
        return {},{}
    monkeypatch.setattr(A.C,'load',load)
    monkeypatch.setattr(A.C,'verdict',lambda *a,**k:{'status':'complete','passes':False})
    out=A.primary_eligibility()
    assert set(visited)==set(A.PRIMARY)
    assert all(not row['passes'] for row in out.values())


@pytest.mark.parametrize('orphan',['seed1.npz','seed1.receipt.json','unknown.bin'])
def test_unplanned_mode_artifacts_cannot_pass(tmp_path,monkeypatch,orphan):
    monkeypatch.setattr(W,'MODES',tmp_path)
    directory=tmp_path/'current32/fold0';directory.mkdir(parents=True)
    (directory/'seed0.receipt.json').write_text('{}')
    A.mode_artifact_set([('current32',0,0)])
    (directory/orphan).write_text('{}')
    with pytest.raises(ValueError):A.mode_artifact_set([('current32',0,0)])


def test_scope_and_completeness_have_distinct_proof_paths():
    paths={A.audit_path({'scope':s,'complete':c})
           for s in ('screen_only','full_conditional_study') for c in (False,True)}
    assert len(paths)==4


@pytest.mark.parametrize('draws,blind,expected',[
    (32,False,[[1,1,0,4],[1,1,0,4],[1,0,1,3],[1,0,1,3],[0,1,1,3],[0,1,1,3]]),
    (32,True,[[2,1,2,1],[2,1,1,2],[2,1,1,1],[1,2,1,1],[1,2,1,1],[1,1,2,1]]),
    (128,False,[[3,3,3,13],[3,3,2,14],[3,3,2,13],[3,3,2,13],[3,2,3,13],[3,2,3,13]])])
def test_declared_snapshot_and_stage_allocation_examples(draws,blind,expected):
    # Manually tabulated contiguous runs + round-robin mix; train counts7,6,6,6.
    assert A.expected_allocation(draws,3,[i%4 for i in range(25)],blind)==expected


def test_row_finishing_during_payload_hashing_cannot_skip_conditional_eligibility(tmp_path,monkeypatch):
    root=tmp_path/'results';modes=tmp_path/'modes'
    monkeypatch.setattr(W,'ROOT',root);monkeypatch.setattr(W,'MODES',modes)
    monkeypatch.setattr(A,'sources',lambda:{})
    missing=('radial32',2,1)
    def create(key):
        name,f,r=key;path=root/name/f'fold{f}/seed{r}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{"challenge":{"recipe":"fixed"},"pruned_source_sha256":"hash"}')
    for name in (*A.PRIMARY,*A.CONTROLS):
        for f in range(3):
            for r in (0,1):
                if (name,f,r)!=missing:create((name,f,r))
    def cache(fold):create(missing);return {}
    monkeypatch.setattr(A.P,'verify_cache',cache)
    monkeypatch.setattr(A.P,'verify_fit',lambda *a:{})
    monkeypatch.setattr(A,'cache_metadata',lambda f,c:c)
    monkeypatch.setattr(A,'one',lambda *args:{'recipe':args[0]})
    def no_eligibility():raise AssertionError('initial screen was incomplete')
    monkeypatch.setattr(A,'primary_eligibility',no_eligibility)
    result=A.audit()
    assert not result['complete'] and result['status']=='incomplete'
    assert result['verified_artifacts']==47 and result['required_artifacts']==48
    assert result['confirmation_disposition']=='eligibility_incomplete'
    assert result['missing']==[{'recipe':'radial32','fold':2,'replicate':1}]
