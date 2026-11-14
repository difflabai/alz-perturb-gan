"""Synthetic gate and tamper tests; no model, real cache, or training execution."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cov_common as C
import resource_common as U
import generate_covariance as D
import compare_covariance as COMPARE
import audit_covariance as AUDIT
import run_covariance as RUN


def gpu_info():
    return {'physical_selector':'1','memory_fraction':.04,'device_name':'synthetic GPU',
            'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'allow_tf32':False,
            'cudnn_deterministic':True,'cudnn_benchmark':False,'total_device_bytes':10000,
            'max_allocated_bytes':100,'max_reserved_bytes':200}


def cov_info(n_train):
    return {'features':2500,'PCA_offdiagonal':1225,'random_upper_including_diagonal':1275,
        'cross_space_terms':False,'covariance_divisor':'cells-1','reference_bags_per_donor':16,
        'reference_cells_per_bag':128,'reference_donors':n_train,'reference_weighting':'equal donors',
        'total_feature_width':4650,'native_kappa':C.KAPPA,'random_basis_seed':20261113,
        'random_sign_canonicalization':False,'reference_indices_sha256':'b'*64,'scale':1.,
        'energies':{'released150_reference_energy':1.,'covmom_centered_reference_energy':1.}}


def row(score,profile=1.):
    historical={k:1. for k in COMPARE.ORIGINAL.HISTORICAL}
    historical['pb_rmse']=profile;historical['pc_spread_rep']=1.
    block={g:{'interval_score':score,'coverage':.8,'below':.1,'above':.1,'width':.5}
           for g in ('mean','sd','combined')}
    return {'historical':historical,'intervals':{s:{'percentile':copy.deepcopy(block)} for s in ('train','heldout')}}


def configure_rows(tmp_path,monkeypatch,fresh_score=1.2,missing=None):
    monkeypatch.setattr(C,'ROOT',tmp_path);sources={'synthetic':'source'}
    monkeypatch.setattr(C,'sources',lambda:sources)
    monkeypatch.setattr(D,'finalized',lambda *args:True)
    (tmp_path/'source_manifest.json').write_text(json.dumps(sources))
    for name,score in (('original32',1.),('current32',1.),('empirical32',1.5),
                       ('training_control32',fresh_score),('covariance32',.5)):
        for fold in range(3):
            for seed in (0,1):
                if (name,fold,seed)==missing:continue
                p=tmp_path/name/f'fold{fold}/seed{seed}.json';p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(row(score,.999 if name=='covariance32' else 1.)))


def test_both_frozen_and_fresh_control_verdicts_must_pass(tmp_path,monkeypatch):
    configure_rows(tmp_path,monkeypatch)
    report=COMPARE.report((0,1,2))
    assert report['passes'] and all(v['passes'] for v in report['verdicts'].values())
    assert len(report['score_sha256'])==30


def test_frozen_default_gain_does_not_override_fresh_control_failure(tmp_path,monkeypatch):
    configure_rows(tmp_path,monkeypatch,fresh_score=.1)
    report=COMPARE.report((0,1,2))
    assert report['verdicts']['frozen_default_control']['passes']
    assert not report['verdicts']['fresh_training_control']['passes'] and not report['passes']


def test_missing_fresh_training_pair_cannot_pass(tmp_path,monkeypatch):
    configure_rows(tmp_path,monkeypatch,missing=('training_control32',2,1))
    report=COMPARE.report((0,1,2))
    assert report['verdicts']['fresh_training_control']['status']=='incomplete' and not report['passes']


def test_stored_json_key_roundtrip_is_exact_and_rejects_small_numeric_change():
    value={0:{'score':.5},1:{'score':.6}}
    saved=json.loads(json.dumps(value))
    assert AUDIT.serialized_equal(saved,value)
    saved['0']['score']+=1e-10
    assert not AUDIT.serialized_equal(saved,value)


def make_fit(tmp_path,monkeypatch,role='control'):
    monkeypatch.setattr(C,'FITS',tmp_path/'fits');monkeypatch.setattr(C,'ROOT',tmp_path/'results')
    fit=C.fit_path(role,0,0);fit.mkdir(parents=True)
    anchor={'manifest_sha256':'a'*64,'train_donors':['a','b'],'heldout_donors':['c']}
    monkeypatch.setattr(C.B.P,'verify_cache',lambda *args,**kwargs:anchor)
    sources={'synthetic':'source'};monkeypatch.setattr(C,'verification_sources',lambda:sources)
    monkeypatch.setattr(C,'sources',lambda:sources)
    weights={}
    for name in D.WEIGHTS:
        (fit/name).write_bytes(('synthetic-checkpoint-'+name).encode());weights[name]=C.B.sha(fit/name)
    flags=C.T.parse_arm('pbulk' if role=='control' else C.ARM)
    config={'flags':list(flags),'arm':C.T.arm_name(flags),'fold':0,'seed':0,'panel':'hvg2000','n_genes':2000,
        'updates':20000,'device':U.TRAIN_DEVICE,'recipe':json.loads(json.dumps(C.T.recipe(C.T.arms.train_options(flags)).to_dict())),
        'train_donors':anchor['train_donors'],'heldout_donors':anchor['heldout_donors'],
        'cache_manifest_SHA256':anchor['manifest_sha256'],'critic_features':2150 if role=='control' else 4650,
        'cache':str((C.RUNS/'caches/hvg2000/fold0').resolve()),'protocol_split':True,'batch':[32,128],
        'stream_root':C.T.STREAM_ROOT,'recipe_total_steps':20000,'particlegan_version':C.T.VERSION,
        'native_recipe':'bispf_optimized'}
    spec=D.resolved_training_spec(role)
    config.update({k:v for k,v in spec.items() if k!='train_departures'})
    config['snapshot_updates']=C.T.snapshot_updates(20000)
    config['r2_arms']={'pbulk':{'name':'pbulk','value':None,'info':{}}}
    config['departures']=spec['train_departures']
    if role=='candidate':
        cov=cov_info(len(anchor['train_donors']))
        config['r2_arms']['d.covmom']={'name':'d.covmom','value':None,'info':cov}
        config['departures']['d.covmom']={'kind':'critic architecture','value':None,'info':cov}
    receipt={'status':'training_completed','completed_updates':20000,'endpoint':20000,'device':U.TRAIN_DEVICE,'threads':3,
        'full_20k_endpoint':True,'full_horizon_endpoint':True,
        'snapshots_missing':[],'final_SHA256':weights['final.pt'],
        'snapshots_SHA256':{str(u):weights[f'snap_{u}.pt'] for u in C.T.snapshot_updates(20000)}}
    C.B.write(fit/'config.json',config);C.B.write(fit/'fit_receipt.json',receipt)
    C.B.write(fit/'checkpoint_receipt.json',{'status':'completed','completed_updates':20000,'path':'final.pt','SHA256':weights['final.pt']})
    (fit/'learning_curve.jsonl').write_text('{"step":20000}\n')
    C.B.write(fit/'study_training_receipt.json',{'status':'native_training_completed_TRAIN_only','role':role,'fold':0,'seed':0,
        'fit':str(fit),'sources':sources,'automatic_generation_deferred':True,'cache_manifest_sha256':anchor['manifest_sha256'],
        'fit_receipt_sha256':C.B.sha(fit/'fit_receipt.json'),'cpu_affinity':[0,1,2],'peak_logical_task_bytes':1000,
        'gpu_resource':gpu_info(),
        'cache_files_opened':[str((C.RUNS/'caches/hvg2000/fold0'/n).resolve()) for n in
            ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')],'weight_sha256':weights})
    return fit,weights


@pytest.mark.parametrize('role',['control','candidate'])
def test_fresh_native_fit_lineage_hashes_all_retained_weights(tmp_path,monkeypatch,role):
    fit,weights=make_fit(tmp_path,monkeypatch,role)
    actual,hashes=D.new_fit(role,0,0)
    assert actual==fit and all(hashes[k]==v for k,v in weights.items())


@pytest.mark.parametrize('change',['weight','TRAIN_scope','recipe','seed','cache_path','source','endpoint'])
def test_new_fit_tamper_is_rejected(tmp_path,monkeypatch,change):
    fit,_=make_fit(tmp_path,monkeypatch)
    if change=='weight':(fit/'snap_17000.pt').write_bytes(b'changed')
    elif change in ('recipe','seed','cache_path'):
        path=fit/'config.json';x=C.B.P.read(path)
        if change=='recipe':x['recipe']['reg_kappa']=.1
        elif change=='seed':x['seed']=1
        else:x['cache']='/different/cache'
        C.B.write(path,x)
    elif change in ('TRAIN_scope','source'):
        path=fit/'study_training_receipt.json';x=C.B.P.read(path)
        if change=='TRAIN_scope':x['cache_files_opened'].append(str(C.RUNS/'caches/hvg2000/fold0/heldout/counts.npz'))
        else:x['sources']={}
        C.B.write(path,x)
    else:
        path=fit/'fit_receipt.json';x=C.B.P.read(path);x['completed_updates']=19999;C.B.write(path,x)
    with pytest.raises(ValueError):D.new_fit('control',0,0)


def test_pruned_control_requires_complete_verified_hash_links_and_absence(tmp_path,monkeypatch):
    fit,weights=make_fit(tmp_path,monkeypatch);_,hashes=D.new_fit('control',0,0)
    score=C.ROOT/'training_control32/fold0/seed0.json';C.B.write(score,{'synthetic':'score'})
    C.B.write(fit/'control_weight_pruning.json',{'status':'verified_and_pruned_control_weights','sources':C.sources(),
        'fit_artifacts_sha256':hashes,'weight_sha256':weights,'score_sha256':C.B.sha(score)})
    with pytest.raises(ValueError):D.new_fit('control',0,0,allow_pruned=True)
    for name in weights:(fit/name).unlink()
    assert D.new_fit('control',0,0,allow_pruned=True)[1]==hashes
    with pytest.raises(ValueError):D.new_fit('control',0,0)
    C.B.write(score,{'synthetic':'changed_score'})
    with pytest.raises(ValueError):D.new_fit('control',0,0,allow_pruned=True)


@pytest.mark.parametrize('field',['learning_rates','training_options','hyperparameters','native_ema_decay',
    'critic_layout','departures','snapshot_updates','study','round','checkpoint_every','folds_file'])
@pytest.mark.parametrize('role',['control','candidate'])
def test_resolved_effective_config_cannot_disagree_with_recipe(tmp_path,monkeypatch,field,role):
    fit,_=make_fit(tmp_path,monkeypatch,role);path=fit/'config.json';cfg=C.B.P.read(path)
    if field=='learning_rates':cfg[field]['critic']*=2
    elif field=='training_options':cfg[field]['reg']=2
    elif field=='hyperparameters':cfg[field]['prior_reg']=.5
    elif field=='critic_layout':cfg[field]['blocks'][-1]['columns'][0]+=1
    elif field=='snapshot_updates':cfg[field]=cfg[field][:-1]
    elif field=='departures':cfg[field]['undeclared_extra_loss']={}
    elif field=='study':cfg[field]='another_study'
    elif field=='round':cfg[field]=4
    elif field=='checkpoint_every':cfg[field]+=1
    elif field=='folds_file':cfg[field]='/another/folds.json'
    else:cfg[field]=.5
    C.B.write(path,cfg)
    with pytest.raises(ValueError):D.new_fit(role,0,0)


@pytest.mark.parametrize('field,value',[('memory_fraction',.05),('physical_selector','0'),
    ('device_name',''),('deterministic_algorithms',False),('allow_tf32',True),('cublas_workspace_config',''),
    ('cudnn_deterministic',False),('cudnn_benchmark',True),('total_device_bytes',0),
    ('max_allocated_bytes',401),('max_reserved_bytes',-1)])
def test_gpu_resource_mismatch_rejected(tmp_path,monkeypatch,field,value):
    fit,_=make_fit(tmp_path,monkeypatch);path=fit/'study_training_receipt.json';record=C.B.P.read(path)
    record['gpu_resource'][field]=value;C.B.write(path,record)
    with pytest.raises(ValueError):D.new_fit('control',0,0)


def calibration(n_train=25):
    return {'biascal':{'n_train_targets':n_train,'states':sorted(C.B.A.STATES),'delta':[0.]*2000},
        'cellshape':{'alpha':.5},'zcal':{'n_train_targets':n_train,'states':sorted(C.B.A.STATES),
            's':[-6.,0.,6.]+[-.5]*1997,'joint_biascal':True}}


def test_negative_zero_and_boundary_log_theta_shifts_are_valid():
    D.validate_calibration(calibration(),25)


@pytest.mark.parametrize('value',[-6.000001,6.000001,float('nan'),float('inf')])
def test_invalid_log_theta_shifts_rejected(value):
    extra=calibration();extra['zcal']['s'][0]=value
    with pytest.raises(ValueError):D.validate_calibration(extra,25)


def fingerprints():
    return {key:'e'*64 for key in (*C.B.H.ARRAYS,*C.B.H.IDENTITY)}


def make_scored(tmp_path,monkeypatch,recipe='current32'):
    monkeypatch.setattr(C,'ROOT',tmp_path/'results');monkeypatch.setattr(C,'MODES',tmp_path/'modes')
    monkeypatch.setattr(C,'RUNS',tmp_path/'runs')
    cache=C.RUNS/'caches/hvg2000/fold0';C.B.write(cache/'manifest.json',{'synthetic':True})
    anchor={'manifest_sha256':C.B.sha(cache/'manifest.json'),
        'train_donors':[f'train{x}' for x in range(25)],'heldout_donors':[f'held{x}' for x in range(12)]}
    monkeypatch.setattr(C.B.P,'verify_cache',lambda *a,**kw:anchor)
    monkeypatch.setattr(C.B.A,'cache_metadata',lambda *a,**kw:anchor)
    monkeypatch.setattr(C.B.A,'allocation',lambda *a,**kw:None)
    fit=tmp_path/'synthetic_fit';hashes={'config.json':'c'*64}
    monkeypatch.setattr(D,'recipe_fit',lambda *a,**kw:(fit,hashes))
    sources={str(C.B.W.R5/'architecture_score.py'):'a'*64,str(C.B.W.R5/'interval_metrics.py'):'b'*64}
    monkeypatch.setattr(C,'sources',lambda:sources)
    archive=C.MODES/recipe/'fold0/seed0.npz';receipt=archive.with_suffix('.receipt.json')
    score=C.ROOT/recipe/'fold0/seed0.json'
    info={'challenge_recipe':recipe,'computational_replicate':0,'sources':sources,
        'generation_streams':C.NAMESPACE,'generation_device':'cpu','targets':37,'M':32,'K':256,
        'generated_SHA256':'f'*64,'cache_manifest_sha256':anchor['manifest_sha256'],'cpu_affinity':[0,1,2],
        'array_fingerprints':fingerprints(),'fit_artifacts_sha256':hashes,'fit':str(fit),
        'runtime':{'particlegan':str(C.T.VERSION)},'heldout_targets':12,
        'mode':C.B.W.MODES_BY_NAME['current32'],'states_used':C.B.A.STATES,
        'state_stage_allocation':{},'mode_extra':calibration(),
        'baseline':'resample_blind','empirical_role':'reference','empirical_stream':0,'kpc':64,
        'frac_cells_depth_unmatched':.1,
        'source_selection':'training donors only; target excluded when in training'}
    scored=row(1.);scored['intervals']={group:{'percentile':copy.deepcopy(scored['intervals'][group]['percentile']),
        'rank':copy.deepcopy(scored['intervals'][group]['percentile'])} for group in ('train','heldout')}
    scored.update(fold=0,M=32,source=str(archive.resolve()),pruned_source_sha256='f'*64,
        fingerprint={'generated_sha256':'f'*64,'cache_sha256':anchor['manifest_sha256'],
            'score_code_sha256':'a'*64,'interval_code_sha256':'b'*64})
    def save():
        C.B.write(receipt,info)
        scored['challenge']={'recipe':recipe,'fold':0,'replicate':0,'sources':sources,
            'receipt_sha256':C.B.sha(receipt),'generation_device':'cpu','K':256,'PC_CELLS':64}
        C.B.write(score,scored)
    save()
    return info,scored,save,sources,archive


@pytest.mark.parametrize('recipe',['current32','empirical32'])
def test_complete_synthetic_row_validates(tmp_path,monkeypatch,recipe):
    *_,sources,archive=make_scored(tmp_path,monkeypatch,recipe)
    assert D.finalized(recipe,0,0,sources)


@pytest.mark.parametrize('change',['source','scorer','interval_code','nonfinite','miss_partition','negative_width',
    'score_below_width','coverage_bounds','generated_digest','affinity','runtime','held_targets','joint',
    'log_s','unpruned','missing_fingerprint'])
def test_finalized_rejects_metadata_counterexamples(tmp_path,monkeypatch,change):
    info,scored,save,sources,archive=make_scored(tmp_path,monkeypatch)
    block=scored['intervals']['heldout']['rank']['sd']
    if change=='source':scored['source']='/wrong/source.npz'
    elif change=='scorer':scored['fingerprint']['score_code_sha256']='0'*64
    elif change=='interval_code':scored['fingerprint']['interval_code_sha256']='0'*64
    elif change=='nonfinite':block['interval_score']=float('nan')
    elif change=='miss_partition':block['below']=.2
    elif change=='negative_width':block['width']=-.1
    elif change=='score_below_width':block['interval_score']=.1
    elif change=='coverage_bounds':block['coverage']=1.01
    elif change=='generated_digest':
        info['generated_SHA256']=scored['pruned_source_sha256']=scored['fingerprint']['generated_sha256']='invalid'
    elif change=='affinity':info['cpu_affinity']=[]
    elif change=='runtime':info['runtime']['particlegan']='other'
    elif change=='held_targets':info['heldout_targets']=0
    elif change=='joint':info['mode_extra']['zcal']['joint_biascal']=False
    elif change=='log_s':info['mode_extra']['zcal']['s'][0]=7.
    elif change=='missing_fingerprint':info['array_fingerprints'].pop('genes')
    else:archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(b'synthetic-unpruned')
    # JSON writer intentionally disallows NaN, so test its pure validation directly.
    if change=='nonfinite':
        with pytest.raises(ValueError):D.validate_intervals(scored)
        return
    save()
    with pytest.raises(ValueError):D.finalized('current32',0,0,sources)


@pytest.mark.parametrize('change',['kpc','depth','replicate','pool'])
def test_empirical_metadata_cannot_spoof_reference(tmp_path,monkeypatch,change):
    info,scored,save,sources,_=make_scored(tmp_path,monkeypatch,'empirical32')
    if change=='kpc':info['kpc']=32
    elif change=='depth':info['frac_cells_depth_unmatched']=1.1
    elif change=='replicate':info['empirical_stream']=1
    else:info['source_selection']='all donors'
    save()
    with pytest.raises(ValueError):D.finalized('empirical32',0,0,sources)


@pytest.mark.parametrize('change',['missing','failed','stale','incomplete'])
def test_confirmation_requires_successful_complete_current_screen(tmp_path,monkeypatch,change):
    configure_rows(tmp_path,monkeypatch,fresh_score=.1 if change=='failed' else 1.2,
        missing=('training_control32',2,1) if change=='incomplete' else None)
    report=COMPARE.report((0,1,2))
    if change!='missing':
        if change=='stale':report['scope']='changed'
        C.B.write(C.ROOT/'screen_comparison.json',report)
    with pytest.raises(ValueError):D.require_confirmation_eligible(3,C.sources())


def test_confirmation_records_exact_screen_and_all30_score_hashes(tmp_path,monkeypatch):
    configure_rows(tmp_path,monkeypatch);report=COMPARE.report((0,1,2))
    C.B.write(C.ROOT/'screen_comparison.json',report)
    link=D.require_confirmation_eligible(8,C.sources())
    assert link['screen_comparison_sha256']==C.B.sha(C.ROOT/'screen_comparison.json')
    assert len(link['screen_score_sha256'])==30
    assert D.require_confirmation_eligible(0,C.sources()) is None


@pytest.mark.parametrize('entrypoint',['inference','fit','generate'])
def test_public_confirmation_paths_stop_before_cache_or_training(tmp_path,monkeypatch,entrypoint):
    configure_rows(tmp_path,monkeypatch,fresh_score=.1)
    C.B.write(C.ROOT/'screen_comparison.json',COMPARE.report((0,1,2)))
    monkeypatch.setattr(C,'cpu',lambda:None);monkeypatch.setattr(C,'prerequisite',lambda:None)
    monkeypatch.setattr(C,'full_launch_ready',lambda:C.sources())
    def forbidden(*args,**kwargs):raise AssertionError('forbidden payload or worker reached')
    monkeypatch.setattr(C.B.P,'verify_cache',forbidden);monkeypatch.setattr(RUN,'call',forbidden)
    with pytest.raises(ValueError):
        if entrypoint=='inference':RUN.inference('covariance32',3,0)
        elif entrypoint=='fit':RUN.fit('candidate',3,0)
        else:D.run('covariance32',3,0)


def make_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'ROOT',tmp_path/'results');monkeypatch.setattr(C,'MODES',tmp_path/'modes')
    monkeypatch.setattr(C,'RUNS',tmp_path/'runs')
    anchor={'train_donors':[f'train{x}' for x in range(25)],'heldout_donors':[f'held{x}' for x in range(12)]}
    monkeypatch.setattr(C.B.P,'verify_cache',lambda *a,**kw:anchor)
    fit=tmp_path/'fit';hashes={'config.json':'f'*64}
    monkeypatch.setattr(D,'new_fit',lambda *a,**kw:(fit,hashes))
    sources={'synthetic':'metadata'};extra=calibration()
    proof={'passed':True,'sources':sources,'fit':str(fit),'fit_artifacts_sha256':hashes,'fold':0,'seed':0,
        'M':32,'K':256,'train_donors':anchor['train_donors'],'array_fingerprints':fingerprints(),
        'complete_delta':extra['biascal']['delta'],'complete_s':extra['zcal']['s'],
        'state_names':C.B.A.STATES,'state_index':[[0]*32 for _ in anchor['train_donors']],
        'conditions_float32_sha256':'c'*64,'heldout_payloads_opened':False,
        'pruned_archive_sha256':{'direct.npz':'d'*64,'passthrough.npz':'d'*64},
        'cache_files_opened':[str((C.RUNS/'caches/hvg2000/fold0'/n).resolve()) for n in
            ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')]}
    info={'mode_extra':extra,'targets':25,'heldout_targets':0,'M':32,'K':256,'generation_device':'cpu',
        'states_used':C.B.A.STATES,'mode':C.B.W.MODES_BY_NAME['current32'],'fit':str(fit.resolve()),
        'generated_SHA256':'d'*64}
    def save():
        receipts={}
        for name in ('direct','passthrough'):
            path=C.MODES/'orchestration_identity'/f'{name}.receipt.json';C.B.write(path,info)
            receipts[path.name]=C.B.sha(path)
        proof['generation_receipt_sha256']=receipts
        C.B.write(C.ROOT/'control_orchestration_identity.json',proof)
    save();return proof,info,save,sources


def test_full_identity_accepts_negative_and_zero_log_shifts(tmp_path,monkeypatch):
    proof,info,save,sources=make_identity(tmp_path,monkeypatch)
    assert D.verify_identity(sources)['passed']


@pytest.mark.parametrize('change',['shift','joint','cache_scope','fit','archive_set','held_targets','fingerprint'])
def test_full_identity_rejects_control_false_passes(tmp_path,monkeypatch,change):
    proof,info,save,sources=make_identity(tmp_path,monkeypatch)
    if change=='shift':proof['complete_s'][0]=info['mode_extra']['zcal']['s'][0]=7.
    elif change=='joint':info['mode_extra']['zcal']['joint_biascal']=False
    elif change=='cache_scope':proof['cache_files_opened'].append('/other/cache/train/counts.npz')
    elif change=='fit':info['fit']='/other/fit'
    elif change=='archive_set':proof['pruned_archive_sha256']={'other.npz':'d'*64,'extra.npz':'d'*64}
    elif change=='held_targets':info['heldout_targets']=1
    else:proof['array_fingerprints'].pop('genes')
    save()
    with pytest.raises(ValueError):D.verify_identity(sources)


def audit_fixture(tmp_path,monkeypatch,full=False,screen_passes=False,missing=None):
    """Every I/O endpoint below is temporary metadata; no real auditor/cache/model runs."""
    monkeypatch.setattr(C,'ROOT',tmp_path/'results');monkeypatch.setattr(C,'MODES',tmp_path/'modes')
    monkeypatch.setattr(C,'FITS',tmp_path/'fits');monkeypatch.setattr(C,'HERE',tmp_path/'analysis/covariance')
    sources={'synthetic':'metadata'}
    monkeypatch.setattr(C,'sources',lambda:sources);monkeypatch.setattr(C,'full_launch_ready',lambda:sources)
    monkeypatch.setattr(C,'cpu',lambda:None);monkeypatch.setattr(C,'prerequisite',lambda:None)
    monkeypatch.setattr(C,'disk_bytes',lambda:1000)
    monkeypatch.setattr(D,'finalized',lambda *a:True)
    monkeypatch.setattr(D,'new_fit',lambda role,f,s,**kw:(C.fit_path(role,f,s),{}))
    monkeypatch.setattr(D,'verify_identity',lambda sources:{'passed':True})
    monkeypatch.setattr(C.B.P,'verify_cache',lambda *a,**kw:{'synthetic':'cache'})
    original_sha=C.B.sha
    def sha(path):
        if Path(path).name=='default_config.json':return 'fecbfce1e92d31a5d8c3d646f0a6dfb036c975b599dbdc880c332c1023bb3c23'
        if Path(path).name=='microglia.py':return '40d6102e77fa6a4ece2c64cd431db3d907449d7bd69335cb597370d5f860b80e'
        return original_sha(path)
    monkeypatch.setattr(C.B,'sha',sha)
    for name in ('source_manifest.json','training_verification.json','gpu_training_verification.json',
                 'control_orchestration_identity.json'):C.B.write(C.ROOT/name,sources)
    C.B.write(C.HERE/'source_manifest_prelaunch.json',sources)
    keys=COMPARE.job_keys(range(9) if full else range(3))
    def add(key):
        name,f,s=key
        C.B.write(C.ROOT/name/f'fold{f}/seed{s}.json',{'challenge':{'synthetic':True},'pruned_source_sha256':'f'*64})
        C.B.write(C.MODES/name/f'fold{f}/seed{s}.receipt.json',{'generated_SHA256':'f'*64,'array_fingerprints':{}})
    for key in keys:
        if key!=missing:add(key)
    def report(folds,confirmation=False,ready_keys=None):
        wanted=COMPARE.job_keys(folds);ready=AUDIT.snapshot_ready(wanted) if ready_keys is None else wanted&ready_keys
        return {'passes':bool(wanted<=ready and (screen_passes or confirmation)),
            'score_sha256':{str(C.ROOT/n/f'fold{f}/seed{s}.json'):C.B.sha(C.ROOT/n/f'fold{f}/seed{s}.json')
                for n,f,s in ready}}
    monkeypatch.setattr(COMPARE,'report',report)
    if COMPARE.job_keys(range(3))<=AUDIT.snapshot_ready(keys):
        C.B.write(C.ROOT/'screen_comparison.json',report(range(3)))
    if full and missing is None:C.B.write(C.ROOT/'confirmation_comparison.json',report(range(3,9),True))
    return add


def test_audit_rejects_arbitrary_subset_before_any_cache_reads(tmp_path,monkeypatch):
    audit_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):AUDIT.audit((0,))


def test_completed_failed_screen_is_terminal30_artifacts(tmp_path,monkeypatch):
    audit_fixture(tmp_path,monkeypatch)
    result=AUDIT.audit((0,1,2))
    assert result['complete'] and result['study_complete'] and not result['confirmation_eligible']
    assert result['verified_artifacts']==result['required_artifacts']==30


def test_completed_successful_screen_does_not_claim_study_completion(tmp_path,monkeypatch):
    audit_fixture(tmp_path,monkeypatch,screen_passes=True)
    result=AUDIT.audit((0,1,2))
    assert result['complete'] and not result['study_complete'] and result['confirmation_eligible']


def test_full_audit_requires_successful_screen_and90_artifacts(tmp_path,monkeypatch):
    audit_fixture(tmp_path,monkeypatch,full=True,screen_passes=True)
    result=AUDIT.audit(tuple(range(9)))
    assert result['study_complete'] and result['verified_artifacts']==result['required_artifacts']==90


def test_failed_screen_cannot_be_laundered_into_full_audit(tmp_path,monkeypatch):
    audit_fixture(tmp_path,monkeypatch,full=True)
    with pytest.raises(ValueError):AUDIT.audit(tuple(range(9)))


@pytest.mark.parametrize('full',[False,True])
def test_complete_audit_requires_phase_comparison(tmp_path,monkeypatch,full):
    audit_fixture(tmp_path,monkeypatch,full=full,screen_passes=full)
    (C.ROOT/('confirmation_comparison.json' if full else 'screen_comparison.json')).unlink()
    with pytest.raises(ValueError):AUDIT.audit(tuple(range(9)) if full else (0,1,2))


@pytest.mark.parametrize('tree',['score','mode','unpruned_mode','fit','ineligible_confirmation'])
def test_complete_audit_rejects_orphans(tmp_path,monkeypatch,tree):
    audit_fixture(tmp_path,monkeypatch)
    if tree=='score':path=C.ROOT/'unexpected/fold0/seed0.json'
    elif tree=='mode':path=C.MODES/'current32/fold0/seed9.receipt.json'
    elif tree=='unpruned_mode':path=C.MODES/'current32/fold0/seed0.npz'
    elif tree=='fit':path=C.FITS/'candidate/unplanned/fold0/final.pt'
    else:path=C.ROOT/'covariance32/fold3/seed0.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    with pytest.raises(ValueError):AUDIT.audit((0,1,2))


def test_late_finalized_row_stays_missing_in_audit_snapshot(tmp_path,monkeypatch):
    missing=('covariance32',2,1);add=audit_fixture(tmp_path,monkeypatch,missing=missing)
    def cache(*args,**kwargs):
        add(missing);return {'synthetic':'cache'}
    monkeypatch.setattr(C.B.P,'verify_cache',cache)
    result=AUDIT.audit((0,1,2),require_complete=False)
    assert not result['complete'] and not result['study_complete'] and not result['confirmation_eligible']
    assert result['verified_artifacts']==29 and result['missing']==[list(missing)]


def test_partial_tree_accepts_planned_npz_name_not_receipt_npz(tmp_path,monkeypatch):
    missing=('covariance32',2,1);audit_fixture(tmp_path,monkeypatch,missing=missing)
    path=C.MODES/'covariance32/fold2/seed1.npz';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'synthetic unfinalized archive')
    assert not AUDIT.audit((0,1,2),require_complete=False)['complete']
    path.rename(path.with_name('seed1.receipt.npz'))
    with pytest.raises(ValueError):AUDIT.audit((0,1,2),require_complete=False)
