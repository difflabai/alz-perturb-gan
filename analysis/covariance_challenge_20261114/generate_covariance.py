"""Matched inference, full real passthrough identity, and verifiable pruning."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import sys
import time
import cov_common as C
import resource_common as U
import covmom_gpu  # explicit registration of the prospectively amended training arm
B=C.B;np=C.np;G=C.G
META=('config.json','fit_receipt.json','study_training_receipt.json','checkpoint_receipt.json','learning_curve.jsonl')
WEIGHTS=('final.pt',*(f'snap_{u}.pt' for u in C.T.snapshot_updates(20000)))


def serialized_equal(left,right):
    return json.loads(json.dumps(left,allow_nan=False))==json.loads(json.dumps(right,allow_nan=False))


def valid_affinity(value):
    return (isinstance(value,list) and 1<=len(value)<=3 and
            all(type(x) is int and x>=0 for x in value) and value==sorted(set(value)))


def resolved_training_spec(role):
    """Resolve native scalar options/layout without opening a cache or weight."""
    B.require(role in ('control','candidate'),'unknown training role')
    flags=C.T.parse_arm('pbulk' if role=='control' else C.ARM)
    options=C.T.arms.train_options(flags);rec=C.T.recipe(options)
    columns=[{'flag':None,'name':'released','columns':[0,150],'cell_dim':None}]
    if role=='candidate':
        columns.append({'flag':'d.covmom','name':'signed_covariance','columns':[150,2650],'cell_dim':None})
    start=150 if role=='control' else 2650
    columns.append({'flag':'pbulk','name':'pb','columns':[start,start+2000],'cell_dim':None})
    rd=rec.to_dict()|{k:getattr(rec,k) for k in (
        'lr_schedule','lr_anneal_start','lr_floor','network_lr_floor','network_lr_horizon_cap',
        'ema_decay','total_steps','batch_size','num_particles','z_dim','lr','d_lr_mult','prior_lr_mult')}
    return json.loads(json.dumps({
        'study':C.T.STUDY,'round':C.T.ROUND,'checkpoint_every':C.T.CHECKPOINT_EVERY,
        'folds_file':str(C.T.FOLDS_JSON_R2 if C.T.FOLDS_JSON_R2.exists() else C.T.FOLDS_JSON_R1),
        'recipe':rec.to_dict(),'hyperparameters':dict(C.T.T005),'training_options':options.as_dict(),
        'learning_rates':{'generator':rec.lr,'critic':rec.lr*rec.d_lr_mult,'particle':rec.lr*rec.prior_lr_mult,
            'schedule':{'lr_schedule':rec.lr_schedule,'lr_anneal_start':rec.lr_anneal_start,'lr_floor':rec.lr_floor,
                'network_lr_floor':rec.resolved_network_lr_floor,'network_lr_horizon_cap':rec.network_lr_horizon_cap}},
        'shadow_ema_decays':list(C.T.SHADOW_EMA_DECAYS),'native_ema_decay':rec.ema_decay,
        'default_generation_state':'raw','critic_module':None,'generator_kwargs':{},
        'critic_layout':{'n_features':2150 if role=='control' else 4650,'cond_dim':7,'hidden':128,
            'batch':32,'cells':128,'blocks':columns},
        'train_departures':C.T.arms.train_departures(flags,rd)}))


def validate_resolved_config(cfg,role):
    spec=resolved_training_spec(role)
    for key,value in spec.items():
        if key!='train_departures':B.require(cfg[key]==value,f'resolved training {key} differs')
    B.require(set(cfg['r2_arms'])==({'pbulk'} if role=='control' else {'pbulk','d.covmom'})
        and cfg['r2_arms']['pbulk']['name']=='pbulk' and cfg['r2_arms']['pbulk']['value'] is None,
        'resolved plug-in set differs')
    departures=cfg['departures'];expected=dict(spec['train_departures'])
    if role=='candidate':
        info=cfg['r2_arms']['d.covmom']
        B.require(info['name']=='d.covmom' and info['value'] is None,'covariance plug-in identity differs')
        expected['d.covmom']={'kind':'critic architecture','value':None,'info':info['info']}
        cov=info['info']
        B.require(cov['features']==2500 and cov['PCA_offdiagonal']==1225
            and cov['random_upper_including_diagonal']==1275 and cov['cross_space_terms'] is False
            and cov['covariance_divisor']=='cells-1' and cov['reference_bags_per_donor']==16
            and cov['reference_cells_per_bag']==128 and cov['reference_donors']==len(cfg['train_donors'])
            and cov['reference_weighting']=='equal donors' and cov['total_feature_width']==4650
            and cov['native_kappa']==C.KAPPA and cov['random_basis_seed']==20261113
            and cov['random_sign_canonicalization'] is False and B.valid_digest(cov['reference_indices_sha256'])
            and np.isfinite(cov['scale']) and cov['scale']>0,'covariance reference specification differs')
        energies=cov['energies']
        B.require(all(np.isfinite(v) and v>=0 for v in energies.values())
            and energies['released150_reference_energy']>0
            and energies['covmom_centered_reference_energy']>0,'invalid covariance reference energies')
    B.require(departures==expected,'declared native departures differ')


def require_confirmation_eligible(fold,sources):
    """Public training/inference guard; only completed frozen screen can admit extra folds."""
    B.require(fold in range(9) and sources==C.sources(),'invalid fold or confirmation sources')
    if fold<3:return None
    import compare_covariance as COMPARE
    live=COMPARE.report(tuple(range(3)))
    saved_path=C.ROOT/'screen_comparison.json'
    B.require(saved_path.exists() and live['passes'] is True and len(live['score_sha256'])==30
        and serialized_equal(B.P.read(saved_path),live),'confirmation requires exact successful stored screen')
    return {'screen_comparison_sha256':B.sha(saved_path),'screen_score_sha256':live['score_sha256']}


def new_fit(role,fold,seed,allow_pruned=False):
    fit=C.fit_path(role,fold,seed);cfg=B.P.read(fit/'config.json')
    record=B.P.read(fit/'study_training_receipt.json');receipt=B.P.read(fit/'fit_receipt.json')
    checkpoint=B.P.read(fit/'checkpoint_receipt.json');anchor=B.P.verify_cache(fold,payloads=False)
    flags=C.T.parse_arm('pbulk' if role=='control' else C.ARM)
    B.require(cfg['flags']==list(flags) and cfg['arm']==C.T.arm_name(flags) and cfg['fold']==fold and cfg['seed']==seed
        and cfg['panel']=='hvg2000' and cfg['n_genes']==2000 and cfg['updates']==20000 and cfg['device']==U.TRAIN_DEVICE
        and cfg['recipe']==json.loads(json.dumps(C.T.recipe(C.T.arms.train_options(flags)).to_dict()))
        and cfg['train_donors']==anchor['train_donors'] and cfg['heldout_donors']==anchor['heldout_donors']
        and cfg['cache_manifest_SHA256']==anchor['manifest_sha256']
        and cfg['cache']==str((C.RUNS/f'caches/hvg2000/fold{fold}').resolve()) and cfg['protocol_split'] is True
        and cfg['batch']==[32,128] and cfg['stream_root']==C.T.STREAM_ROOT and cfg['recipe_total_steps']==20000
        and cfg['particlegan_version']==C.T.VERSION and cfg['native_recipe']=='bispf_optimized'
        and cfg['critic_features']==(2150 if role=='control' else 4650),'new fit config differs')
    validate_resolved_config(cfg,role)
    B.require(cfg['snapshot_updates']==C.T.snapshot_updates(20000),'native snapshot schedule differs')
    B.require(record['status']=='native_training_completed_TRAIN_only' and record['role']==role
        and record['fold']==fold and record['seed']==seed and record['fit']==str(fit)
        and record['sources']==C.sources() and record['automatic_generation_deferred'] is True
        and record['cache_manifest_sha256']==anchor['manifest_sha256']
        and record['fit_receipt_sha256']==B.sha(fit/'fit_receipt.json')
        and valid_affinity(record['cpu_affinity']) and 0<=record['peak_logical_task_bytes']<3000000000,'new training lineage differs')
    gpu=record['gpu_resource']
    B.require(gpu['physical_selector']=='1' and gpu['memory_fraction']==.04
        and isinstance(gpu['device_name'],str) and bool(gpu['device_name'].strip())
        and gpu['cublas_workspace_config']==':4096:8' and gpu['deterministic_algorithms'] is True
        and gpu['allow_tf32'] is False and gpu['cudnn_deterministic'] is True and gpu['cudnn_benchmark'] is False
        and type(gpu['total_device_bytes']) is int and gpu['total_device_bytes']>0
        and all(type(gpu[k]) is int and 0<=gpu[k]<=gpu['total_device_bytes']*.04
                for k in ('max_allocated_bytes','max_reserved_bytes')),
        'matched GPU resource declaration differs')
    if fold>=3:validate_confirmation_link(record['confirmation_eligibility'])
    permitted={str((C.RUNS/f'caches/hvg2000/fold{fold}'/n).resolve())
               for n in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')}
    B.require(set(record['cache_files_opened'])==permitted,'new training cache scope differs')
    B.require(receipt['status']=='training_completed' and receipt['completed_updates']==receipt['endpoint']==20000
        and receipt['full_20k_endpoint'] is True and receipt['full_horizon_endpoint'] is True
        and receipt['device']==U.TRAIN_DEVICE and receipt['threads']==3 and not receipt['snapshots_missing']
        and checkpoint['status']=='completed' and checkpoint['completed_updates']==20000
        and checkpoint['path']=='final.pt' and checkpoint['SHA256']==receipt['final_SHA256'],
        'new native endpoint/receipt differs')
    weights=record['weight_sha256']
    B.require(set(weights)==set(WEIGHTS) and all(B.valid_digest(h) for h in weights.values())
        and set(receipt['snapshots_SHA256'])=={str(u) for u in C.T.snapshot_updates(20000)}
        and weights['final.pt']==receipt['final_SHA256']
        and all(weights[f'snap_{u}.pt']==receipt['snapshots_SHA256'][str(u)] for u in C.T.snapshot_updates(20000)),
        'new checkpoint lineage differs')
    metadata={n:B.sha(fit/n) for n in META};expected={**metadata,**weights}
    pruned=fit/'control_weight_pruning.json'
    if pruned.exists():
        proof=B.P.read(pruned)
        B.require(role=='control' and allow_pruned and proof['status']=='verified_and_pruned_control_weights'
            and proof['sources']==C.sources() and proof['fit_artifacts_sha256']==expected
            and proof['weight_sha256']==weights and not any((fit/n).exists() for n in WEIGHTS),
            'control pruning proof differs')
        B.require(proof['score_sha256']==B.sha(C.ROOT/'training_control32'/f'fold{fold}/seed{seed}.json'),
                  'control pruning score link differs')
    else:B.require(all(B.sha(fit/n)==h for n,h in weights.items()),'new weight file changed')
    return fit,expected


def recipe_fit(recipe,fold,seed,allow_pruned=False):
    if recipe in ('original32','current32'):
        return C.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}',B.P.verify_fit(fold,seed)
    if recipe in ('training_control32','covariance32'):
        return new_fit('control' if recipe=='training_control32' else 'candidate',fold,seed,allow_pruned)
    B.require(recipe=='empirical32','unknown recipe');return None,{}


def generation_call(fit,cache,output,mode):
    return G.generate_fit(fit,cache,mode=mode,out=output,device='cpu')


@contextmanager
def TRAIN_targets():
    old=G.targets;G.targets=B.ZC.train_targets
    try:yield
    finally:G.targets=old


def validate_calibration(extra,n_train,original=False):
    bias=extra['biascal']
    B.require(bias['n_train_targets']==n_train and bias['states']==sorted(B.A.STATES)
        and np.asarray(bias['delta']).shape==(2000,) and np.isfinite(bias['delta']).all(),
        'bias calibration scope differs')
    if not original:
        z=extra['zcal'];shifts=np.asarray(z['s'])
        B.require(extra['cellshape']['alpha']==.5 and z['n_train_targets']==n_train
            and z['states']==sorted(B.A.STATES) and shifts.shape==(2000,) and np.isfinite(shifts).all()
            and (np.abs(shifts)<=B.ZC.S_MAX).all() and z['joint_biascal'] is True,
            'cellshape/log-dispersion calibration scope differs')


def validate_intervals(row):
    for group in ('train','heldout'):
        for construction in ('percentile','rank'):
            for summary in ('mean','sd','combined'):
                block=row['intervals'][group][construction][summary]
                B.require(all(np.isfinite(block[k]) for k in ('coverage','below','above','width','interval_score'))
                    and all(0<=block[k]<=1 for k in ('coverage','below','above'))
                    and block['width']>=0 and block['interval_score']+1e-12>=block['width']
                    and abs(block['coverage']+block['below']+block['above']-1)<1e-10,
                    'invalid interval metric or miss partition')


def validate_confirmation_link(link):
    screen=C.ROOT/'screen_comparison.json';saved=B.P.read(screen)
    B.require(saved['passes'] is True and link=={'screen_comparison_sha256':B.sha(screen),
        'screen_score_sha256':saved['score_sha256']},'confirmation lineage differs')


def verify_identity(sources):
    p=C.ROOT/'control_orchestration_identity.json';proof=B.P.read(p)
    fit,hashes=new_fit('control',0,0,allow_pruned=True)
    anchor=B.P.verify_cache(0,payloads=False)
    B.validate_fingerprints(proof['array_fingerprints'])
    B.require(proof['passed'] is True and proof['sources']==sources and proof['fit']==str(fit)
        and proof['fit_artifacts_sha256']==hashes and proof['fold']==proof['seed']==0
        and proof['M']==32 and proof['K']==256 and len(proof['complete_delta'])==len(proof['complete_s'])==2000
        and proof['train_donors']==anchor['train_donors'] and np.isfinite(proof['complete_delta']).all()
        and np.isfinite(proof['complete_s']).all() and (np.abs(np.asarray(proof['complete_s']))<=B.ZC.S_MAX).all()
        and proof['state_names']==B.A.STATES and proof['heldout_payloads_opened'] is False
        and set(proof['pruned_archive_sha256'])=={'direct.npz','passthrough.npz'}
        and all(B.valid_digest(h) for h in proof['pruned_archive_sha256'].values())
        and not any((C.MODES/'orchestration_identity').glob('*.npz')),
        'full real passthrough calibration identity not proven')
    B.require(set(proof['generation_receipt_sha256'])=={'direct.receipt.json','passthrough.receipt.json'},
              'wrong identity receipt set')
    indices=np.asarray(proof['state_index'])
    B.require(indices.shape==(len(anchor['train_donors']),32) and indices.dtype.kind in 'iu'
        and ((indices>=0)&(indices<6)).all() and B.valid_digest(proof['conditions_float32_sha256']),
        'identity allocation shape/domain differs')
    permitted={str((C.RUNS/'caches/hvg2000/fold0'/n).resolve())
               for n in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')}
    B.require(set(proof['cache_files_opened'])==permitted,'identity TRAIN read scope differs')
    extras=[]
    for name,digest in proof['generation_receipt_sha256'].items():
        path=C.MODES/'orchestration_identity'/name
        info=B.P.read(path)
        validate_calibration(info['mode_extra'],len(anchor['train_donors']))
        B.require(B.sha(path)==digest and info['mode_extra']['biascal']['delta']==proof['complete_delta']
            and info['mode_extra']['zcal']['s']==proof['complete_s']
            and info['targets']==len(anchor['train_donors']) and info['heldout_targets']==0
            and info['M']==32 and info['K']==256 and info['generation_device']=='cpu'
            and info['states_used']==B.A.STATES and info['mode']==B.W.MODES_BY_NAME['current32']
            and info['fit']==str(fit.resolve())
            and info['generated_SHA256']==proof['pruned_archive_sha256'][name.replace('.receipt.json','.npz')],
            'identity calibration receipt link differs')
        extras.append(info['mode_extra'])
    B.require(extras[0]==extras[1],'complete fresh joint calibration metadata differs')
    return proof


def identity():
    C.cpu();sources=C.full_launch_ready();C.prerequisite();G.M=32
    if (C.ROOT/'control_orchestration_identity.json').exists():verify_identity(sources);return
    opened=set();cache,anchor=C.train_scope(0,opened)
    B.require(C.disk_bytes()<2850000000,'less than150MB full identity headroom')
    fit,hashes=new_fit('control',0,0);G.generation_streams=lambda f,j:C.streams(f,j,0)
    root=C.MODES/'orchestration_identity';root.mkdir(parents=True,exist_ok=True)
    B.require(not any(p.is_file() for p in root.rglob('*')),'partial identity files; inspect before resuming')
    outputs=[root/'direct.npz',root/'passthrough.npz'];mode=B.W.MODES_BY_NAME['current32']
    with TRAIN_targets():
        _,direct=G.generate_fit(fit,cache,mode=mode,out=outputs[0],device='cpu')
        _,wrapped=generation_call(fit,cache,outputs[1],mode)
    fingerprints=[B.archive_fingerprints(p) for p in outputs]
    B.require(fingerprints[0]==fingerprints[1] and direct['mode_extra']==wrapped['mode_extra'],
              'fresh full joint calibration/generation passthrough differs')
    with np.load(outputs[0],allow_pickle=False) as left,np.load(outputs[1],allow_pickle=False) as right:
        B.require(left['targets'].tolist()==right['targets'].tolist()==anchor['train_donors']
            and not left['is_heldout'].any() and all(np.array_equal(left[k],right[k])
                for k in ('state_names','state_index','conditions')),'identity target/allocation differs')
        allocation={'state_names':left['state_names'].tolist(),'state_index':left['state_index'].tolist(),
                    'conditions_float32_sha256':hashlib.sha256(np.ascontiguousarray(left['conditions']).tobytes()).hexdigest()}
    archives={p.name:B.sha(p) for p in outputs}
    receipts={p.with_suffix('.receipt.json').name:B.sha(p.with_suffix('.receipt.json')) for p in outputs}
    for p in outputs:p.unlink()
    C.check_train_scope(cache,opened)
    B.write(C.ROOT/'control_orchestration_identity.json',{'passed':True,'sources':sources,'fit':str(fit),
        'fit_artifacts_sha256':hashes,'fold':0,'seed':0,'M':32,'K':256,'train_donors':anchor['train_donors'],
        'array_fingerprints':fingerprints[0],'complete_delta':direct['mode_extra']['biascal']['delta'],
        'complete_s':direct['mode_extra']['zcal']['s'],**allocation,'pruned_archive_sha256':archives,
        'generation_receipt_sha256':receipts,
        'cache_files_opened':sorted(opened),'heldout_payloads_opened':False,
        'scope':'Full freshly refitted TRAIN joint calibration and real inference orchestration identity, not architecture equivalence'})
    B.require(C.sources()==sources and C.disk_bytes()<3000000000,'identity source/cap differs');verify_identity(sources)


def finalized(recipe,fold,seed,sources):
    B.require(recipe in C.RECIPES and fold in range(9) and seed in (0,1),'invalid scored job')
    score=C.ROOT/recipe/f'fold{fold}/seed{seed}.json'
    if not score.exists():return False
    row=B.P.read(score);path=C.MODES/recipe/f'fold{fold}/seed{seed}.npz';receipt=path.with_suffix('.receipt.json')
    info=B.P.read(receipt);meta=row['challenge'];cache=C.RUNS/f'caches/hvg2000/fold{fold}'
    B.require(meta=={'recipe':recipe,'fold':fold,'replicate':seed,'sources':sources,'receipt_sha256':B.sha(receipt),
        'generation_device':'cpu','K':256,'PC_CELLS':64},'scored identity differs')
    B.require(info['challenge_recipe']==recipe and info['computational_replicate']==seed and info['sources']==sources
        and info['generation_streams']==C.NAMESPACE and info['generation_device']=='cpu' and info['targets']==37
        and row['fold']==fold and row['M']==info['M']==32 and info['K']==256
        and row['pruned_source_sha256']==row['fingerprint']['generated_sha256']==info['generated_SHA256']
        and info['cache_manifest_sha256']==row['fingerprint']['cache_sha256']==B.sha(cache/'manifest.json')
        and not path.exists() and valid_affinity(info['cpu_affinity'])
        and B.valid_digest(info['generated_SHA256']),'receipt/cache/pruning differs')
    B.require(row['source']==str(path.resolve())
        and row['fingerprint']['score_code_sha256']==sources[str(B.W.R5/'architecture_score.py')]
        and row['fingerprint']['interval_code_sha256']==sources[str(B.W.R5/'interval_metrics.py')],
        'score source/math fingerprint differs')
    validate_intervals(row)
    B.require(all(np.isfinite(v) for v in row['historical'].values())
        and row['historical']['pc_spread_rep']>0,'nonfinite historical metric or invalid spread')
    B.require(info['runtime']['particlegan']==str(C.T.VERSION),'runtime ParticleGAN version differs')
    B.validate_fingerprints(info['array_fingerprints'])
    fit,hashes=recipe_fit(recipe,fold,seed,allow_pruned=True)
    B.require(info['fit_artifacts_sha256']==hashes and (fit is None or info['fit']==str(fit)),'fit lineage differs')
    anchor=B.P.verify_cache(fold,payloads=False)
    if fold>=3:validate_confirmation_link(info['confirmation_eligibility'])
    if recipe=='empirical32':
        B.require(info['baseline']=='resample_blind' and info['empirical_role']=='reference'
            and info['empirical_stream']==seed and info['kpc']==64
            and np.isfinite(info['frac_cells_depth_unmatched']) and 0<=info['frac_cells_depth_unmatched']<=1
            and info['source_selection']=='training donors only; target excluded when in training',
            'empirical source/stream differs')
    else:
        expected_mode=B.W.MODES_BY_NAME['original32' if recipe=='original32' else 'current32']
        B.require(info['mode']==expected_mode and info['states_used']==B.A.STATES
            and info['heldout_targets']==len(anchor['heldout_donors']),'wrong model readout/states/targets')
        B.A.allocation(info['state_stage_allocation'],B.A.cache_metadata(fold,anchor),32,recipe)
        validate_calibration(info['mode_extra'],len(anchor['train_donors']),recipe=='original32')
    if recipe=='covariance32':
        verify_identity(sources)
        B.require(info['orchestration_identity_sha256']==B.sha(C.ROOT/'control_orchestration_identity.json'),
                  'candidate identity link differs')
    return True


def run(recipe,fold,seed):
    C.cpu();sources=C.full_launch_ready();C.prerequisite();G.M=32
    B.require(recipe in C.RECIPES and fold in range(9) and seed in (0,1),'invalid job')
    eligibility=require_confirmation_eligible(fold,sources)
    B.P.verify_cache(fold)
    if finalized(recipe,fold,seed,sources):return
    fit,hashes=recipe_fit(recipe,fold,seed)
    if recipe=='covariance32':verify_identity(sources)
    output=C.MODES/recipe/f'fold{fold}/seed{seed}.npz';score=C.ROOT/recipe/f'fold{fold}/seed{seed}.json'
    B.require(not score.exists() and not any(output.parent.glob(output.stem+'*')),'partial outputs; inspect before resuming')
    B.require(C.disk_bytes()<2850000000,'less than150MB generation headroom');output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();cache=C.RUNS/f'caches/hvg2000/fold{fold}'
    G.generation_streams=lambda f,j:C.streams(f,j,seed)
    if recipe=='empirical32':
        real=B.AS.E.load_real(cache);B.BL._target_rng=lambda r,t:C.empirical_rng(r,t,seed)
        arrays,info=B.BL.generate('resample_blind',real,M=32,K=256,kpc=64)
        B.AS.E.save_npz(output,**arrays)
        info.update(baseline='resample_blind',empirical_role='reference',empirical_stream=seed,
            source_selection='training donors only; target excluded when in training',targets=37)
    else:
        _,info=generation_call(fit,cache,output,B.W.MODES_BY_NAME['original32' if recipe=='original32' else 'current32'])
        info.update(state_stage_allocation=B.W.allocation(output))
    info.update(challenge_recipe=recipe,computational_replicate=seed,generation_streams=C.NAMESPACE,
        sources=sources,fit_artifacts_sha256=hashes,generated_SHA256=B.sha(output),generation_device='cpu',
        seconds_total=time.monotonic()-started,cache_manifest_sha256=B.sha(cache/'manifest.json'),
        array_fingerprints=B.archive_fingerprints(output),cpu_affinity=sorted(os.sched_getaffinity(0)),
        runtime={'python':sys.version,'numpy':np.__version__,'torch':str(C.torch.__version__),'particlegan':str(C.T.VERSION)})
    if recipe=='covariance32':info['orchestration_identity_sha256']=B.sha(C.ROOT/'control_orchestration_identity.json')
    if eligibility is not None:info['confirmation_eligibility']=eligibility
    B.write(output.with_suffix('.receipt.json'),info);B.AS.score(output,cache,score)
    B.require(C.sources()==sources,'sources changed during inference/scoring')
    row=B.P.read(score);row['pruned_source_sha256']=row['fingerprint']['generated_sha256']
    row['challenge']={'recipe':recipe,'fold':fold,'replicate':seed,'sources':sources,
        'receipt_sha256':B.sha(output.with_suffix('.receipt.json')),'generation_device':'cpu','K':256,'PC_CELLS':64}
    B.write(score,row);output.unlink()
    B.require(finalized(recipe,fold,seed,sources) and C.disk_bytes()<3000000000,'inference did not finalize within cap')
    print(B.H.json.dumps({'recipe':recipe,'fold':fold,'replicate':seed}),flush=True)


def prune_control(fold,seed):
    C.cpu();sources=C.full_launch_ready()
    require_confirmation_eligible(fold,sources)
    B.require(finalized('training_control32',fold,seed,sources),'control score must finalize before weight pruning')
    if (fold,seed)==(0,0):verify_identity(sources)
    fit,hashes=new_fit('control',fold,seed,allow_pruned=True)
    path=fit/'control_weight_pruning.json'
    if path.exists():return
    record=B.P.read(fit/'study_training_receipt.json')
    B.write(path,{'status':'verified_and_pruned_control_weights','sources':sources,'fit_artifacts_sha256':hashes,
        'weight_sha256':record['weight_sha256'],'score_sha256':B.sha(C.ROOT/'training_control32'/f'fold{fold}/seed{seed}.json'),
        'scope':'Weights verified immediately before pruning; later verification uses this receipt, not rehashing absent weights'})
    for name in WEIGHTS:(fit/name).unlink()
    new_fit('control',fold,seed,allow_pruned=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',choices=(*C.RECIPES,'identity','prune_control'),required=True)
    p.add_argument('--fold',type=int,choices=range(9),default=0);p.add_argument('--replicate',type=int,choices=(0,1),default=0)
    a=p.parse_args()
    with C.leases():
        if a.recipe=='identity':identity()
        elif a.recipe=='prune_control':prune_control(a.fold,a.replicate)
        else:run(a.recipe,a.fold,a.replicate)
