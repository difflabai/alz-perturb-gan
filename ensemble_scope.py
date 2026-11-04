"""Standard-library-only exact scope for two retained pure GAN parents."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/alz')
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
OBS={'enabled':True,'cell_hidden':64,'cell_features':25,'feature_scale':.2,'pool_epsilon':1e-6,'layernorm_epsilon':1e-5}

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def manifest():
    q=read(ROOT/'ensemble_parents.json')
    assert q['schema']==1 and q['fold']==q['training_seed']==0 and q['native_recipe']=='bispf_optimized'
    assert q['parent_completed_updates']==20000 and q['parent_training_updates_total']==40000 and q['real_bag_replacements']==0
    assert q['mask_seed']==20261108 and q['candidate_learned_weight']==.25 and q['replicates']==[0,1] and len(q['parents'])==2
    specs=[('PG8','variant','CAL_GPU0_V1_latent8_v1_f0_pg_s0','444c22a5829b79d8ed0cd300be255d9410916d6c2248f930cf4cc0f886744da2'),
           ('learned_observation','learned','CAL_GPU0_V1_learned_obs_v1_f0_pg_s0','9e59988d216ef5f6d5fabf798ba204898a4d71905906938d486147e893b4df11')]
    for parent,(name,adapter,attempt,pin) in zip(q['parents'],specs):
        assert parent=={'name':name,'adapter':adapter,'checkpoint_path':str(ROOT/'runs'/attempt/'fit/final.pt'),
                        'checkpoint_SHA256':pin,'completed_updates':20000,'variant_config':CFG,
                        'score_output':str(ROOT/'runs'/attempt),'learned_observation_config':None if adapter=='variant' else OBS}
    return q

def reference_payload(weight):
    assert type(weight) in (float,int) and weight in (0.,.25,1.)
    q=manifest()
    return {'schema':1,'type':'pure_ensemble_complete_parent_references','status':'completed','fold':0,'seed':0,
            'completed_updates':40000,'fixed_endpoint':40000,'parent_completed_updates_each':20000,'new_model_updates':0,
            'learned_weight':weight,'mask_seed':20261108,'parents':q['parents'],
            'parent_manifest_SHA256':sha(ROOT/'ensemble_parents.json'),'protocol_SHA256':sha(ROOT/'ensemble_protocol.json'),
            'cpu_qualification_SHA256':sha(ROOT/'analysis/ensemble_cpu_qualification.json'),
            'parent_audit_SHA256':sha(ROOT/'analysis/ensemble_parent_checkpoint_audit.json')}

def verify_reference(candidate):
    assert candidate['checkpoint_SHA256']==sha(candidate['checkpoint_path'])
    assert read(candidate['checkpoint_path'])==reference_payload(candidate['ensemble_config']['learned_weight'])
    assert candidate['ensemble_config']=={'learned_weight':candidate['ensemble_config']['learned_weight'],'mask_seed':20261108,'bags_per_stage':200}
    for parent in manifest()['parents']:
        assert sha(parent['checkpoint_path'])==parent['checkpoint_SHA256']

def validate_authorization(auth):
    q=manifest();pins=auth['files_SHA256'];operation=auth['operation'];weight=auth['ensemble_weight']
    assert auth['cap_seconds']==900 and auth['fixed_endpoint']==40000 and auth['cell']['fold']==auth['cell']['seed']==0
    assert auth.get('hybrid_config') is None and auth['variant_config']==CFG and auth['pure_formulation']=='pg8_learned_ensemble75'
    assert type(weight) in (int,float) and weight in (0.,.25,1.)
    required=[ROOT/'ensemble_parents.json',ROOT/'ensemble_protocol.json',ROOT/'ensemble_plan.json',
              ROOT/'analysis/ensemble_cpu_qualification.json',ROOT/'analysis/ensemble_parent_checkpoint_audit.json']
    assert auth['ensemble_manifest_SHA256']==sha(required[0])
    for path in required:assert pins.get(str(path))==sha(path)
    plan=read(ROOT/'ensemble_plan.json');assert set(plan)=={'enabled','learned_weight','mask_seed'} and type(plan['enabled']) is bool and plan['learned_weight']==.25 and plan['mask_seed']==20261108
    proof=read(ROOT/'analysis/ensemble_cpu_qualification.json');audit=read(ROOT/'analysis/ensemble_parent_checkpoint_audit.json')
    assert proof['status']=='PASSED_SYNTHETIC_CPU_PURE_ENSEMBLE_COMPOSITION' and proof['torch_imported'] is False
    assert all(proof[k] for k in ['exact_whole_bag_mask_and_count_verified','zero_one_control_data_and_dtype_bitwise_replay','sequential_global_RNG_branch_isolation_verified','malformed_incomplete_and_deadline_refusal_before_yield'])
    assert audit['status']=='VERIFIED_COMPLETE_PURE_ENSEMBLE_PARENT_CHECKPOINTS' and audit['parent_manifest_SHA256']==sha(required[0]) and audit['CUDA_initialized'] is False
    for evidence in (proof,audit):
        for path,pin in evidence['source_SHA256'].items():assert pins.get(path)==pin==sha(path)
    for index,parent in enumerate(q['parents']):
        p=Path(parent['checkpoint_path']);receipt=p.with_name('checkpoint_receipt.json');r=read(receipt)
        assert pins.get(str(p))==parent['checkpoint_SHA256']==sha(p) and pins.get(str(receipt))==sha(receipt)
        assert r['status']=='completed' and r['completed_updates']==20000 and r['path']==p.name and r['SHA256']==parent['checkpoint_SHA256']
        assert audit['parents'][index]['checkpoint_SHA256']==parent['checkpoint_SHA256'] and audit['parents'][index]['exact_official_recipe_and_complete_G_D_prior_load']
        assert audit['parents'][index]['finite_model_floating_values']==[371177,376066][index]
    if operation=='qualify_pure_ensemble':
        assert weight in (0.,1.) and auth['qualification_only'] is True
    else:
        assert operation=='evaluate_pure_ensemble' and weight==.25 and auth['qualification_only'] is False and plan['enabled'] is True
        for label in ('pg8','learned'):
            path=ROOT/'analysis'/f'ensemble_control_{label}_verification.json';control=read(path)
            assert pins.get(str(path))==sha(path) and control['status']=='VERIFIED_NATIVE_PURE_ENSEMBLE_PARENT_CONTROL'
            assert control['control_weight']==(0. if label=='pg8' else 1.) and control['parent_manifest_SHA256']==sha(required[0])
            assert control['complete_clean_settlement_verified'] and control['arrays_and_integer_exemplars_bitwise_equal']
            assert control['qualified_execution_source_SHA256']==auth['research_source_SHA256']
            assert control['cpu_qualification_SHA256']==sha(ROOT/'analysis/ensemble_cpu_qualification.json')
            for p,pin in control['verification_source_SHA256'].items():assert pins.get(p)==pin==sha(p)
