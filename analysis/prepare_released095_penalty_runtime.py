"""Generate one disabled, source-bound penalty trial after audited bridge rejection."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path('/ml2/alz')
TEMPLATE=ROOT/'released_v095_bridge_runtime'
PROTOCOL=ROOT/'analysis/released_penalty_v095/protocol.json'
PENALTY_PROOF=PROTOCOL.with_name('CPU_qualification.json')
REFERENCE=ROOT/'analysis/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0-result.json'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assignment(source,name,value):
    source,n=re.subn('^'+name+'=.*$',lambda _:name+'='+value,source,flags=re.M)
    assert n==1,name
    return source


def make(identity):
    reference=read(REFERENCE)
    assert reference['owned_release_and_clean_settlement_verified'] is True
    assert reference['official095_origin_version_and_continuous_domain_verified'] is True
    assert reference['candidate']['completed_updates']==reference['candidate']['fixed_endpoint']==20000
    assert reference['candidate']['particlegan_version']=='0.9.5'
    assert reference['decision']['accept'] is False
    assert read(TEMPLATE/'plan.json')['enabled'] is False,'Commit negative evidence and actually revert the bridge activation first'
    qualified=read(ROOT/'analysis/released_v095_bridge_runtime_CPU_qualification.json')
    assert qualified['tests_passed']==25
    for p,h in qualified['source_SHA256'].items():assert sha(p)==h
    protocol=read(PROTOCOL);settings=[s for s in protocol['stage1']['configs'] if s['id']==identity]
    assert len(settings)==1
    setting=settings[0];penalty=read(PENALTY_PROOF)
    assert penalty['configurations_checked']==12 and penalty['actual_optimizer_pairs_executed']==36
    for p,h in penalty['source_SHA256'].items():assert sha(p)==h
    row=[x for x in penalty['checks'] if x['candidate_id']==identity]
    assert len(row)==1 and row[0]['parameters']==setting['parameters'] and row[0]['complete_next_pair_state_byte_exact']
    field=setting['changed_field'];label=identity.rsplit('_m',1)[1]
    suffix={'reg_coeff':'regcoeff','reg_kappa':'regkappa','prior_reg':'priorreg'}[field]+label
    experiment='released_v095_'+suffix;folder=ROOT/(experiment+'_runtime')
    assert not folder.exists()
    observation=ROOT/'analysis'/(experiment+'_initial_resource_observation.json')
    assert not observation.exists()
    query=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    observation.write_text(json.dumps({'status':'READ_ONLY_GPU0_SNAPSHOT_NOT_ADMISSION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu_uuid':query[0].strip(),'GPU_total_bytes':int(query[1])*2**20,'GPU_used_bytes':int(query[2])*2**20,'GPU_free_bytes':int(query[3])*2**20,
        'host_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,
        'memory_pressure':Path('/proc/pressure/memory').read_text(),'GPU_context_created':False,'allocation_or_lease_created':False},indent=2)+'\n')
    namespace='gpu0-v095-'+suffix+'-20261105'
    scope_name='ONE_OFFICIAL095_'+suffix.upper()+'_SEED0_TRIAL_ONLY'
    folder.mkdir()
    for path in TEMPLATE.glob('*.py'):
        source=path.read_text().replace('released_v095_bridge',experiment).replace('v095_lr_g050_d050_p050',identity)
        source=source.replace('gpu0-released-v095-bridge-20261105',namespace)
        source=source.replace('ONE_OFFICIAL095_BEST_RATE_BRIDGE_SEED0_TRIAL_ONLY',scope_name)
        if path.name=='preset_scope.py':
            source=assignment(source,'PROTOCOL',"ROOT/'analysis/released_penalty_v095/protocol.json'")
            source=assignment(source,'HYPER',repr(setting['parameters']))
            source=source.replace("PLAN=HERE/'plan.json'", "PLAN=HERE/'plan.json'\nPENALTY_PROOF=ROOT/'analysis/released_penalty_v095/CPU_qualification.json'\nREFERENCE=ROOT/'analysis/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0-result.json'\nREFERENCE_PLAN=ROOT/'released_v095_bridge_runtime/plan.json'")
            source=source.replace('return [PROTOCOL,MODEL_PROOF', 'return [PENALTY_PROOF,REFERENCE,REFERENCE_PLAN,PROTOCOL,MODEL_PROOF')
            source=source.replace('PROSPECTIVE_RELEASED_PARTICLEGAN_HYPERPARAMETER_SEARCH_NOT_LAUNCHED','PROSPECTIVE_OFFICIAL095_TARGETED_PENALTY_FOLLOWUP_NOT_LAUNCHED')
            source=source.replace("assert len(p['stage1']['configs'])==27", "assert len(p['stage1']['configs'])==12")
            # Targeted rows additionally record the changed scalar and factor.
            source=source.replace("assert [s for s in p['stage1']['configs'] if s['id']==CANDIDATE_ID]==[{'id':CANDIDATE_ID,'parameters':HYPER}]", "settings=[s for s in p['stage1']['configs'] if s['id']==CANDIDATE_ID];assert len(settings)==1 and settings[0]['parameters']==HYPER")
            marker="    proof=read(MODEL_PROOF)"
            replacement="""    ref=read(REFERENCE)
    assert ref['owned_release_and_clean_settlement_verified'] is True and ref['official095_origin_version_and_continuous_domain_verified'] is True
    assert ref['decision']['accept'] is False and read(REFERENCE_PLAN)['enabled'] is False
    assert ref['candidate']['completed_updates']==ref['candidate']['fixed_endpoint']==20000
    assert ref['candidate']['particlegan_version']=='0.9.5' and ref['candidate']['released_hyperparameters']==p['baseline_parameters']
    penalty=read(PENALTY_PROOF)
    assert penalty['status']=='ALL12_OFFICIAL095_ONE_FIELD_PENALTY_CONTROLS_FULLWIDTH_CPU_REPLAY_PASS' and penalty['configurations_checked']==12
    assert penalty['actual_optimizer_pairs_executed']==36 and penalty['CUDA_initialized'] is False and penalty['GPU_job_created'] is False
    tested=[x for x in penalty['checks'] if x['candidate_id']==CANDIDATE_ID]
    assert len(tested)==1 and tested[0]['parameters']==HYPER and tested[0]['complete_next_pair_state_byte_exact'] is True
    for path,pin in penalty['source_SHA256'].items():assert pins[path]==pin==sha(path)
    proof=read(MODEL_PROOF)"""
            assert marker in source;source=source.replace(marker,replacement)
        elif path.name=='prepare.py':
            source=source.replace("pins.update(b.read(scope.COMPAT_PROOF)['source_SHA256'])", "pins.update(b.read(scope.PENALTY_PROOF)['source_SHA256'])\n    pins.update(b.read(scope.COMPAT_PROOF)['source_SHA256'])")
            source=source.replace('Official0.9.5 version bridge, continuous150D, G/D/prior rates=(0.006,0.009,0.015)', 'Official0.9.5 one-field penalty control '+identity+', continuous150D, fixedG/D/prior rates=(0.006,0.009,0.015)')
        elif path.name=='resource_profile.py':
            parent_auth=read(ROOT/'artifacts/allocations/CAL_GPU0_V1_released_v095_bridge_f0_pg_s0/authorization.json')
            source=assignment(source,'PREVIOUS_REGISTRY',"Path("+repr(str(Path(parent_auth['registry_path'])/'registry.json'))+")")
        elif path.name=='test_scope.py':
            source=source.replace('PROSPECTIVE_RELEASED_PARTICLEGAN_HYPERPARAMETER_SEARCH_NOT_LAUNCHED','PROSPECTIVE_OFFICIAL095_TARGETED_PENALTY_FOLLOWUP_NOT_LAUNCHED')
            extra="""    def test_reference_complete095_rejection_and_actual_disabled_plan_required(self):
        auth=self.auth();rd,sh=self.active(auth);original=scope.read
        for change in ({'decision':{'accept':True}}, {'owned_release_and_clean_settlement_verified':False}):
            ref=copy.deepcopy(original(scope.REFERENCE));ref.update(change)
            with rd,sh,patch.object(scope,'read',side_effect=lambda p:ref if Path(p)==scope.REFERENCE else (dict(scope.ENABLED) if Path(p)==scope.PLAN else original(p))),self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
            rd,sh=self.active(auth)
        with rd,sh,patch.object(scope,'read',side_effect=lambda p:{'enabled':True} if Path(p)==scope.REFERENCE_PLAN else (dict(scope.ENABLED) if Path(p)==scope.PLAN else original(p))),self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
        self.assertNotIn('torch',sys.modules)

"""
            source=source.replace('    def test_model_proof_whole_preset_scope_and_owner_window',extra+'    def test_model_proof_whole_preset_scope_and_owner_window')
        elif path.name=='qualify.py':
            source=source.replace("'tests_passed':25", "'tests_passed':26,'reference_completed095_joint_rejection_and_disabled_plan_refusals_verified':True")
            source=source.replace('scope.NATIVE_PROOF,scope.PARAMETERIZATION', 'scope.NATIVE_PROOF,scope.PENALTY_PROOF,scope.REFERENCE,scope.REFERENCE_PLAN,scope.PARAMETERIZATION')
        ast.parse(source);(folder/path.name).write_text(source)
    policy=read(TEMPLATE/'policy.prepared.json');policy['registry_path']=str(ROOT/'artifacts'/namespace/'live_registry')
    policy['resource_basis']['observation_path']=str(observation);policy['resource_basis']['observation_SHA256']=sha(observation)
    (folder/'policy.prepared.json').write_text(json.dumps(policy,indent=2)+'\n')
    plan=read(TEMPLATE/'plan.json');plan.update(enabled=False,experiment=experiment,candidate_id=identity)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'status':'GENERATED_DISABLED_OFFICIAL095_PENALTY_RUNTIME_REQUIRES_ACTUAL_CPU_QUALIFICATION','candidate_id':identity,'runtime':str(folder),'parameters':setting['parameters'],'reference_result_SHA256':sha(REFERENCE),'GPU_job_started':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate-id',required=True);make(parser.parse_args().candidate_id)
