"""Read the real native178 checkpoint and closed resource lineage without CUDA."""
import datetime,json,math,os,shutil,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT/'hpo500_r3_runtime',ROOT/'analysis/hpo500/model',ROOT/'model',ROOT,ROOT/'routed_resume_runtime',SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
import preset_scope as scope
from release_env import activate
activate()
import torch,released_adapter as model,released_update as update
from state_fingerprint import fingerprint

def main():
    attempt='CAL_GPU0_V1_h5_c008_search_f0_pg_s0'
    ap=ROOT/'artifacts/allocations'/attempt/'authorization.json';auth=scope.read(ap)
    ledger=Path(auth['ledger_path']);job=scope.read(ledger)['jobs'][0]
    settlement=ledger.with_name(attempt+'-settlement.json');closed=scope.read(settlement)
    assert job['status']=='failed' and job['reservation_released']
    assert job['error']=="Capacity('Protected/registered GPU growth no longer fits; stop owned worker only')"
    assert closed['uncertain'] is None and closed['settlement']['status']=='settled'
    for key in ('child_identity','guard_identity'):assert not Path('/proc',str(job[key]['pid'])).exists()
    assert not scope.read(Path(auth['registry_path'])/'registry.json')['jobs']
    driver=scope.read(ROOT/'artifacts/hpo500-r2/driver_identity.json');assert not Path('/proc',str(driver['pid'])).exists()
    original=ROOT/'runs'/attempt/'fit/recovery.pt';receipt=scope.read(original.parent/'checkpoint_receipt.json')
    assert receipt['SHA256']==scope.sha(original)=='b14934c3ee0095d3a3445b442853a53e668c78848f298660304f2f6a75221e90'
    q=torch.load(original,map_location='cpu',weights_only=False)
    assert q['schema']==1 and q['method']=='pg' and q['fold']==q['seed']==0
    assert q['status']=='incomplete' and q['completed_updates']==q['policy']['completed_steps']==q['usage']['steps']==178
    assert q['fixed_endpoint']==500 and q['recipe']==q['policy']['recipe'] and q['recipe']['total_steps']==20000
    assert q['recipe']['ema_decay']==0 and q['recipe'].get('generator_gradient_response','none')=='none'
    assert q['policy']['device']=='cuda:0' and q['particlegan_version']=='0.9.6'
    assert q['penalty_options']=={'input_domain':'continuous'} and q['native_recipe']=='bispf_optimized'
    assert q['released_hyperparameters']==auth['released_hyperparameters']==scope.read(scope.proposal_path('c008'))['hyperparameters']
    assert q['variant_config']==auth['variant_config']==scope.CFG
    assert q['cache_manifest_SHA256']==scope.sha(Path(auth['cache_dir'])/'manifest.json')
    assert q['training_streams']==model.training_streams(0,0)
    assert int(q['usage']['draws'].sum())==178*32 and int(q['usage']['unsampled_moved_steps'].sum())==0
    update.configure(scope.CFG);update.configure_hyperparameters(q['released_hyperparameters'])
    assert update.recipe().to_dict()==q['recipe']
    def finite(value):
        if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all())
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(tuple,list)):
            for v in value:finite(v)
        elif isinstance(value,float):assert math.isfinite(value)
    finite(q)
    archive=ROOT/'artifacts/hpo500-recovery-r3/parent';archive.mkdir(parents=True,exist_ok=False)
    checkpoint=archive/'native178.pt';curve=archive/'learning_curve.jsonl'
    shutil.copyfile(original,checkpoint);shutil.copyfile(original.parent/'learning_curve.jsonl',curve)
    charge=closed['settlement']['charged_GPU_seconds'];remaining=3600-math.ceil(charge);assert remaining==3429
    assert not torch.cuda.is_initialized()
    proof=dict(status='ACTUAL_HPO500_RESOURCE_INTERRUPTION_NATIVE178_CPU_AUDIT_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_attempt=attempt,parent_authorization_path=str(ap),parent_authorization_SHA256=scope.sha(ap),parent_ledger_path=str(ledger),parent_ledger_SHA256=scope.sha(ledger),parent_settlement_path=str(settlement),parent_settlement_SHA256=scope.sha(settlement),checkpoint_path=str(checkpoint),checkpoint_SHA256=scope.sha(checkpoint),learning_curve_path=str(curve),learning_curve_SHA256=scope.sha(curve),whole_parent_scientific_fingerprint=fingerprint(q),native_policy_fingerprint=fingerprint(q['policy']),completed_updates=178,fixed_endpoint=500,seed=0,hyperparameters=q['released_hyperparameters'],variant_config=q['variant_config'],cache_dir=auth['cache_dir'],parent_charged_GPU_seconds=charge,remaining_trial_cap_seconds=remaining,parent_expiry_unix=auth['expires_unix'],owned_processes_and_lease_absent=True,clean_settlement=True,scientific_nonfinite_failure=False,native_device_metadata_relabelled=False,CUDA_initialized=False,same_model_and_all_random_state_CUDA_restore_required_before179=True,source_SHA256={str(p):scope.sha(p) for p in [HERE/'audit_parent_cpu.py',scope.MODEL_DIR/'released_adapter.py',scope.MODEL_DIR/'released_update.py']})
    scope.PARENT_PROOF.write_text(json.dumps(proof,indent=2)+'\n')
    protocol=dict(status='PROSPECTIVE_RESOURCE_ONLY_HPO500_NATIVE178_CONTINUATION',UTC=proof['UTC'],original_scientific_protocol_path=str(scope.PROTOCOL),original_scientific_protocol_SHA256=scope.sha(scope.PROTOCOL),parent_CPU_audit_path=str(scope.PARENT_PROOF),parent_CPU_audit_SHA256=scope.sha(scope.PARENT_PROOF),resource_change='No scientific or resource margins change. Distinct physical continuation after a clean transient foreign-growth yield; preserve all178 native updates, all random state, original trial expiry and cumulative3600second active allowance. Both cards inspected; GPU0 fits with retained2GiB own/3GiBdriver/foreign growth margins; GPU1 does not fit those same margins.',interrupted_candidate='c008',interrupted_phase='search',interrupted_seed=0,resume_from_updates=178,endpoint=500,new_model_updates=322,continuation_cap_seconds=remaining,continuation_expiry_no_later_than_parent=auth['expires_unix'],fresh_phase_cap_seconds=3600,resume_GPU=auth['gpu_uuid'],parent_charged_GPU_seconds=charge,single_logical_result_and_total_charge='Logicalc008 has one result; parent178+continuation322 phases and charges included. No seed reset, dropping, restart or clock/deadline reset.',later_trials='Unchanged25selection/four fresh-seed confirmation; prior8audits retained including native48 recovery. Original phase guards remain.',project_GPU_hour_cap=1024,study_GPU_hour_cap=24,automatic_ATH_promotion=False)
    scope.RECOVERY_PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ['status','completed_updates','checkpoint_SHA256','whole_parent_scientific_fingerprint','parent_charged_GPU_seconds','remaining_trial_cap_seconds','CUDA_initialized']},indent=2))
if __name__=='__main__':main()
