"""Create an inactive guarded20k->40k extension; no allocation or launch."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/ml2/alz')
TEMPLATE = ROOT / 'released_rate_g100d100p050_runtime'
NEW = ROOT / 'released_best_extend40k_runtime'
EXT_PROTOCOL = ROOT / 'analysis/released_best_rate_extension40k_protocol.json'
read = lambda path: json.loads(Path(path).read_bytes())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignment(source, name, value):
    source, count = re.subn(r'^' + re.escape(name) + r'=.*$', lambda match: name + '=' + value, source, flags=re.M)
    assert count == 1, name
    return source


def main():
    assert not NEW.exists()
    proof = read(ROOT / 'analysis/released_rate_g100d100p050_runtime_CPU_qualification.json')
    assert proof['tests_passed'] == 24
    for path, pin in proof['source_SHA256'].items():
        assert sha(path) == pin, path
    extension = read(EXT_PROTOCOL)
    assert extension['initial_completed_updates'] == 20000 and extension['fixed_endpoint'] == 40000
    assert sha(extension['checkpoint_path']) == extension['checkpoint_SHA256']
    NEW.mkdir()
    for path in TEMPLATE.glob('*.py'):
        source = path.read_text().replace('released_rate_g100d100p050_runtime', NEW.name)
        source = source.replace('released_rate_g100d100p050', 'released_best_extend40k')
        source = source.replace('lr_g100_d100_p050', extension['candidate_id'])
        source = source.replace('ONE_RELEASED_LR_G100_D100_P050_SEED0_TRIAL_ONLY', 'ONE_RELEASED_HALF_RATE_20000_TO40000_EXTENSION_ONLY')
        source = source.replace('gpu1-released-g100d100p050-20261104', 'gpu1-released-best-extend40k-20261105')
        source = source.replace('train_released_hyperparameters', 'extend_released_hyperparameters')
        if path.name == 'binding.py':
            source = source.replace("'continue_hybrid','train_extended_hybrid'", "'extend_released_hyperparameters','continue_hybrid','train_extended_hybrid'")
        elif path.name == 'resource_profile.py':
            source = assignment(source, 'PREVIOUS_REGISTRY', "ROOT/'artifacts/gpu1-released-g050d050p100-20261104/live_registry/registry.json'")
        elif path.name == 'preset_scope.py':
            source = assignment(source, 'HYPER', repr(extension['released_hyperparameters']))
            source = source.replace("RUNTIME_PROOF=", "EXT_DIR=ROOT/'analysis/released_extension40k'\nEXT_PROTOCOL=ROOT/'analysis/released_best_rate_extension40k_protocol.json'\nEXT= json.loads(EXT_PROTOCOL.read_bytes())\nEXT_PROOF=EXT_DIR/'CPU_qualification.json'\nPARENT_AUDIT=EXT_DIR/'parent20k_CPU_component_audit.json'\nRUNTIME_PROOF=")
            source = source.replace("'fixed_endpoint':20000", "'fixed_endpoint':40000")
            source = source.replace("'training_seed':0,'experiment':CELL['experiment'],'candidate_id':CANDIDATE_ID}", "'training_seed':0,'experiment':CELL['experiment'],'candidate_id':CANDIDATE_ID,\n    'initial_completed_updates':20000,'new_model_updates':20000,'native_recipe_schedule_horizon':20000}")
            source = source.replace("    return [PROTOCOL,", "    return [EXT_PROTOCOL,EXT_PROOF,PARENT_AUDIT,EXT_DIR/'extension.py',*[Path(phase[key]) for phase in EXT['charged_prior_phases'] for key in ('authorization_path','ledger_path','settlement_path')],*[Path(EXT[key]) for key in ('parent_result_path','parent_authorization_path','checkpoint_path','selection_readout_snapshot_path')],Path(EXT['training_log']['immutable_copy_path']),PROTOCOL,")
            source = source.replace("assert type(auth['fixed_endpoint']) is int and auth['fixed_endpoint']==20000", "assert type(auth['fixed_endpoint']) is int and auth['fixed_endpoint']==40000")
            source = source.replace("auth['pure_formulation']=='released_hyperparameters'", "auth['pure_formulation']=='released_horizon40k'")
            source = source.replace("auth['initial_completed_updates']==0", "auth['initial_completed_updates']==20000")
            source = source.replace("assert [s for s in p['stage1']['configs'] if s['id']==CANDIDATE_ID]==[{'id':CANDIDATE_ID,'parameters':HYPER}]", "assert [s for s in p['stage1']['configs'] if s['id']==EXT['reference_rate_candidate_id']]==[{'id':EXT['reference_rate_candidate_id'],'parameters':HYPER}]")
            source = source.replace("    assert read(PLAN)==ENABLED,", "    validate_extension_inputs(auth)\n    assert read(PLAN)==ENABLED,")
            source = source.replace("for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):", "for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,EXT_DIR):")
            source = source.replace("        assert proof['status']=='PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION'", "        assert proof['status']=='PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION'\n        assert proof['tests_passed']==27 and proof['extension_parent_source_and_external_cap_refusals_verified'] is True")
            source += '''\n\ndef validate_extension_inputs(auth):
    assert EXT['status']=='PROSPECTIVE_RELEASED_BEST_RATE_20K_TO40K_DURATION_EXTENSION_NOT_LAUNCHED'
    assert auth['extension']==EXT and auth['extension_protocol_SHA256']==sha(EXT_PROTOCOL)
    assert auth['native_recipe_schedule_horizon']==20000
    assert auth['extension_qualification_SHA256']==sha(EXT_PROOF)
    assert EXT['initial_completed_updates']==20000 and EXT['fixed_endpoint']==40000 and EXT['new_model_updates']==20000
    assert EXT['native_recipe_schedule_horizon']==20000 and EXT['public_execution_limit']==40000
    assert EXT['released_hyperparameters']==HYPER and EXT['variant_config']==CFG
    assert EXT['new_calibration_checkpoints_after_extension_training']==[30000,40000] and EXT['eligible_for_selection_updates']==[40000]
    assert EXT['parent_calibration_informed_extension_declaration'] is True and EXT['new_extension_calibration_only_after_full40k_training'] is True
    for key in ('parent_result','parent_authorization','parent_audit'):
        assert sha(EXT[key+'_path'])==EXT[key+'_SHA256']
    assert sha(EXT['checkpoint_path'])==EXT['checkpoint_SHA256']
    assert sha(EXT['selection_readout_snapshot_path'])==EXT['selection_readout_snapshot_SHA256']
    assert sha(EXT['training_log']['immutable_copy_path'])==EXT['training_log']['SHA256']
    audit=read(PARENT_AUDIT);result=read(EXT['parent_result_path']);candidate=result['candidate']
    assert audit['status']=='ACTUAL_COMPLETED20K_RELEASED_PARENT_CPU_COMPONENT_ARCHIVE_AUDIT_PASSED'
    assert audit['CUDA_initialized'] is False and audit['model_floating_values']==371177
    assert audit['whole_parent_scientific_fingerprint']==EXT['whole_parent_scientific_fingerprint']
    assert result['owned_release_and_clean_settlement_verified'] is True and result['predeclared_all_three_checkpoint_saved_scores_and_counts_verified'] is True
    assert candidate['status']=='completed' and candidate['completed_updates']==candidate['fixed_endpoint']==20000
    assert candidate['released_hyperparameters']==HYPER and candidate['variant_config']==CFG
    assert result['charged_GPU_seconds_total_scientific_trial']==EXT['parent_charged_GPU_seconds']
    phases=EXT['charged_prior_phases'];assert sum(phase['charged_GPU_seconds'] for phase in phases)==EXT['parent_charged_GPU_seconds']
    for phase in phases:
        for key in ('authorization','ledger','settlement'):assert sha(phase[key+'_path'])==phase[key+'_SHA256']
        parent_auth=read(phase['authorization_path']);job=read(phase['ledger_path'])['jobs'][0];settlement=read(phase['settlement_path'])
        assert job['reservation_released'] is True and settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
        assert settlement['settlement']['charged_GPU_seconds']==phase['charged_GPU_seconds']
        assert not read(Path(parent_auth['registry_path'])/'registry.json')['jobs']
        plans=[Path(arg).resolve().parent/'plan.json' for arg in parent_auth['command'] if Path(arg).name=='worker_gpu0.py']
        assert len(plans)==1 and plans[0].is_relative_to(ROOT) and str(plans[0]) in parent_auth['files_SHA256']
        assert read(plans[0])['enabled'] is False
    proof=read(EXT_PROOF)
    assert proof['status']=='PASSED_RELEASED_PUBLIC_EXECUTION_CAP_EXTENSION_FULLWIDTH_CPU_REPLAY_AND_REFUSALS'
    assert proof['tests_passed']==4 and proof['CUDA_initialized'] is False and proof['GPU_job_or_allocation_started'] is False
    pins=auth['files_SHA256']
    for group in (proof['source_SHA256'],audit['source_SHA256'],EXT['source_SHA256']):
        for path,pin in group.items():assert pins[path]==pin==sha(path)
'''
        elif path.name == 'prepare.py':
            source = source.replace("operation='extend_released_hyperparameters',fixed_endpoint=20000", "operation='extend_released_hyperparameters',fixed_endpoint=40000")
            source = source.replace("pure_formulation='released_hyperparameters'", "pure_formulation='released_horizon40k'")
            source = source.replace("initial_completed_updates=0,new_model_updates=20000", "initial_completed_updates=20000,new_model_updates=20000,extension=copy.deepcopy(scope.EXT),extension_protocol_SHA256=b.sha(scope.EXT_PROTOCOL),extension_qualification_SHA256=b.sha(scope.EXT_PROOF),native_recipe_schedule_horizon=20000")
            source = source.replace("scope.MODEL_DIR) for p", "scope.MODEL_DIR,scope.EXT_DIR) for p")
            source = source.replace("model=b.read(scope.MODEL_PROOF);", "pins.update(b.read(scope.EXT_PROOF)['source_SHA256']);pins.update(b.read(scope.PARENT_AUDIT)['source_SHA256']);pins.update(scope.EXT['source_SHA256'])\n    model=b.read(scope.MODEL_PROOF);")
            source = source.replace("'endpoint':20000,'base_model_updates':0", "'endpoint':40000,'base_model_updates':20000")
            start = source.index("source_review='One released")
            end = source.index(")\n    auth['command']", start)
            source = source[:start] + "source_review='One separately declared same-model20k->40k duration extension through public execution_limit=40000. Native recipe/schedule horizon20000 and all saved clocks/eleven streams retained; full CUDA/scientific restore required before20001. Only40k eligible;30k diagnostic after full extension training; observed parent calibration informed this hypothesis. SameG.006/D.009/P.015, fixed architecture/counts/data. Finite6h sharedGPU1 phase inside existing5day/1024hour scope. Exact native reload/count/score/cleanup/charge audit; no automatic successor or promotion'" + source[end:]
        elif path.name == 'test_scope.py':
            source = source.replace("{'fixed_endpoint':40000}", "{'fixed_endpoint':20000}")
            source = source.replace("'fixed_endpoint':20000,'completed_updates':5000,'policy':{'completed_steps':5000}", "'fixed_endpoint':40000,'completed_updates':30000,'initial_completed_updates':20000,'native_recipe_schedule_horizon':20000,'recipe':{'total_steps':20000},'policy':{'completed_steps':30000,'device':'cuda:0','recipe':{'total_steps':20000}}")
            source = source.replace('worker.validate_curve_checkpoint(q,auth,5000)', 'worker.validate_curve_checkpoint(q,auth,30000)')
            source = source.replace('for updates in (0,7013,19999,20000)', 'for updates in (0,5000,20000,40000)')
            source = source.replace("runtimeproof={'status':", "runtimeproof={'tests_passed':27,'extension_parent_source_and_external_cap_refusals_verified':True,'status':")
        elif path.name == 'qualify.py':
            source = source.replace("'tests_passed':24", "'tests_passed':27,'extension_parent_source_and_external_cap_refusals_verified':True")
            source = source.replace("scope.MODEL_DIR.glob('*.py'),", "scope.MODEL_DIR.glob('*.py'),*scope.EXT_DIR.glob('*.py'),scope.EXT_PROTOCOL,scope.EXT_PROOF,scope.PARENT_AUDIT,")
        (NEW / path.name).write_text(source)
    policy = read(TEMPLATE / 'policy.prepared.json')
    policy['registry_path'] = str(ROOT / 'artifacts/gpu1-released-best-extend40k-20261105/live_registry')
    policy['resource_basis']['foreign_control'] = 'NONE. New completed20k-to40k duration phase; saved native recipe horizon20000 and state preserved through public execution_limit. Same shared resources/guards. Full fresh capacity/source review and separate activation/admission required.'
    (NEW / 'policy.prepared.json').write_text(json.dumps(policy, indent=2) + '\n')
    plan = {'enabled': False, 'native_recipe': 'bispf_optimized', 'fixed_endpoint': 40000, 'pure_only': True,
            'training_seed': 0, 'experiment': 'released_best_extend40k', 'candidate_id': extension['candidate_id'],
            'initial_completed_updates': 20000, 'new_model_updates': 20000, 'native_recipe_schedule_horizon': 20000}
    (NEW / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    observation = ROOT / 'analysis/released_best_extend40k_initial_resource_observation.json'
    observation.write_text(json.dumps({'status': 'SOURCE_PREPARATION_ONLY_NOT_ADMISSION',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'predecessor_attempt': 'CAL_GPU0_V1_released_rate_g050d050p100_f0_pg_s0',
        'GPU_job_or_allocation_created': False, 'fresh_complete_shared_GPU1_capacity_required_at_admission': True}, indent=2) + '\n')
    for path in NEW.glob('*.py'):
        ast.parse(path.read_text())
    print(json.dumps({'status': 'GENERATED_DISABLED_DURATION_RUNTIME_REQUIRES_WORKER_FINISHER_AND_QUALIFICATION', 'runtime': str(NEW)}))


if __name__ == '__main__':
    main()
