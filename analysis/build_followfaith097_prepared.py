"""Generate a disabled follow-up namespace without changing any active source."""
import ast, json, shutil
from pathlib import Path
ROOT=Path('/ml2/alz');SOURCE=ROOT/'localfaith097_runtime'
DEST=ROOT/'followfaith097_runtime';STUDY=ROOT/'analysis/followfaith097'

VALIDATOR='''def validate_protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_RELEASED097_FULL20K_FOLLOWUP_COORDINATE_SEARCH','Complete original12fit audit and frozen follow-up design required before admission'
 assert p['native_version']=='0.9.7' and p['endpoint']==20000 and p['selection_seeds']==[0,1] and p['fresh_confirmation_seeds']==[2,3,4]
 assert p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only'] and p['metric_role']=='calibration_selection'
 assert p['formulation']==FORMULATION and not p['automatic_promotion'] and not p['SOTA_claim'] and not p['full_goal_complete']
 assert p['design_evidence_SHA256']
 for path,pin in p['design_evidence_SHA256'].items():assert sha(path)==pin
 audit=read(p['baseline_audit_path']);assert audit['status']=='INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS'
 assert audit['required_new_attempts']==12 and audit['required_attempts']==16
 assert audit['formulation']==FORMULATION and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero']
 assert audit['training_seeds']==[0,1] and audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read']
 assert audit['model_parity_proof_path']==str(MODEL_PROOF) and audit['model_parity_proof_SHA256']==sha(MODEL_PROOF)
 frozen=read(STUDY/'frozen_design.json')
 expected=resolve_design(audit,p['design_now_unix'],read(WINDOW)['expires_unix'],read(p['matrix_cost_forecast_path'])['estimated_matrix_GPU_hours'])
 assert frozen==expected and frozen['status']=='PROPOSED_BOUNDED_RELEASED097_LOWER_DIRECTION_FOLLOWUP'
 assert p['candidates']==frozen['candidates'] and 1<=len(p['candidates'])<=2 and p['maximum_new_fits']==2*len(p['candidates'])
 assert [q['id'] for q in p['candidates']]==ENABLED['candidate_ids']
 old=read(p['upstream_control_audit_path']);assert old['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS'
 assert p['upstream_control_audit_path']==read(audit['protocol_path'])['baseline_audit_path']
 combined={**old['result_SHA256'],**audit['result_SHA256']};assert len(combined)==16 and p['reference_result_SHA256']==combined
 controls=p['reference_controls'];ids={'f000','f001',*[f't{i:03}' for i in range(6)]}
 assert len(controls)==16 and {(q['candidate_id'],q['seed']) for q in controls}=={(c,s) for c in ids for s in (0,1)}
 for q in controls:
  original=read(q['source_result_path']);assert sha(q['source_result_path'])==q['source_result_SHA256']==combined[q['attempt']]
  assert q==dict(original,read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0,source_result_path=q['source_result_path'],source_result_SHA256=q['source_result_SHA256'])
  assert q['from_scratch'] and q['physical_start_update']==0 and q['finite']
 grid=read(GRID_MODEL_PROOF);assert grid['status']=='ALL4_POSSIBLE_FOLLOWUP_SETTING_SEED_NATIVE_TRAINER_PREFIX_CASES_PASS' and grid['actual_cases']==4 and grid['actual_optimizer_pairs_per_implementation']==12 and not grid['CUDA_initialized'] and not grid['real_ALZ_data_read'] and not grid['native_package_modified']
 for path,pin in grid['source_SHA256'].items():assert sha(path)==pin
 for candidate in p['candidates']:
  for seed in (0,1):
   entry=next(q for q in grid['rows'] if q['anchor_id']=='f000' and q['candidate_id']==candidate['id'] and q['seed']==seed)
   assert entry['hyperparameters']==candidate['hyperparameters'] and entry['optimizer_pairs_per_loop']==3 and entry['exact_models_optimizers_rng_and_update_clock'] and entry['exact_all_losses_and_penalty'] and entry['from_scratch']
 proof=read(MODEL_PROOF);assert proof['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS' and proof['prior_regularizer_domain']=='all512rows' and proof['explicit_bound_critic_thermal'] and not proof['native_package_modified']
 for path,pin in proof['source_SHA256'].items():assert sha(path)==pin
 return p
'''

SELECTOR='''"""Combined exploratory selection;16 old endpoints are labeled observed controls."""
from released097_selection import summarize,proposed_confirmation,require
def summarize_local(candidates,new_rows,control_rows):
 require(1<=len(candidates)<=2,'Only the frozen one/two follow-up settings allowed')
 old_ids={'f000','f001',*[f't{i:03}' for i in range(6)]}
 require(len(control_rows)==16 and {(r['candidate_id'],r['seed']) for r in control_rows}=={(c,s) for c in old_ids for s in (0,1)},'All16 original observed endpoints required')
 require(len(new_rows)==2*len(candidates) and {(r['candidate_id'],r['seed']) for r in new_rows}=={(c['id'],s) for c in candidates for s in (0,1)},'Every new paired endpoint required')
 require(not ({c['id'] for c in candidates}&old_ids),'New and previous controls must be distinct')
 for r in control_rows:require(r['read_only_previously_measured_reference'] and r['new_model_fitted'] is False and r['new_GPU_charge']==0 and r['source_result_path'] and r['source_result_SHA256'],'Observed controls cannot be warm starts or new charges')
 for r in new_rows:require(not r.get('read_only_previously_measured_reference',False) and r.get('new_model_fitted',True) is not False and r['charged_GPU_seconds']>0,'Each new outcome must have its own real fit/cost')
 choices=[dict(candidate_id=c['id'],hyperparameters=c['hyperparameters']) for c in candidates]+[dict(candidate_id=cid,hyperparameters=next(r['hyperparameters'] for r in control_rows if r['candidate_id']==cid)) for cid in sorted(old_ids)]
 summary=summarize(choices,[*new_rows,*control_rows],[0,1]);summary['reference_id']='f000';summary['proposed_confirmation_configurations']=proposed_confirmation(summary,'f000')
 for c in summary['candidates']:c['previously_measured_control']=c['candidate_id'] in old_ids
 summary.update(required_new_fits=2*len(candidates),previously_measured_control_fits=16,controls_refitted=False,control_seeds_previously_observed=True,independent_population_confirmation=False)
 return summary
'''

def main():
    assert not DEST.exists();DEST.mkdir()
    STUDY.mkdir(exist_ok=True)
    for name in ('proposals','results'):(STUDY/name).mkdir(exist_ok=True)
    for path in SOURCE.glob('*.py'):
        value=path.read_text().replace('localfaith097','followfaith097').replace('lf097_','uf097_')
        value=value.replace('LOCAL_COORDINATE','FOLLOWUP_COORDINATE')
        value=value.replace('released097_local_search_design','released097_followup_design')
        value=value.replace('PREPARED_PENDING_COMPLETE_CORRECTED_BASELINES_AND_LOCAL_DESIGN','PREPARED_PENDING_COMPLETE_ORIGINAL12_AND_FOLLOWUP_DESIGN')
        value=value.replace('Complete corrected baseline audit','Complete original12fit audit')
        if path.name=='preset_scope.py':
            start=value.index('def validate_protocol():');end=value.index('def cell(',start)
            value=value[:start]+VALIDATOR+value[end:]
            value=value.replace("candidate='t000'","candidate='u000'")
        if path.name=='resource_profile.py':
            value=value.replace("PREVIOUS_REGISTRY=Path('/ml2/alz/artifacts/gpu1-faithful097-paired-search-20261107/live_registry/registry.json')", "PREVIOUS_REGISTRY=Path('/ml2/alz/artifacts/gpu1-localfaith097-paired-search-20261107/live_registry/registry.json')")
        if path.name=='test_factory.py':
            value=value.replace("q['actual_cases'],24","q['actual_cases'],4").replace("q['actual_optimizer_pairs_per_implementation'],72","q['actual_optimizer_pairs_per_implementation'],12")
        if path.name=='test_prepared.py':value=value.replace("s.cell('t000'","s.cell('u000'")
        ast.parse(value);(DEST/path.name).write_text(value)
    policy=json.loads((SOURCE/'policy.prepared.json').read_text())
    policy['registry_path']=policy['registry_path'].replace('localfaith097','followfaith097')
    (DEST/'policy.prepared.json').write_text(json.dumps(policy,indent=2)+'\n')
    (DEST/'plan.json').write_text(json.dumps(dict(enabled=False,experiment='followfaith097',reason='Original12fit audit and bounded design not yet available'),indent=2)+'\n')
    p=json.loads((ROOT/'analysis/localfaith097/protocol.json').read_text())
    for key in ('baseline_audit_path','design_now_unix','matrix_cost_forecast_path','selection_rule','confirmation_rule'):p.pop(key,None)
    p.update(status='PREPARED_PENDING_COMPLETE_ORIGINAL12_AND_FOLLOWUP_DESIGN',candidates=[],reference_controls=[],maximum_new_fits=4,design_evidence_SHA256={})
    (STUDY/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    shutil.copyfile(ROOT/'analysis/localfaith097/native_release.json',STUDY/'native_release.json')
    (STUDY/'selection.py').write_text(SELECTOR)
    driver=(ROOT/'analysis/run_localfaith097.py').read_text().replace('localfaith097','followfaith097').replace('LOCAL_COORDINATE','FOLLOWUP_COORDINATE').replace('previously_measured_control_fits=4','previously_measured_control_fits=16')
    ast.parse(driver);(ROOT/'analysis/run_followfaith097.py').write_text(driver)
    print('Prepared disabled follow-up runtime and driver; no allocation or activation.')
if __name__=='__main__':main()
