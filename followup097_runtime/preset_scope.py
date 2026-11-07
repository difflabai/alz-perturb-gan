"""Fresh matched seeds2..4 of audited paired-selection choices, never warm starts."""
import hashlib,importlib.util,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/followup097'
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json';MODEL_DIR=STUDY/'model'
MODEL_PROOF=ROOT/'analysis/released_tuning_v096/CPU_qualification.json';HPO_PROOF=ROOT/'analysis/hpo500/model_CPU_qualification.json'
NATIVE_PROOF=STUDY/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
RULE=ROOT/'analysis/refinement097_followup_selection_rule_20261107.json';SELECTOR=ROOT/'analysis/refinement097_followup_selection.py'
BASELINE_PROOF=ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ENABLED=dict(enabled=True,experiment='followup097',native_version='0.9.7',formulation='native097_bispf_optimized_nonEMA',endpoint=20000,
 training_seeds=[2,3,4],candidate_ids=[q['candidate_id'] for q in read(PROTOCOL)['selected_configurations']],owned_GPU_GiB=2,calibration_only=True,all_training_from_scratch=True)
def selector():
 assert read(RULE)['source_SHA256'][str(SELECTOR)]==sha(SELECTOR)
 spec=importlib.util.spec_from_file_location('frozen_followup_selection',SELECTOR);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate_protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_MATCHED_SEED234_FOLLOWUP_AFTER_PAIRED_REFINEMENT','All20 paired selection endpoints and independent audit required before followup admission'
 assert p['native_version']=='0.9.7' and p['endpoint']==20000 and p['training_seeds']==[2,3,4] and p['selection_seeds']==[0,1]
 assert p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None
 assert p['calibration_only'] and p['metric_role']=='calibration_selection' and p['reference_models_refitted'] is False and p['reference_measurements_previously_observed']
 assert not p['automatic_promotion'] and not p['SOTA_claim'] and not p['full_goal_complete']
 assert p['selection_evidence_SHA256']
 for path,pin in p['selection_evidence_SHA256'].items():assert sha(path)==pin
 audit=read(p['paired_selection_audit_path']);report=read(p['paired_selection_report_path'])
 assert audit['status']=='INDEPENDENT_NATIVE097_PAIRED_REFINEMENT20_FITS_AUDIT_PASS' and audit['required_attempts']==20 and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero']
 assert audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read']
 assert audit['report_SHA256']==sha(p['paired_selection_report_path'])
 for key,value in audit['summary'].items():assert report[key]==value
 expected=report['proposed_followup_configurations'];assert p['all_configurations_for_ranking']==expected
 new=[c for c in expected if c['candidate_id']!='d000'];assert p['selected_configurations']==new and 1<=len(new)<=2
 assert ENABLED['candidate_ids']==[q['candidate_id'] for q in new]
 assert len(p['reference_controls'])==3 and {r['seed'] for r in p['reference_controls']}=={2,3,4}
 ref_h=next(c['hyperparameters'] for c in expected if c['candidate_id']=='d000')
 for r in p['reference_controls']:
  original=read(r['source_result_path']);assert sha(r['source_result_path'])==r['source_result_SHA256']
  assert r['candidate_id']==original['candidate_id']=='d000' and r['seed']==original['seed'] and r['hyperparameters']==original['hyperparameters']==ref_h
  assert r['metrics']==original['metrics'] and r['finite']==original['finite'] and r['read_only_previously_measured_reference'] and r['new_model_fitted'] is False and r['new_GPU_charge']==0
 selector();return p
def cell(candidate='r000',seed=2,phase='followup'):
 validate_protocol();assert candidate in ENABLED['candidate_ids'] and type(seed) is int and seed in (2,3,4) and phase=='followup'
 return dict(experiment='followup097_'+candidate,fold=0,method='pg',seed=seed)
def proposal_path(cid):assert cid in ENABLED['candidate_ids'];return STUDY/'proposals'/(cid+'.json')
def validate_proposal(q):
 p=validate_protocol();assert q['id'] in ENABLED['candidate_ids'] and q['native_version']=='0.9.7' and q['endpoint']==20000 and q['protocol_SHA256']==sha(PROTOCOL)
 assert q['hyperparameters']==next(c['hyperparameters'] for c in p['selected_configurations'] if c['candidate_id']==q['id'])
 assert q['fold']==q['initial_completed_updates']==0 and q['selection_seeds']==[0,1] and q['training_seeds']==[2,3,4]
 assert q['from_scratch'] and q['source_checkpoint'] is None and not q['heldout_results_used']
 assert set(q['hyperparameters'])=={'generator_lr','critic_lr','particle_lr','reg_coeff','reg_kappa','prior_reg'} and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in q['hyperparameters'].values())
 return q['hyperparameters']
def protocol_paths(auth):
 p=read(PROTOCOL)
 return [PROTOCOL,PLAN,MODEL_PROOF,HPO_PROOF,NATIVE_PROOF,ENGINE_PROOF,BASELINE_PROOF,WINDOW,RULE,SELECTOR,proposal_path(auth['candidate_id']),ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json',*[Path(x) for x in p['selection_evidence_SHA256']]]
def validate_authorization(auth,*,require_runtime_proof=True):
 p=validate_protocol();v=auth['cell'];assert v==cell(auth['candidate_id'],v['seed'],auth['phase'])
 assert auth['released_hyperparameters']==validate_proposal(read(proposal_path(auth['candidate_id']))) and auth['variant_config']==CFG
 assert auth['proposal_SHA256']==sha(proposal_path(auth['candidate_id'])) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['operation']=='followup097_cell' and auth['metric_role']=='calibration_selection' and auth['role']=='alz_core_pg'
 assert auth['fixed_endpoint']==auth['new_model_updates']==20000 and auth['initial_completed_updates']==0
 assert auth['is_continuation'] is False and auth['reuse_completed_model'] is False and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and read(PLAN)==ENABLED
 assert auth['particlegan_version']=='0.9.7' and auth['penalty_options']=={'input_domain':'continuous'}
 assert not auth['cpu_fixture'] and not auth['independent_current_device_review_claimed'] and auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and auth['dynamic_shared_gpu0'] is False
 assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths(auth):assert pins[str(path)]==sha(path)
 assert read(MODEL_PROOF)['status']=='PASSED_ACTUAL_RELEASED_PARAMETER_SYNTHETIC_CPU_QUALIFICATION' and read(MODEL_PROOF)['native_version']=='0.9.6'
 assert read(HPO_PROOF)['native_update_byte_identical'] and not read(HPO_PROOF)['CUDA_initialized']
 for name in ('released_update.py','released_adapter.py'):assert (MODEL_DIR/name).read_bytes()==(ROOT/'analysis/released_tuning_v096'/name).read_bytes()
 eq=read(ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json');assert eq['all_core_runtime_files_byte_identical'] and eq['dependency_requirements_unchanged']
 baseline=read(BASELINE_PROOF);assert baseline['status']=='ALL_FOUR_MATCHED_FRESH_SEED_RESAMPLING_CONTROLS_CPU_VERIFIED' and set(baseline['metrics_by_seed'])=={'1','2','3','4'} and not baseline['CUDA_initialized']
 for evidence in (read(MODEL_PROOF)['source_SHA256'],read(HPO_PROOF)['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256'],baseline['source_SHA256'],baseline['files_SHA256']):
  for path,pin in evidence.items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='NATIVE097_MATCHED_SEED234_FOLLOWUP_SCOPE_ACCOUNTING_CPU_PASS' and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
