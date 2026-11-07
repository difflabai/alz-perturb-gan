"""Twenty paired-seed fresh20k fits; complete reference audit precedes admission."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/refinement097'
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json';MODEL_DIR=STUDY/'model'
MODEL_PROOF=ROOT/'analysis/released_tuning_v096/CPU_qualification.json';HPO_PROOF=ROOT/'analysis/hpo500/model_CPU_qualification.json'
NATIVE_PROOF=STUDY/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
DESIGN=ROOT/'analysis/distribution097_conditional_next_search_design_20261107.json';SELECTOR=ROOT/'analysis/refinement097_selection.py'
SCALAR_PROOF=ROOT/'analysis/distribution097_conditional_continuation_CPU_model_qualification_20261107.json'
BASELINE_PROOF=STUDY/'resampling_controls_CPU.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
IDS=[f'r{i:03}' for i in range(10)]
ENABLED=dict(enabled=True,experiment='refinement097',native_version='0.9.7',formulation='native097_bispf_optimized_nonEMA',endpoint=20000,
  selection_seeds=[0,1],candidate_ids=IDS,owned_GPU_GiB=2,calibration_only=True,all_training_from_scratch=True)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_CONDITIONAL_PAIRED_SEED_FULL20K_NATIVE097_REFINEMENT','Complete original reference confirmation audit required before refinement admission'
 assert p['native_version']=='0.9.7' and p['endpoint']==20000 and p['selection_seeds']==[0,1] and p['maximum_new_fits']==20
 assert p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only'] and p['metric_role']=='calibration_selection'
 assert not p['automatic_promotion'] and not p['SOTA_claim'] and not p['full_goal_complete']
 assert p['candidates']==read(DESIGN)['candidates'] and [c['id'] for c in p['candidates']]==IDS
 for path,pin in p['design_evidence_SHA256'].items():assert sha(path)==pin
 audit=read(p['original_reference_confirmation_audit_path']);decision=read(p['original_reference_next_decision_path'])
 assert audit['status']=='INDEPENDENT_FRESH_NATIVE097_MATCHED_SEED_CONFIRMATION_AUDIT_PASS' and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero']
 assert audit['activation_actually_reverted'] and not audit['heldout_values_read'] and not audit['SOTA_claim']
 assert decision['calibration_refinement_triggered'] and decision['complete_audit_SHA256']==sha(p['original_reference_confirmation_audit_path'])
 assert not decision['benchmark_candidate_frozen'] and not decision['GPU_allocation_created']
 scalar=read(SCALAR_PROOF);assert scalar['actual_full2000gene_optimizer_pairs']==40 and scalar['actual_full_configuration_seed_cases']==20 and not scalar['CUDA_initialized']
 for candidate in p['candidates']:
  assert candidate['training_seeds']==[0,1] and candidate['endpoint']==20000 and candidate['from_scratch'] and candidate['source_checkpoint'] is None
  changed=[k for k,v in candidate['hyperparameters'].items() if v!=read(DESIGN)['reference_hyperparameters'][k]];assert changed==[candidate['changed_field']]
  assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in candidate['hyperparameters'].values())
  for seed in (0,1):
   entry=next(q for q in scalar['rows'] if q['candidate_id']==candidate['id'] and q['seed']==seed)
   assert entry['hyperparameters']==candidate['hyperparameters'] and entry['exact_next_pair_state_and_diagnostics_at500_and20000_limits']
 return p
def cell(candidate='r000',seed=0,phase='search'):
 validate_protocol();assert candidate in IDS and type(seed) is int and seed in (0,1) and phase=='search'
 return dict(experiment='refine097_'+candidate,fold=0,method='pg',seed=seed)
def proposal_path(cid):assert cid in IDS;return STUDY/'proposals'/(cid+'.json')
def validate_proposal(q):
 p=validate_protocol();assert q['id'] in IDS and q['native_version']=='0.9.7' and q['endpoint']==20000 and q['protocol_SHA256']==sha(PROTOCOL)
 assert q['hyperparameters']==next(c['hyperparameters'] for c in p['candidates'] if c['id']==q['id'])
 assert q['fold']==q['initial_completed_updates']==0 and q['training_seeds']==[0,1] and q['from_scratch'] and q['source_checkpoint'] is None and not q['heldout_results_used']
 return q['hyperparameters']
def protocol_paths(auth):
 p=read(PROTOCOL)
 return [PROTOCOL,PLAN,MODEL_PROOF,HPO_PROOF,NATIVE_PROOF,ENGINE_PROOF,BASELINE_PROOF,SCALAR_PROOF,WINDOW,DESIGN,SELECTOR,proposal_path(auth['candidate_id']),
   ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json',*[Path(x) for x in p['design_evidence_SHA256']]]
def validate_authorization(auth,*,require_runtime_proof=True):
 p=validate_protocol();v=auth['cell'];assert v==cell(auth['candidate_id'],v['seed'],auth['phase'])
 assert auth['released_hyperparameters']==validate_proposal(read(proposal_path(auth['candidate_id']))) and auth['variant_config']==CFG
 assert auth['proposal_SHA256']==sha(proposal_path(auth['candidate_id'])) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['operation']=='refinement097_cell' and auth['metric_role']=='calibration_selection' and auth['role']=='alz_core_pg'
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
 baseline=read(BASELINE_PROOF);assert baseline['status']=='TWO_MATCHED_SELECTION_SEED_RESAMPLING_CONTROLS_CPU_VERIFIED' and set(baseline['metrics_by_seed'])=={'0','1'} and not baseline['CUDA_initialized']
 for evidence in (read(MODEL_PROOF)['source_SHA256'],read(HPO_PROOF)['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256'],baseline['source_SHA256'],baseline['files_SHA256'],read(SCALAR_PROOF)['source_SHA256']):
  for path,pin in evidence.items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='NATIVE097_PAIRED_REFINEMENT_SCOPE_ACCOUNTING_CPU_PASS' and q['actual_cells_checked']==20 and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
