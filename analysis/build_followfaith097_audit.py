"""Generate compositional independent audit while preserving the original audit."""
import ast
from pathlib import Path
ROOT=Path('/ml2/alz');dest=ROOT/'analysis/audit_followfaith097.py'
def main():
    assert not dest.exists()
    text=(ROOT/'analysis/audit_localfaith097.py').read_text()
    text=text.replace('localfaith097','followfaith097').replace('lf097_','uf097_').replace('LOCAL_COORDINATE','FOLLOWUP_COORDINATE')
    start=text.index(" baseline=read(protocol['baseline_audit_path'])")
    end=text.index(' summary=selector.summarize_local',start)
    text=text[:start]+''' upstream=read(protocol['baseline_audit_path']);baseline=read(protocol['upstream_control_audit_path'])
 assert upstream['status']=='INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS' and upstream['required_attempts']==16 and upstream['required_new_attempts']==12
 assert baseline['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS'
 for old in (baseline,upstream):
  assert old['formulation']==FORMULATION and old['activation_actually_reverted'] and old['owned_processes_and_leases_absent'] and not old['heldout_values_read'] and old['all_new_models_from_update_zero'] and not old['CUDA_initialized']
  assert old['model_parity_proof_path']==str(MODEL_PROOF) and old['model_parity_proof_SHA256']==sha(MODEL_PROOF)
  for f,pin in old['source_SHA256'].items():assert sha(f)==pin
 combined={**baseline['result_SHA256'],**upstream['result_SHA256']};assert len(combined)==16 and combined==protocol['reference_result_SHA256']
 for control in protocol['reference_controls']:
  original=read(control['source_result_path']);assert sha(control['source_result_path'])==control['source_result_SHA256']==combined[control['attempt']]
  assert control==dict(original,read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0,source_result_path=control['source_result_path'],source_result_SHA256=control['source_result_SHA256'])
  for f,pin in original['files_SHA256'].items():assert sha(f)==pin;hashed+=1
''' + text[end:]
    text=text.replace('required_attempts=len(expected)+4','required_attempts=len(expected)+16')
    text=text.replace('previously_measured_control_attempts=4','previously_measured_control_attempts=16')
    text=text.replace("Path(protocol['baseline_audit_path'])]","Path(protocol['baseline_audit_path']),Path(protocol['upstream_control_audit_path'])]")
    text=text.replace("limitation='Adaptive paired coordinate selection on the same four calibration donors, including four explicitly previously measured baseline/control endpoints. Fresh seed2..4 confirmation and45new external models remain.'",
        "inherited_independent_calibration_replica_rescores=baseline['calibration_replicates_independently_rescored']+upstream['calibration_replicates_independently_rescored'],upstream_selection_audit_path=protocol['baseline_audit_path'],limitation='Compositional follow-up audit: new models rescored here;16 previous endpoints rehashed against original closed independent audits. Adaptive selection remains on four calibration donors. Freshseed2..4 confirmation and all45new external models remain.'")
    ast.parse(text);dest.write_text(text);print(dest)
if __name__=='__main__':main()
