"""Hash-only scientific handoff; target metric reports are opaque here."""
import hashlib,json
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def paths(root):
 root=Path(root);return [root/'analysis/benchmark097/matrix_complete.json',root/'analysis/benchmark097/comparison.json',root/'analysis/benchmark097_calibration_excluded53_report.json',root/'analysis/benchmark097_infra1/scientific_reports_completed.json',root/'artifacts/benchmark097-infra1-pipeline/completed.json']
def ready(root):
 root=Path(root);matrix,primary,sensitivity,barrier,child=paths(root);assert all(p.is_file() for p in (matrix,primary,sensitivity,barrier,child))
 m=read(matrix);q=read(barrier);source=root/'analysis/run_benchmark097_infra1_pipeline.py';proof_path=Path(q['runtime_qualification_path'])
 assert proof_path==root/'analysis/benchmark097_infra1/runtime_CPU_qualification_v6.json'
 proof=read(proof_path)
 assert m['matrix_complete'] and m['complete_cells']==m['required_cells']==45 and m['physical_attempts']==48 and len(m['result_SHA256'])==45
 expected={f'f{f}_{method}_s{seed}':f"CAL_GPU0_V1_{'external097infra1' if (f,method,seed) in ((2,'pg',1),(2,'cfgen',1),(1,'cfgen',2)) else 'external097'}_f{f}_{method}_s{seed}" for f in range(3) for seed in range(5) for method in ('pg','scvi','cfgen')}
 assert m['logical_to_physical_attempt']==expected and set(m['result_SHA256'])==set(expected.values())
 assert m['all_new_training_from_scratch'] and not m['failed_weights_resumed'] and m['scientific_protocol_unchanged']
 failed=['CAL_GPU0_V1_external097_f1_cfgen_s2','CAL_GPU0_V1_external097_f2_cfgen_s1','CAL_GPU0_V1_external097_f2_pg_s1'];assert m['infrastructure_failed_attempts']==failed and set(m['failed_result_SHA256'])==set(failed)
 extra=[]
 for key,pin in m['failed_result_SHA256'].items():
  path=root/'analysis/benchmark097/results'/(key+'.json');assert sha(path)==pin;extra.append(path)
 assert q['status']=='RECOVERED_LOGICAL45_ORIGINAL57_PRIMARY_AND_PRESPECIFIED53_COMPLETE' and q['complete_fresh_models']==45 and q['physical_attempts']==48 and q['failed_attempts']==failed
 assert q['original57_primary_preserved'] and q['prespecified53_complete'] and q['new_generated_counts_for53']==0
 assert q['source_pipeline_path']==str(source) and q['source_pipeline_SHA256']==proof['source_SHA256'][str(source)]==sha(source)
 assert proof['status']=='ONE_FRESH_INFRASTRUCTURE_REPLACEMENT_NATIVE_SCOPE_CPU_PASS' and proof['actual_tests']==22 and proof['native_cell_device_bindings']==6
 assert q['runtime_qualification_path']==str(proof_path) and q['runtime_qualification_SHA256']==sha(proof_path)
 assert q['report_SHA256']=={str(p):sha(p) for p in (matrix,primary,sensitivity)}
 assert read(child)['status']=='RECOVERED45_57_53_READY_FRESH_SCALAR_CALIBRATION_AND_FINAL_AUDIT_REMAIN'
 for name in ('benchmark097_dual_runtime','benchmark097_balanced_runtime','benchmark097_infra1_runtime'):assert not read(root/name/'plan.json')['enabled']
 for pattern in ('gpu*-benchmark097-dual-20261107','gpu*-benchmark097-balanced-20261107','gpu*-benchmark097-infra1'):
  for registry in (root/'artifacts').glob(pattern+'/live_registry/registry.json'):assert not read(registry)['jobs']
 return {str(p):sha(p) for p in (matrix,primary,sensitivity,barrier,child,source,proof_path,*extra)}
