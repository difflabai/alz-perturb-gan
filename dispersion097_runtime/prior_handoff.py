"""Hash-only handoff after recovered benchmark AND all six scalar-LR fits."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.append(str(ROOT/'analysis'))
import scalarcal097_recovered_handoff as recovered
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def committed_sha(root,path):
 relative=Path(path).relative_to(root)
 data=subprocess.check_output(['git','show','HEAD:'+relative.as_posix()],cwd=root)
 return hashlib.sha256(data).hexdigest()
def paths(root=ROOT):
 root=Path(root)
 return [*recovered.paths(root),root/'analysis/scalarcal097/matrix_complete.json',root/'analysis/scalarcal097/comparison.json',root/'artifacts/scalarcal097-after-recovered-science-v5/completed.json']
def ready(root=ROOT):
 root=Path(root);pins=recovered.ready(root)
 matrix,report,complete=paths(root)[-3:];assert all(p.is_file() for p in (matrix,report,complete))
 q=read(matrix);marker=read(complete)
 expected={f'CAL_GPU0_V1_scalarcal097_{arm}_f0_pg_s{seed}' for seed in (5,6,7) for arm in ('fixed','adaptive')}
 assert q['status']=='ALL_SIX_FRESH_SCALAR_LR_NATIVE_MODELS_AND_CALIBRATION_AUDITS_COMPLETE'
 assert q['matrix_complete'] and q['complete_cells']==q['required_cells']==6 and set(q['result_SHA256'])==expected
 assert q['all_new_training_from_scratch'] and q['initial_completed_updates']==0 and q['calibration_only']
 extra=[]
 for key,pin in q['result_SHA256'].items():
  path=root/'analysis/scalarcal097/results'/(key+'.json');assert sha(path)==pin;extra.append(path)
 assert marker['status']=='SCALARCAL097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS'
 assert marker['report_path']==str(report)
 assert sha(report)==committed_sha(root,report),'Prior scalar report differs from its completed committed artifact'
 assert not read(root/'scalarcal097_runtime/plan.json')['enabled']
 for registry in (root/'artifacts').glob('gpu*-scalarcal097-20261107/live_registry/registry.json'):assert not read(registry)['jobs']
 proof_path=root/'analysis/scalarcal097/runtime_CPU_qualification_v11.json';proof=read(proof_path)
 assert proof['status']=='SCALARCAL097_SIX_FRESH_CELLS_PREPARED_SCOPE_ACCOUNTING_CPU_PASS' and proof['actual_tests']==39 and proof['native_cell_device_bindings']==12
 for path,pin in proof['source_SHA256'].items():assert sha(path)==pin
 pins.update({str(p):sha(p) for p in (matrix,report,complete,proof_path,*extra)})
 # Quality-bearing result/report bytes are hashed, never decoded here.
 return pins
