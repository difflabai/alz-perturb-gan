"""Same frozen science, ninety bindings, actual confined guard/drain tests."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled'] and not s.RUNTIME_PROOF.exists()
 paths=[*HERE.glob('*.py'),*[HERE/f'policy.gpu{i}.prepared.json' for i in (0,1)],*s.protocol_paths(s.protocol()),*list((s.ROOT/'benchmark097_dual_runtime').glob('*.py')),s.ROOT/'analysis/run_benchmark097_dual.py',s.ROOT/'analysis/aggregate_benchmark097_dual.py',s.ROOT/'analysis/score_calibration_excluded53_dual.py']
 pins={str(p):b.sha(p) for p in paths if p!=s.PLAN};results=[];count=0
 for name in ('test_integration.py','test_scope.py','test_dispatch.py','test_handoff.py','test_pair.py','test_numerics.py'):
  code='import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")'
  p=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(HERE),str(HERE/name)],capture_output=True,text=True,timeout=300)
  print(p.stderr);results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
 assert 'torch' not in sys.modules
 for path,pin in pins.items():assert b.sha(path)==pin,path
 q=dict(status='EXTERNAL097_SAME45_BALANCED_RESOURCE_SCOPE_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tests=count,native_cell_device_bindings=90,actual_all90_native_preimport_source_closures_checked=True,actual_CPU_dispatcher_signal_isolation_test=True,actual_two_concurrent_confined_CPU_guard_workers=True,actual_driver_failure_drain_and_observation_timeout_tests=True,two_physical_policy_profiles=True,scientific_worker_and_sampling_and_native_audit_byte_identical=True,numerical57_and_prespecified53_analysis_unchanged=True,all_completed_models_retained_and_never_refitted=True,all_unstarted_models_require_update0=True,canonical_Controller_unchanged=True,tests=results,source_SHA256=pins,GPU_allocation_created=False,CUDA_initialized=False,plan_enabled=False,actual_handoff_not_yet_executed=True,operator_source_review_and_exact_current_capacity_still_required=True,post_handoff_real_source_closure_gate_still_required=True,full_goal_complete=False,SOTA_claim=False)
 s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k] for k in ('status','actual_tests','native_cell_device_bindings','GPU_allocation_created')}))
if __name__=='__main__':main()
