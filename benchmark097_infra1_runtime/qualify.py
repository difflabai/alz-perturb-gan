"""Actual isolated guard accounting tests plus four replacement preimport closures."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled'] and not s.RUNTIME_PROOF.exists()
 paths=[*HERE.glob('*.py'),*[HERE/f'policy.gpu{i}.prepared.json' for i in (0,1)],*s.protocol_paths(s.protocol()),s.ROOT/'analysis/run_benchmark097_infra1.py',s.ROOT/'analysis/run_benchmark097_infra1_pipeline.py',s.ROOT/'analysis/benchmark097_recovered_lineage.py',s.ROOT/'analysis/aggregate_benchmark097_recovered.py']
 pins={str(p):b.sha(p) for p in paths if p!=s.PLAN};results=[];count=0
 for name in ('test_integration.py','test_scope.py'):
  code='import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")'
  p=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(HERE),str(HERE/name)],capture_output=True,text=True,timeout=240)
  print(p.stderr,flush=True);results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
 for p,pin in pins.items():assert b.sha(p)==pin,p
 q=dict(status='ONE_FRESH_INFRASTRUCTURE_REPLACEMENT_NATIVE_SCOPE_CPU_PASS',scope_revision=2,actual_tests=count,native_cells=2,native_cell_device_bindings=4,actual_four_native_preimport_source_closures=True,actual_confined_guard_worker_and_canonical_settlement=True,scientific_worker_native_auditor_backend_byte_identical=True,unchanged_physical_bounds=True,source_SHA256=pins,tests=results,GPU_allocation_created=False,CUDA_initialized=False,all_real_replacements_still_pending=True,original43_successes_and2_failures_must_close_before_activation=True,full_goal_complete=False,SOTA_claim=False)
 s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k] for k in ('status','actual_tests','native_cell_device_bindings','GPU_allocation_created')}))
if __name__=='__main__':main()
