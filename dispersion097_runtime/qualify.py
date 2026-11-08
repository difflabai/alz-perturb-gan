"""Confined real Controller/guard fixtures and all12 prepared source closures."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled'] and not s.RUNTIME_PROOF.exists()
 pins={str(p):b.sha(p) for p in [*HERE.glob('*.py'),*[HERE/f'policy.gpu{i}.prepared.json' for i in (0,1)],*s.protocol_paths()] if p!=s.PLAN}
 records=[];count=0
 for name in ('test_integration.py','test_scope.py','test_driver.py','test_pair.py'):
  code='import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")'
  result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(HERE),str(HERE/name)],capture_output=True,text=True,timeout=300)
  print(result.stderr,flush=True);records.append(dict(file=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
  assert result.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',result.stderr).group(1))
 assert 'torch' not in sys.modules
 for path,pin in pins.items():assert b.sha(path)==pin,path
 q=dict(status='DISPERSION097_SIX_FRESH_CELLS_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tests=count,native_cells_checked=6,native_cell_device_bindings=12,tests=records,source_SHA256=pins,unchanged_canonical_Controller=True,actual_confined_owner_worker_resource_guard_and_settlement=True,actual_two_simultaneous_confined_physical_guard_chains=True,CUDA_initialized=False,GPU_allocation_created=False,plan_enabled=False,all45_and_both_reports_and_six_LR_required=True,actual_upstream_handoff_not_yet_met=True,real_theta_fits=0,full_goal_complete=False,SOTA_claim=False)
 s.RUNTIME_PROOF.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k] for k in ('status','actual_tests','native_cells_checked','GPU_allocation_created')}),flush=True)
if __name__=='__main__':main()
