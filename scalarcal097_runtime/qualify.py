"""Prepared six-cell scope and real confined Controller/guard qualification."""
import datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled'] and not s.RUNTIME_PROOF.exists()
 pins={str(p):b.sha(p) for p in [*HERE.glob('*.py'),*[HERE/f'policy.gpu{i}.prepared.json' for i in (0,1)],*s.protocol_paths(),s.ROOT/'analysis/run_scalarcal097.py',s.ROOT/'analysis/after_benchmark097_scalarcal.py',s.ROOT/'analysis/aggregate_scalarcal097.py'] if p!=s.PLAN}
 results=[];count=0
 for name in ('test_integration.py','test_scope.py','test_driver.py','test_pair.py'):
  code='import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")'
  result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(HERE),str(HERE/name)],capture_output=True,text=True,timeout=240)
  print(result.stderr);results.append(dict(file=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',result.stderr).group(1))
 assert 'torch' not in sys.modules
 for path,pin in pins.items():assert b.sha(path)==pin
 report=dict(status='SCALARCAL097_SIX_FRESH_CELLS_PREPARED_SCOPE_ACCOUNTING_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tests=count,native_cells_checked=6,fullwidth_endpoint_and_scalar_CPU_parity_proof_SHA256=b.sha(s.ENDPOINT_PROOF),tests=results,source_SHA256=pins,unchanged_canonical_Controller=True,actual_confined_owner_worker_resource_guard_and_settlement=True,actual_two_simultaneous_confined_physical_guard_chains=True,CUDA_initialized=False,GPU_allocation_created=False,plan_enabled=False,prior45_and_both_CPU_reports_required=True,actual_upstream_handoff_not_yet_met=True,live_review_and_current_capacity_still_required=True,real_adaptive_fits=0,SOTA_claim=False,full_goal_complete=False)
 s.RUNTIME_PROOF.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','actual_tests','native_cells_checked','GPU_allocation_created')}))
if __name__=='__main__':main()
