"""Confined controller tests plus synthetic selection scope; never a deployment."""
import argparse,datetime,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as s
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=s.STUDY/'prepared_CPU_qualification.json');args=parser.parse_args()
 assert args.output.resolve().parent==s.STUDY.resolve() and not args.output.exists()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not s.read(s.PLAN)['enabled']
 assert s.read(s.PROTOCOL)['status']=='PREPARED_NOT_FROZEN_PG_SELECTION_AND_CONFIRMATION_PENDING'
 results=[];count=0
 for name in ('test_integration.py','test_scope_prepared.py'):
  cmd=[sys.executable,'-I','-S','-B','-c','import sys,runpy;sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=[p];runpy.run_path(p,run_name="__main__")',str(HERE),str(HERE/name)]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=240);print(p.stderr)
  results.append(dict(file=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,name
  count+=int(re.search(r'Ran (\d+) tests?',p.stderr).group(1))
 assert count==23 and 'torch' not in sys.modules
 paths=[*HERE.glob('*.py'),s.PROTOCOL,s.PLAN,HERE/'policy.prepared.json',*s.PROOFS]
 q=dict(status='PREPARED_ACCOUNTING_AND_SYNTHETIC_SELECTION_SCOPE_CPU_TESTS_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_tests=count,synthetic_frozen_selection_fixture_all45_cells_checked=True,actual_frozen_PG_candidate_qualified=False,
        actual_unfrozen_candidate_admission_refused=True,actual_CFGen_role_confined_controller_settlement_test=True,
        real_GPU_allocation_created=False,CUDA_initialized=False,plan_enabled=False,live_runtime_deployment_qualified=False,
        tests=results,source_SHA256={str(p):b.sha(p) for p in paths},
        remaining='Freeze the actual confirmed PG candidate, verify its evidence and selected native model, rerun actual all45-cell authorization/source qualification, review current shared resource forecast and exact source, then enable. This prepared proof cannot satisfy the deployment runtime-proof gate.')
 out=args.output;out.write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps({k:q[k] for k in ('status','actual_tests','actual_frozen_PG_candidate_qualified','live_runtime_deployment_qualified')}))
if __name__=='__main__':main()
